using System.Runtime.CompilerServices;
using GraphRag.Config;
using GraphRag.Tokenization;
using Microsoft.Extensions.AI;

namespace GraphRag.Query.Drift;

public sealed class DriftSearchEngine
{
    private readonly DriftSearchRetriever retrieve;
    private readonly DriftSearchReasoner reasoner;
    private readonly DriftSearchLimits limits;

    public DriftSearchEngine(IChatClient chatClient, DriftSearchRetriever retrieve,
        DriftSearchConfig configuration, ChatOptions? chatOptions = null)
    {
        ArgumentNullException.ThrowIfNull(chatClient);
        ArgumentNullException.ThrowIfNull(retrieve);
        limits = DriftSearchLimits.From(configuration);
        this.retrieve = retrieve;
        reasoner = new(chatClient, chatOptions, limits.OutputTokens);
    }

    public async IAsyncEnumerable<DriftSearchUpdate> SearchAsync(string question,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(question);
        cancellationToken.ThrowIfCancellationRequested();
        var tokenizer = TokenizerRegistry.GetTokenizer();
        if (tokenizer.CountTokens(question) > limits.TotalTokens)
        {
            throw new ArgumentException("The DRIFT question exceeds its input-token budget.", nameof(question));
        }
        var state = new DriftSearchState(question, limits, tokenizer);
        yield return new(DriftSearchPhase.Primer, question, 0, 0, 0);
        var context = state.AcceptEvidence(
            await RetrieveAsync(new(question, DriftSearchPhase.Primer, 0, limits.Results), cancellationToken),
            limits.Results, limits.PrimerTokens);
        if (context.Length != 0)
        {
            yield return new(DriftSearchPhase.Primer, question, 0, 0, state.EvidenceCount, EvidenceDelta: context);
            var primer = await reasoner.ReasonAsync(DriftSearchPrompts.Primer, question, context,
                state.IsBudgetLimited, cancellationToken);
            state.ModelCalls++;
            state.AddReasoning(primer, 0);
            yield return new(DriftSearchPhase.Primer, question, 0, state.ModelCalls, state.EvidenceCount,
                Finding: new(primer.Answer, primer.CitationIds, primer.Confidence));
            await foreach (var update in RefineAsync(state, cancellationToken))
            { yield return update; }
        }
        var final = await CompleteAsync(question, state, cancellationToken);
        yield return new(DriftSearchPhase.Complete, question, 0, state.ModelCalls, state.EvidenceCount, state.Complete(final));
    }

    private async IAsyncEnumerable<DriftSearchUpdate> RefineAsync(DriftSearchState state,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        while (true)
        {
            var questions = state.TakeQuestions();
            if (questions.Length == 0)
            { yield break; }
            foreach (var question in questions)
            {
                yield return new(DriftSearchPhase.LocalSearch, question.Query, question.Depth,
                    state.ModelCalls, state.EvidenceCount);
            }
            var retrieved = await Task.WhenAll(questions.Select(question => RetrieveAsync(
                new(question.Query, DriftSearchPhase.LocalSearch, question.Depth, limits.Results), cancellationToken)));
            var contexts = retrieved.Select(values => state.AcceptEvidence(values, limits.Results, limits.LocalTokens)).ToArray();
            var requests = questions.Select((question, index) => (question, context: contexts[index]))
                .Where(static item => item.context.Length != 0).ToArray();
            foreach (var request in requests)
            {
                yield return new(DriftSearchPhase.LocalSearch, request.question.Query, request.question.Depth,
                    state.ModelCalls, state.EvidenceCount, EvidenceDelta: request.context);
            }
            var reasoning = await Task.WhenAll(requests.Select(item => reasoner.ReasonAsync(
                DriftSearchPrompts.Local, item.question.Query, item.context, state.IsBudgetLimited, cancellationToken)));
            state.ModelCalls += reasoning.Length;
            for (var index = 0; index < reasoning.Length; index++)
            {
                state.AddReasoning(reasoning[index], requests[index].question.Depth);
                yield return new(DriftSearchPhase.LocalSearch, requests[index].question.Query, requests[index].question.Depth,
                    state.ModelCalls, state.EvidenceCount,
                    Finding: new(reasoning[index].Answer, reasoning[index].CitationIds, reasoning[index].Confidence));
            }
        }
    }

    private async Task<DriftSearchReasoning?> CompleteAsync(string question, DriftSearchState state,
        CancellationToken cancellationToken)
    {
        if (state.ModelCalls <= 1)
        { return state.Primer; }
        var final = await reasoner.ReasonAsync(DriftSearchPrompts.Reduce, question,
            state.Evidence, state.IsBudgetLimited, cancellationToken);
        state.ModelCalls++;
        return final;
    }

    private async Task<DriftSearchEvidence[]> RetrieveAsync(DriftSearchRequest request,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return await retrieve(request, cancellationToken)
            ?? throw new InvalidDataException("The DRIFT retriever returned no result collection.");
    }
}
