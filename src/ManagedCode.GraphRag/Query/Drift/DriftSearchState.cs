using System.Text.Json;
using Microsoft.ML.Tokenizers;

namespace GraphRag.Query.Drift;

internal sealed class DriftSearchState(string question, DriftSearchLimits limits, Tokenizer tokenizer)
{
    private readonly Dictionary<string, (DriftSearchEvidence Value, int Tokens)> evidence = new(StringComparer.Ordinal);
    private readonly HashSet<string> queuedQuestions = new(StringComparer.OrdinalIgnoreCase) { Normalize(question) };
    private readonly Dictionary<Uri, string?> sourceRevisions = [];
    private readonly List<PendingQuestion> pending = [];
    private readonly List<string> questions = [question];
    private int tokens;

    internal int ModelCalls { get; set; }
    internal bool IsBudgetLimited { get; private set; }
    internal DriftSearchReasoning? Primer { get; private set; }
    internal int EvidenceCount => evidence.Count;
    internal DriftSearchEvidence[] Evidence => evidence.Values.Select(static item => item.Value).ToArray();

    internal DriftSearchEvidence[] AcceptEvidence(DriftSearchEvidence[] values, int requestLimit, int tokenLimit)
    {
        if (values.Length > requestLimit)
        {
            throw new InvalidDataException("The DRIFT retriever exceeded its result limit.");
        }
        var accepted = new List<DriftSearchEvidence>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var localTokens = 0;
        foreach (var value in values)
        {
            ValidateEvidence(value);
            if (sourceRevisions.TryGetValue(value.Source, out var revision) && revision != value.Revision)
            {
                throw new InvalidDataException("A DRIFT source revision changed during the search.");
            }
            sourceRevisions.TryAdd(value.Source, value.Revision);
            if (evidence.TryGetValue(value.Id, out var existing) && existing.Value != value)
            {
                throw new InvalidDataException("DRIFT evidence changed within one search revision.");
            }
            if (!seen.Add(value.Id))
            { continue; }
            var size = tokenizer.CountTokens(JsonSerializer.Serialize(value));
            var additionalTokens = evidence.ContainsKey(value.Id) ? 0 : size;
            if (size > tokenLimit - localTokens || additionalTokens > limits.TotalTokens - tokens)
            {
                IsBudgetLimited = true;
                continue;
            }
            if (additionalTokens != 0)
            { evidence.Add(value.Id, (value, size)); tokens += size; }
            localTokens += size;
            accepted.Add(value);
        }
        return accepted.ToArray();
    }

    internal void AddReasoning(DriftSearchReasoning reasoning, int depth)
    {
        Primer ??= reasoning;
        if (reasoning.IsComplete)
        { return; }
        foreach (var candidate in reasoning.FollowUpQuestions)
        {
            var normalized = Normalize(candidate);
            if (!queuedQuestions.Add(normalized))
            { continue; }
            if (depth >= limits.Depth || tokenizer.CountTokens(normalized) > limits.TotalTokens)
            {
                IsBudgetLimited = true;
                continue;
            }
            pending.Add(new(normalized, depth + 1, 1 - reasoning.Confidence));
        }
    }

    internal PendingQuestion[] TakeQuestions()
    {
        var available = Math.Min(limits.ModelCalls - ModelCalls - 1, limits.Followups - questions.Count + 1);
        var selected = pending.OrderByDescending(static item => item.Priority)
            .Take(Math.Min(limits.Concurrency, Math.Max(0, available))).ToArray();
        foreach (var value in selected)
        { pending.Remove(value); questions.Add(value.Query); }
        if (selected.Length == 0 && pending.Count != 0)
        { IsBudgetLimited = true; }
        return selected;
    }

    internal DriftSearchResult Complete(DriftSearchReasoning? reasoning)
    {
        var status = reasoning is null ? (IsBudgetLimited ? DriftSearchStatus.BudgetExhausted : DriftSearchStatus.NoEvidence) :
            IsBudgetLimited ? DriftSearchStatus.BudgetExhausted :
            reasoning.IsComplete ? DriftSearchStatus.Complete : DriftSearchStatus.InsufficientEvidence;
        return new(status, reasoning?.Answer ?? string.Empty, reasoning?.CitationIds ?? [],
            Evidence, questions.ToArray(), ModelCalls, IsBudgetLimited);
    }

    private static void ValidateEvidence(DriftSearchEvidence value)
    {
        if (string.IsNullOrWhiteSpace(value.Id) || string.IsNullOrWhiteSpace(value.Content) ||
            value.Source is null || !value.Source.IsAbsoluteUri)
        {
            throw new InvalidDataException("The DRIFT retriever returned invalid evidence.");
        }
    }

    private static string Normalize(string value) =>
        string.Join(' ', value.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));

    internal sealed record PendingQuestion(string Query, int Depth, double Priority);
}
