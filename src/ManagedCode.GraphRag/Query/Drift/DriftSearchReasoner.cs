using System.Text.Json;
using Microsoft.Extensions.AI;

namespace GraphRag.Query.Drift;

internal sealed class DriftSearchReasoner(IChatClient client, ChatOptions? chatOptions, int? outputTokens)
{
    internal async Task<DriftSearchReasoning> ReasonAsync(string prompt, string question,
        DriftSearchEvidence[] evidence, bool budgetLimited, CancellationToken cancellationToken)
    {
        var options = chatOptions?.Clone() ?? new ChatOptions();
        options.Tools = null;
        options.ToolMode = null;
        if (outputTokens is { } maximum)
        {
            options.MaxOutputTokens = options.MaxOutputTokens is { } selected ? Math.Min(maximum, selected) : maximum;
        }
        var response = await client.GetResponseAsync<DriftSearchReasoning>(
            [
                new ChatMessage(ChatRole.System, prompt),
                new ChatMessage(ChatRole.User, JsonSerializer.Serialize(new ModelInput(question, evidence, budgetLimited)))
            ], options, useJsonSchemaResponseFormat: false, cancellationToken: cancellationToken);
        var result = response.Result;
        Validate(result, evidence);
        return result;
    }

    private static void Validate(DriftSearchReasoning value, DriftSearchEvidence[] evidence)
    {
        var identifiers = evidence.Select(static item => item.Id).ToHashSet(StringComparer.Ordinal);
        if (string.IsNullOrWhiteSpace(value.Answer) || value.CitationIds is null ||
            value.FollowUpQuestions is null || !double.IsFinite(value.Confidence) ||
            value.Confidence is < 0 or > 1 || value.CitationIds.Length == 0 ||
            value.CitationIds.Any(id => !identifiers.Contains(id)) ||
            value.FollowUpQuestions.Any(string.IsNullOrWhiteSpace) ||
            (value.IsComplete && value.FollowUpQuestions.Length != 0))
        {
            throw new InvalidDataException("The DRIFT model returned invalid reasoning or citations.");
        }
    }

    private sealed record ModelInput(string Question, DriftSearchEvidence[] Evidence, bool IsBudgetLimited);
}
