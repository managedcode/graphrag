namespace GraphRag.Query.Drift;

public enum DriftSearchPhase { Primer, LocalSearch, Complete }

public enum DriftSearchStatus { Complete, NoEvidence, InsufficientEvidence, BudgetExhausted }

public sealed record DriftSearchEvidence(string Id, string Title, string Content, Uri Source, string? Revision = null);

public sealed record DriftSearchRequest(string Query, DriftSearchPhase Phase, int Depth, int MaxResults);

public delegate Task<DriftSearchEvidence[]> DriftSearchRetriever(
    DriftSearchRequest request, CancellationToken cancellationToken);

public sealed record DriftSearchResult(
    DriftSearchStatus Status,
    string Answer,
    string[] CitationIds,
    DriftSearchEvidence[] Evidence,
    string[] Questions,
    int ModelCalls,
    bool IsBudgetLimited);

public sealed record DriftSearchUpdate(
    DriftSearchPhase Phase,
    string Query,
    int Depth,
    int ModelCalls,
    int EvidenceCount,
    DriftSearchResult? Result = null,
    DriftSearchEvidence[]? EvidenceDelta = null,
    DriftSearchFinding? Finding = null);

public sealed record DriftSearchFinding(string Answer, string[] CitationIds, double Confidence);

public sealed record DriftSearchReasoning
{
    public required string Answer { get; init; }
    public required string[] CitationIds { get; init; }
    public required string[] FollowUpQuestions { get; init; }
    public required bool IsComplete { get; init; }
    public required double Confidence { get; init; }
}
