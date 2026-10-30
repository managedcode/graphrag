using System.Text.Json;
using GraphRag.Config;
using GraphRag.Graphs;
using GraphRag.Query.Drift;
using ManagedCode.GraphRag.Tests.Infrastructure;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;

namespace ManagedCode.GraphRag.Tests.Integration;

[Collection(nameof(GraphRagApplicationCollection))]
public sealed partial class DriftSearchIntegrationTests(GraphRagApplicationFixture fixture)
{
    [Theory]
    [InlineData("postgres")]
    [InlineData("neo4j")]
    [InlineData("janus")]
    public async Task Search_FollowsStoredRelationshipsAndSynthesizesCitedEvidence(string provider)
    {
        var (store, root, detail) = await SeedAsync(provider);
        using var client = CreateModel(root);
        var engine = new DriftSearchEngine(client,
            (request, token) => RetrieveAsync(store, root, request, token), Configuration());
        var updates = new List<DriftSearchUpdate>();
        await foreach (var update in engine.SearchAsync(root))
        { updates.Add(update); }

        var result = updates.Last().Result!;
        Assert.Equal(DriftSearchStatus.Complete, result.Status);
        Assert.Equal(3, result.ModelCalls);
        Assert.Equal(2, result.Questions.Length);
        Assert.Equal(2, result.Evidence.Length);
        Assert.Contains(detail, result.CitationIds);
        Assert.Contains("detail evidence", result.Answer);
        Assert.Contains(updates, update => update.Phase == DriftSearchPhase.LocalSearch);
        Assert.Contains(updates, update => update.ModelCalls == 0 && update.EvidenceDelta?.Any(item => item.Id == root) == true);
        Assert.Contains(updates, update => update.Result is null && update.EvidenceDelta?.Any(item => item.Id == detail) == true);
        Assert.Contains(updates, update => update.Finding?.CitationIds.Contains(detail) == true);
        Assert.All(result.CitationIds, id => Assert.Contains(result.Evidence, evidence => evidence.Id == id));
        Assert.Equal("detail evidence", (await ReadNodesAsync(store)).Single(node => node.Id == detail).Properties["content"]);
    }

    [Theory]
    [InlineData("postgres")]
    [InlineData("neo4j")]
    [InlineData("janus")]
    public async Task Search_StopsAtWorkBudgetAndReportsIncompleteCoverage(string provider)
    {
        var (store, root, _) = await SeedAsync(provider);
        var turn = 0;
        using var client = new TestChatClientFactory(messages =>
            ModelResponse(messages, ["more evidence " + ++turn], false, localCompletes: false)).CreateClient();
        var config = Configuration();
        config.MaximumModelCalls = 3;
        var result = await CompleteAsync(new DriftSearchEngine(client,
            (request, token) => RetrieveAsync(store, root, request, token), config), root);

        Assert.True(result.IsBudgetLimited);
        Assert.Equal(DriftSearchStatus.BudgetExhausted, result.Status);
        Assert.Equal(3, result.ModelCalls);
        Assert.Equal(2, result.Questions.Length);
        Assert.Contains(result.Evidence, evidence => evidence.Id == root);
        Assert.Contains(await ReadNodesAsync(store), node => node.Id == root);
    }

    [Theory]
    [InlineData("postgres")]
    [InlineData("neo4j")]
    [InlineData("janus")]
    public async Task Search_RejectsInventedCitationsWithoutChangingGraph(string provider)
    {
        var (store, root, detail) = await SeedAsync(provider);
        using var client = new TestChatClientFactory(_ => new ChatResponse(new ChatMessage(ChatRole.Assistant,
            JsonSerializer.Serialize(new DriftSearchReasoning
            {
                Answer = "Invented answer",
                CitationIds = ["unretrieved"],
                FollowUpQuestions = [],
                IsComplete = true,
                Confidence = 1
            })))).CreateClient();
        var engine = new DriftSearchEngine(client,
            (request, token) => RetrieveAsync(store, root, request, token), Configuration());

        await Assert.ThrowsAsync<InvalidDataException>(() => CompleteAsync(engine, root));
        var persisted = await ReadNodesAsync(store);
        Assert.Equal("root evidence", persisted.Single(node => node.Id == root).Properties["content"]);
        Assert.Equal("detail evidence", persisted.Single(node => node.Id == detail).Properties["content"]);
    }

    [Theory]
    [InlineData("postgres")]
    [InlineData("neo4j")]
    [InlineData("janus")]
    public async Task Search_CancellationLeavesStoredGraphAvailable(string provider)
    {
        var (store, root, detail) = await SeedAsync(provider);
        using var client = CreateModel(root);
        var engine = new DriftSearchEngine(client,
            (request, token) => RetrieveAsync(store, root, request, token), Configuration());
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => CompleteAsync(engine, root, new CancellationToken(true)));
        Assert.Contains(await ReadNodesAsync(store), node => node.Id == detail);
        var recovered = await CompleteAsync(engine, root);
        Assert.Equal(DriftSearchStatus.Complete, recovered.Status);
        Assert.Contains(detail, recovered.CitationIds);
    }

    private async Task<(IGraphStore Store, string Root, string Detail)> SeedAsync(string provider)
    {
        var store = fixture.Services.GetRequiredKeyedService<IGraphStore>(provider);
        await store.InitializeAsync();
        var root = Guid.CreateVersion7().ToString("N");
        var detail = Guid.CreateVersion7().ToString("N");
        var label = GraphStoreTestProviders.GetLabel(provider);
        await store.UpsertNodeAsync(root, label, new Dictionary<string, object?> { ["content"] = "root evidence" });
        await store.UpsertNodeAsync(detail, label, new Dictionary<string, object?> { ["content"] = "detail evidence" });
        await store.UpsertRelationshipAsync(root, detail, "related", new Dictionary<string, object?>());
        return (store, root, detail);
    }

    private static IChatClient CreateModel(string root) => new TestChatClientFactory(messages =>
        ModelResponse(messages, ["follow related relationship", "follow related relationship", root], false)).CreateClient();

    private static ChatResponse ModelResponse(IReadOnlyList<ChatMessage> messages, string[] followups, bool complete, bool localCompletes = true)
    {
        var reducing = messages.Any(message => message.Role == ChatRole.System && message.Text!.StartsWith("DRIFT evidence synthesis.", StringComparison.Ordinal));
        var local = messages.Any(message => message.Role == ChatRole.System && message.Text!.StartsWith("DRIFT local refinement.", StringComparison.Ordinal));
        using var input = JsonDocument.Parse(messages.First(message => message.Role == ChatRole.User && message.Text?.StartsWith("{", StringComparison.Ordinal) == true).Text!);
        var evidence = input.RootElement.GetProperty("Evidence").EnumerateArray().ToArray();
        var value = new DriftSearchReasoning
        {
            Answer = string.Join("; ", evidence.Select(item => item.GetProperty("Content").GetString())),
            CitationIds = evidence.Select(item => item.GetProperty("Id").GetString()!).ToArray(),
            FollowUpQuestions = reducing || (local && localCompletes) || complete ? [] : followups,
            IsComplete = reducing || (local && localCompletes) || complete,
            Confidence = 0.8
        };
        return new ChatResponse(new ChatMessage(ChatRole.Assistant, JsonSerializer.Serialize(value)));
    }

    private static async Task<DriftSearchEvidence[]> RetrieveAsync(IGraphStore store, string root,
        DriftSearchRequest request, CancellationToken token)
    {
        var identifiers = new HashSet<string>(StringComparer.Ordinal);
        if (request.Phase == DriftSearchPhase.Primer)
        { identifiers.Add(root); }
        else
        {
            await foreach (var edge in store.GetOutgoingRelationshipsAsync(root, token))
            { identifiers.Add(edge.TargetId); }
        }
        return (await ReadNodesAsync(store, token)).Where(node => identifiers.Contains(node.Id))
            .Take(request.MaxResults)
            .Select(node => new DriftSearchEvidence(node.Id, node.Label, node.Properties["content"]!.ToString()!,
                new Uri(new Uri("https://example.test/knowledge/"), node.Id))).ToArray();
    }

    private static async Task<List<GraphNode>> ReadNodesAsync(IGraphStore store, CancellationToken token = default)
    {
        var result = new List<GraphNode>();
        await foreach (var node in store.GetNodesAsync(cancellationToken: token))
        { result.Add(node); }
        return result;
    }

    private static DriftSearchConfig Configuration() => new()
    {
        Depth = 2,
        DriftFollowupCount = 3,
        Concurrency = 2,
        MaximumModelCalls = 4,
        DataMaxTokens = 2_000,
        PrimerMaxTokens = 1_000,
        LocalSearchMaxDataTokens = 1_000,
        LocalSearchTopKEntities = 2,
        LocalSearchTopKRelationships = 0,
        ReduceMaxCompletionTokens = 256
    };

    private static async Task<DriftSearchResult> CompleteAsync(DriftSearchEngine engine, string question,
        CancellationToken token = default)
    {
        DriftSearchResult? result = null;
        await foreach (var update in engine.SearchAsync(question, token))
        { result = update.Result ?? result; }
        return result ?? throw new InvalidOperationException("DRIFT did not return its terminal result.");
    }
}
