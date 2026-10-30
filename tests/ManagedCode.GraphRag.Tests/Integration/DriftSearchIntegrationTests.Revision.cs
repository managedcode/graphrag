using GraphRag.Query.Drift;

namespace ManagedCode.GraphRag.Tests.Integration;

public sealed partial class DriftSearchIntegrationTests
{
    [Theory]
    [InlineData("postgres")]
    [InlineData("neo4j")]
    [InlineData("janus")]
    public async Task Search_RejectsSourceRevisionChangesInsteadOfMixingEvidence(string provider)
    {
        var (store, root, detail) = await SeedAsync(provider);
        using var client = CreateModel(root);
        var source = new Uri(new Uri("https://example.test/knowledge/"), root);
        async Task<DriftSearchEvidence[]> Retrieve(DriftSearchRequest request, CancellationToken token)
        {
            var revision = request.Phase == DriftSearchPhase.Primer ? "1" : "2";
            if (revision == "2")
            {
                await store.UpsertNodeAsync(detail, GraphStoreTestProviders.GetLabel(provider),
                    new Dictionary<string, object?> { ["content"] = "updated detail evidence" }, token);
            }
            return (await RetrieveAsync(store, root, request, token))
                .Select(item => item with { Source = source, Revision = revision }).ToArray();
        }
        var engine = new DriftSearchEngine(client, Retrieve, Configuration());
        await Assert.ThrowsAsync<InvalidDataException>(() => CompleteAsync(engine, root));
        Assert.Equal("updated detail evidence",
            (await ReadNodesAsync(store)).Single(node => node.Id == detail).Properties["content"]);
    }
}
