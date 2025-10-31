# GraphRAG for .NET

GraphRAG for .NET is a ground-up port of Microsoft’s GraphRAG reference implementation to the modern .NET 9 stack. The port keeps parity with the original Python pipelines while embracing native .NET idioms—dependency injection, logging abstractions, async I/O, and strongly-typed configuration.

> ℹ️ The upstream Python code remains available under [`submodules/graphrag-python`](submodules/graphrag-python) for side-by-side reference. Treat it as read-only unless a task explicitly targets the submodule.

---

## Feature Highlights

- **End-to-end indexing workflows.** All standard GraphRAG stages—document loading, chunking, graph extraction, community building, and summarisation—ship as discrete workflows that can be registered with a single `AddGraphRag(...)` call.
- **Heuristic ingestion & maintenance.** Built-in overlapping chunk windows, semantic deduplication, orphan-node linking, relationship enhancement/validation, and token-budget trimming keep your graph clean without bespoke services.
- **Fast label propagation communities.** A configurable fast label propagation detector (with connected-component fallback) mirrors the behaviour of the GraphRag.Net demo directly inside the pipeline.
- **Pluggable graph stores.** Ready-made adapters for Azure Cosmos DB, Neo4j, and Apache AGE/PostgreSQL conform to `IGraphStore` so you can swap back-ends without touching workflows.
- **Prompt orchestration.** Prompt templates cascade through manual, auto-tuned, and default sources using [Microsoft.Extensions.AI](https://learn.microsoft.com/dotnet/ai/overview) keyed clients for chat and embedding models.
- **Deterministic integration tests.** Testcontainers spin up the real databases, while stub embeddings provide stable heuristics coverage so CI can validate the full pipeline.

---

## Repository Structure

```
graphrag/
├── GraphRag.slnx                          # Solution spanning runtime + test projects
├── Directory.Build.props / Directory.Packages.props
├── src/
│   ├── ManagedCode.GraphRag               # Core pipeline orchestration & abstractions
│   ├── ManagedCode.GraphRag.CosmosDb      # Azure Cosmos DB graph adapter
│   ├── ManagedCode.GraphRag.Neo4j         # Neo4j adapter & Bolt integration
│   └── ManagedCode.GraphRag.Postgres      # Apache AGE/PostgreSQL graph adapter
├── tests/
│   └── ManagedCode.GraphRag.Tests
│       ├── Integration/                   # Live container-backed scenarios
│       └── … unit-level suites
└── submodules/
    └── graphrag-python                    # Original Python implementation (read-only)
```

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| [.NET SDK 9.0](https://dotnet.microsoft.com/download/dotnet/9.0) | The solution targets `net9.0`. Use the in-repo [`dotnet-install.sh`](dotnet-install.sh) helper on CI. |
| Docker Desktop / compatible runtime | Required for Testcontainers-backed integration tests (Neo4j & Apache AGE/PostgreSQL). |
| (Optional) Azure Cosmos DB Emulator | Set `COSMOS_EMULATOR_CONNECTION_STRING` to enable Cosmos-specific tests. |

---

## Quick Start

1. **Clone & initialise submodules**
   ```bash
   git clone https://github.com/<your-org>/graphrag.git
   cd graphrag
   git submodule update --init --recursive
   ```

2. **Install .NET 9 if needed**
   ```bash
   ./dotnet-install.sh --version 9.0.100
   export PATH="$HOME/.dotnet:$PATH"
   ```

3. **Restore & build (always build before testing)**
   ```bash
   dotnet build GraphRag.slnx
   ```

4. **Run the full test suite**
   ```bash
   dotnet test GraphRag.slnx --logger "console;verbosity=minimal"
   ```
   This command restores packages, launches Neo4j and Apache AGE/PostgreSQL containers via Testcontainers, runs unit + integration tests, and tears everything down automatically.

5. **Target a specific scenario (optional)**
   ```bash
   dotnet test tests/ManagedCode.GraphRag.Tests/ManagedCode.GraphRag.Tests.csproj \
       --filter "FullyQualifiedName~HeuristicMaintenanceIntegrationTests" \
       --logger "console;verbosity=normal"
   ```

6. **Format before committing**
   ```bash
   dotnet format GraphRag.slnx
   ```

---

## Using GraphRAG in Your Application

Register GraphRAG services and provide keyed Microsoft.Extensions.AI clients for every model reference:

```csharp
using Azure;
using Azure.AI.OpenAI;
using GraphRag;
using GraphRag.Config;
using Microsoft.Extensions.AI;

var openAi = new OpenAIClient(new Uri(endpoint), new AzureKeyCredential(key));

builder.Services.AddKeyedSingleton<IChatClient>(
    "chat_model",
    _ => openAi.GetChatClient(chatDeployment));

builder.Services.AddKeyedSingleton<IEmbeddingGenerator<string, Embedding>>(
    "embedding_model",
    _ => openAi.GetEmbeddingClient(embeddingDeployment));

builder.Services.AddGraphRag();
```

---

## Pipeline Cache & Extensibility

Every workflow in a pipeline shares the same `IPipelineCache` instance via `PipelineRunContext`. The default DI registration wires up `MemoryPipelineCache`, letting workflows reuse expensive intermediate artefacts (LLM responses, chunk expansions, graph lookups) without recomputation. Swap in your own implementation by registering `IPipelineCache` before invoking `AddGraphRag()`—for example to persist cache entries or aggregate diagnostics.

- **Child scopes.** `MemoryPipelineCache.CreateChild("stage")` prefixes keys with the stage name so multi-step workflows remain isolated.
- **Debug payloads.** Entries can include optional debug data; clearing the cache removes both the value and associated trace metadata.
- **Custom lifetimes.** Register a scoped cache if you want to align the cache with a single HTTP request rather than the default singleton lifetime.

---

## Heuristic Ingestion & Maintenance

The .NET port incorporates the ingestion behaviours showcased in GraphRag.Net directly inside the indexing pipeline:

- **Overlapping chunk windows** produce coherent context spans that survive community trimming.
- **Semantic deduplication** drops duplicate text units by comparing embedding cosine similarity against a configurable threshold.
- **Token-budget trimming** automatically enforces global and per-community token ceilings during summarisation.
- **Orphan-node linking** reconnects isolated entities through high-confidence relationships before finalisation.
- **Relationship enhancement & validation** reconciles LLM output with existing edges to avoid duplicates while strengthening weights.

Configure the heuristics via `GraphRagConfig.Heuristics` (for example in `appsettings.json`):

```json
{
  "GraphRag": {
    "Models": [ "chat_model", "embedding_model" ],
    "EmbedText": {
      "ModelId": "embedding_model"
    },
    "Heuristics": {
      "MinimumChunkOverlap": 128,
      "EnableSemanticDeduplication": true,
      "SemanticDeduplicationThreshold": 0.92,
      "MaxTokensPerTextUnit": 1200,
      "MaxDocumentTokenBudget": 6000,
      "MaxTextUnitsPerRelationship": 6,
      "LinkOrphanEntities": true,
      "OrphanLinkMinimumOverlap": 0.25,
      "OrphanLinkWeight": 0.35,
      "EnhanceRelationships": true,
      "RelationshipConfidenceFloor": 0.35
    }
  }
}
```

See [`docs/indexing-and-query.md`](docs/indexing-and-query.md) for the full list of knobs and how they map to the original research flow.

---

## Community Detection & Graph Analytics

Community creation defaults to the fast label propagation algorithm. Tweak clustering directly through configuration:

```json
{
  "GraphRag": {
    "Models": [ "chat_model", "embedding_model" ],
    "ClusterGraph": {
      "Algorithm": "FastLabelPropagation",
      "MaxIterations": 40,
      "MaxClusterSize": 25,
      "UseLargestConnectedComponent": true,
      "Seed": 3735928559
    }
  }
}
```

If the graph is sparse, the pipeline falls back to connected components to ensure every node participates in a community. The heuristics integration tests (`Integration/HeuristicMaintenanceIntegrationTests.cs`) cover both the label propagation path and the connected-component fallback.

---

## Integration Testing Strategy

- **Real services only.** All graph operations run against containerised Neo4j and Apache AGE/PostgreSQL instances provisioned by Testcontainers.
- **Deterministic heuristics.** `StubEmbeddingGenerator` guarantees stable embeddings so semantic-dedup and token-budget assertions remain reliable.
- **Cross-store validation.** Shared integration fixtures verify that workflows succeed against each adapter (Cosmos tests activate when the emulator connection string is present).
- **Prompt precedence.** Tests validate that manual prompt overrides win over auto-tuned variants while still cascading correctly to the default templates.
- **Telemetry coverage.** Runtime tests assert pipeline callbacks and execution statistics so custom instrumentation keeps working.

To run just the container-backed suite:

```bash
dotnet test tests/ManagedCode.GraphRag.Tests/ManagedCode.GraphRag.Tests.csproj \
    --filter "Category=Integration" \
    --logger "console;verbosity=normal"
```

---

## Additional Documentation & Diagrams

- [`docs/indexing-and-query.md`](docs/indexing-and-query.md) explains how each workflow maps to the GraphRAG research diagrams (default data flow, query orchestrations, prompt tuning strategies) published at [microsoft.github.io/graphrag](https://microsoft.github.io/graphrag/).
- [`docs/dotnet-port-plan.md`](docs/dotnet-port-plan.md) outlines the migration strategy from Python to .NET and references the canonical architecture diagrams used during the port.
- The upstream documentation contains the latest diagrams for indexing, query, and data schema. Use those diagrams when presenting the system—it matches the pipeline implemented here.

---

## Local Cosmos Testing

1. Install and start the [Azure Cosmos DB Emulator](https://learn.microsoft.com/azure/cosmos-db/local-emulator).
2. Export the connection string:
   ```bash
   export COSMOS_EMULATOR_CONNECTION_STRING="AccountEndpoint=https://localhost:8081/;AccountKey=..."
   ```
3. Run `dotnet test`; Cosmos-specific scenarios will seed the emulator and validate storage behaviour.

---

## Development Tips

- **Solution layout.** Open `GraphRag.slnx` in your IDE for a full workspace view.
- **Formatting & analyzers.** Run `dotnet format GraphRag.slnx` before committing.
- **Coding conventions.** Nullable reference types and implicit usings are enabled; keep annotations accurate and suffix async methods with `Async`.
- **Extending graph adapters.** Implement `IGraphStore` and register your service through DI when adding new storage back-ends.

---

## Contributing

1. Fork the repository and create a feature branch from `main`.
2. Make your changes, ensuring `dotnet build GraphRag.slnx` succeeds before you run tests.
3. Execute `dotnet test GraphRag.slnx` (with Docker running) and `dotnet format GraphRag.slnx` before opening a pull request.
4. Include the test output in your PR description and link any related issues.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for detailed guidance.

---

## License & Credits

- Licensed under the [MIT License](LICENSE).
- GraphRAG is © Microsoft. This repository reimplements the pipelines for the .NET ecosystem while staying aligned with the official documentation and diagrams.

Have questions or feedback? Open an issue or start a discussion—we’re actively evolving the .NET port and welcome contributions! 🚀
