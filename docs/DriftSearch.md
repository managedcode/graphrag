# DRIFT over existing graphs

The core query engine consumes bounded prepared evidence from a host-provided `DriftSearchRetriever`.
It does not parse, regenerate or persist a graph and does not require a new database.
A host can retrieve evidence from existing JSON-LD graphs, graph stores or a prepared community-report index.
The primer/local phases tell the host which retrieval context is requested.
No ontology, communities or model-extracted graph data are created by this query engine.

`DriftSearchEngine.SearchAsync` streams primer, local-refinement and terminal updates.
The supplied MEAI `IChatClient` remains owned and instrumented by the host.
Model IDs, provider credentials, tenant authorization, billing and any embeddings are host responsibilities;
the engine never creates an alternative provider client.

The engine performs an evidence-grounded primer, deduplicates focused follow-up questions,
retrieves/refines them in bounded parallel batches, then synthesizes the gathered evidence.
An already-complete primer needs no additional model round trip.
Every model citation must reference evidence actually supplied to that model call.
An evidence ID whose content changes during a search fails visibly; include source revision in IDs.

Work limits come from `DriftSearchConfig`: depth, total follow-up count, concurrency, maximum model calls,
per-phase retrieval result count and retained evidence-token budgets. One model call is reserved for synthesis.
The host must honor the request's result limit and bound evidence contents before returning them.
The engine exposes exhausted budgets rather than claiming the absence of facts.
`Complete` means that this bounded query completed with an evidence-backed answer; it is not
a claim that every node of an arbitrary corpus was traversed.
Cancellation, retrieval errors and invalid model output propagate without replay or direct-search fallback.

The current engine supports existing prepared context, including source/group summaries when supplied.
It is not an indexing pipeline and does not implement automatic Microsoft GraphRAG community extraction.
Disk-backed/full-graph query planning is outside the engine: bounded output alone does not solve
an unbounded JSON-LD cold load.

Acceptance:
1. Primer -> related stored evidence -> synthesis preserves exact citations across PostgreSQL, Neo4j and JanusGraph.
2. Duplicate questions cannot produce repeated retrieval loops; a configured budget is reported explicitly.
3. Invented citations fail and leave the stored graph unchanged.
4. Cancellation leaves the graph readable and a fresh search can complete.

These paths are exercised in `DriftSearchIntegrationTests` over the real provider containers,
with the fixture's deterministic MEAI model provider.

Evidence may include a revision. Keep Source stable across versions and provide Revision when sources can change; mixing revisions in one query fails visibly. Citation IDs remain unique for each source revision.

Streaming updates include retrieved evidence before the model call and cited intermediate findings before synthesis. Hosts can commit these typed values to their existing CQRS journal; the library does not own HTTP connections, command persistence or Stop authorization.
