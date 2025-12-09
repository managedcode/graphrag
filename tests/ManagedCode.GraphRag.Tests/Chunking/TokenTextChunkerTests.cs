using GraphRag.Chunking;
using GraphRag.Config;
using GraphRag.Constants;
using GraphRag.Tokenization;

namespace ManagedCode.GraphRag.Tests.Chunking;

public sealed class TokenTextChunkerTests
{
    private readonly TokenTextChunker _chunker = new();
    private readonly ChunkingConfig _defaultConfig = new()
    {
        Size = 40,
        Overlap = 10,
        EncodingModel = TokenizerDefaults.DefaultEncoding
    };

    [Fact]
    public void Chunk_RespectsTokenBudget()
    {
        var tokenizer = TokenizerRegistry.GetTokenizer(TokenizerDefaults.DefaultEncoding);
        const string baseSentence = "Alice met Bob at the conference and shared insights.";
        var text = string.Join(' ', Enumerable.Repeat(baseSentence, 16));
        var slices = new[] { new ChunkSlice("doc-1", text) };

        var config = new ChunkingConfig
        {
            Size = 40,
            Overlap = 10,
            EncodingModel = TokenizerDefaults.DefaultEncoding
        };

        var totalTokens = tokenizer.EncodeToIds(text).Count;
        var chunks = _chunker.Chunk(slices, config);

        Assert.NotEmpty(chunks);
        Assert.All(chunks, chunk =>
        {
            Assert.Contains("doc-1", chunk.DocumentIds);
            Assert.True(chunk.TokenCount <= config.Size);
            Assert.False(string.IsNullOrWhiteSpace(chunk.Text));
        });

        if (totalTokens > config.Size)
        {
            Assert.True(chunks.Count > 1, "Expected multiple chunks when total tokens exceed configured size.");
        }
    }

    [Fact]
    public void Chunk_CombinesDocumentIdentifiersAcrossSlices()
    {
        var slices = new[]
        {
            new ChunkSlice("doc-1", string.Join(' ', Enumerable.Repeat("First slice carries shared content.", 4))),
            new ChunkSlice("doc-2", string.Join(' ', Enumerable.Repeat("Second slice enriches the narrative.", 4)))
        };

        var config = new ChunkingConfig
        {
            Size = 50,
            Overlap = 10,
            EncodingModel = TokenizerDefaults.DefaultModel
        };

        var chunks = _chunker.Chunk(slices, config);

        Assert.NotEmpty(chunks);
        Assert.Contains(chunks, chunk => chunk.DocumentIds.Contains("doc-1"));
        Assert.Contains(chunks, chunk => chunk.DocumentIds.Contains("doc-2"));
    }

    [Fact]
    public void Chunk_OverlapProducesSharedTokensBetweenAdjacentChunks()
    {
        var tokenizer = TokenizerRegistry.GetTokenizer(TokenizerDefaults.DefaultEncoding);
        const string text = "The quick brown fox jumps over the lazy dog and continues running through the forest until it reaches the river where it stops to drink some water.";
        var slices = new[] { new ChunkSlice("doc-1", text) };

        var config = new ChunkingConfig
        {
            Size = 20,
            Overlap = 5,
            EncodingModel = TokenizerDefaults.DefaultEncoding
        };

        var chunks = _chunker.Chunk(slices, config);

        Assert.True(chunks.Count >= 2, "Need at least 2 chunks to verify overlap");

        for (var i = 0; i < chunks.Count - 1; i++)
        {
            var currentChunkTokens = tokenizer.EncodeToIds(chunks[i].Text);
            var nextChunkTokens = tokenizer.EncodeToIds(chunks[i + 1].Text);

            var lastTokensOfCurrent = currentChunkTokens.TakeLast(config.Overlap).ToArray();
            var firstTokensOfNext = nextChunkTokens.Take(config.Overlap).ToArray();

            Assert.Equal(lastTokensOfCurrent, firstTokensOfNext);
        }
    }

    [Fact]
    public void Chunk_EmptySlicesReturnsEmptyResult()
    {
        var slices = Array.Empty<ChunkSlice>();

        var chunks = _chunker.Chunk(slices, _defaultConfig);

        Assert.Empty(chunks);
    }

    [Fact]
    public void Chunk_SlicesWithEmptyTextReturnsEmptyResult()
    {
        var slices = new[] { new ChunkSlice("doc-1", string.Empty) };

        var chunks = _chunker.Chunk(slices, _defaultConfig);

        Assert.Empty(chunks);
    }

    [Fact]
    public void Chunk_NullSlicesThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => _chunker.Chunk(null!, _defaultConfig));
    }

    [Fact]
    public void Chunk_NullConfigThrowsArgumentNullException()
    {
        var slices = new[] { new ChunkSlice("doc-1", "Some text") };

        Assert.Throws<ArgumentNullException>(() => _chunker.Chunk(slices, null!));
    }

    [Fact]
    public void Chunk_ZeroOverlapProducesNonOverlappingChunks()
    {
        var tokenizer = TokenizerRegistry.GetTokenizer(TokenizerDefaults.DefaultEncoding);
        const string text = "The quick brown fox jumps over the lazy dog and continues running through the forest until it reaches the river.";
        var slices = new[] { new ChunkSlice("doc-1", text) };

        var config = new ChunkingConfig
        {
            Size = 15,
            Overlap = 0,
            EncodingModel = TokenizerDefaults.DefaultEncoding
        };

        var chunks = _chunker.Chunk(slices, config);
        Assert.True(chunks.Count >= 2, "Need at least 2 chunks to verify zero overlap");

        var allChunkTokens = chunks
            .SelectMany(c => tokenizer.EncodeToIds(c.Text))
            .ToList();

        var originalTokens = tokenizer.EncodeToIds(text);

        Assert.Equal(originalTokens.Count, allChunkTokens.Count);
    }

    [Fact]
    public void Chunk_InputSmallerThanChunkSizeReturnsSingleChunk()
    {
        const string shortText = "Hello world";
        var slices = new[] { new ChunkSlice("doc-1", shortText) };

        var config = new ChunkingConfig
        {
            Size = 100,
            Overlap = 10,
            EncodingModel = TokenizerDefaults.DefaultEncoding
        };

        var chunks = _chunker.Chunk(slices, config);

        Assert.Single(chunks);
        Assert.Equal(shortText, chunks[0].Text);
    }

    [Fact]
    public void Chunk_ExactBoundaryProducesExpectedChunkCount()
    {
        var tokenizer = TokenizerRegistry.GetTokenizer(TokenizerDefaults.DefaultEncoding);

        const int chunkSize = 10;
        const int overlap = 2;
        const int step = chunkSize - overlap;

        var targetTokenCount = step * 3 + overlap;
        var words = Enumerable.Range(0, targetTokenCount * 2).Select(i => "word").ToArray();
        var text = string.Join(" ", words);

        var actualTokens = tokenizer.EncodeToIds(text);
        var slices = new[] { new ChunkSlice("doc-1", text) };

        var config = new ChunkingConfig
        {
            Size = chunkSize,
            Overlap = overlap,
            EncodingModel = TokenizerDefaults.DefaultEncoding
        };

        var chunks = _chunker.Chunk(slices, config);

        Assert.True(chunks.Count >= 2, "Should produce multiple chunks");
        Assert.All(chunks.SkipLast(1), chunk => Assert.Equal(chunkSize, chunk.TokenCount));
    }
}
