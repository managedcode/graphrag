using System.Buffers;
using System.Runtime.InteropServices;
using GraphRag.Config;
using GraphRag.Tokenization;

namespace GraphRag.Chunking;

public sealed class TokenTextChunker : ITextChunker
{
    public IReadOnlyList<TextChunk> Chunk(IReadOnlyList<ChunkSlice> slices, ChunkingConfig config)
    {
        ArgumentNullException.ThrowIfNull(slices);
        ArgumentNullException.ThrowIfNull(config);

        if (slices.Count == 0)
        {
            return [];
        }

        var tokenizer = TokenizerRegistry.GetTokenizer(config.EncodingModel);
        var flattened = new List<(int SliceIndex, int Token)>();

        for (var index = 0; index < slices.Count; index++)
        {
            var slice = slices[index];
            var encoded = tokenizer.EncodeToIds(slice.Text.AsSpan());
            for (var i = 0; i < encoded.Count; i++)
            {
                var token = encoded[i];
                flattened.Add((index, token));
            }
        }

        if (flattened.Count == 0)
        {
            return [];
        }

        var chunkSize = Math.Max(1, config.Size);
        var overlap = Math.Clamp(config.Overlap, 0, chunkSize - 1);

        var step = chunkSize - overlap;
        var estimatedChunks = (flattened.Count + step - 1) / step;
        var results = new List<TextChunk>(estimatedChunks);

        var documentIds = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        var start = 0;
        while (start < flattened.Count)
        {
            var end = Math.Min(flattened.Count, start + chunkSize);
            var chunkTokens = CollectionsMarshal.AsSpan(flattened).Slice(start, end - start);
            var tokenValues = ArrayPool<int>.Shared.Rent(chunkTokens.Length);
            documentIds.Clear();

            var lastSliceIndex = -1;
            for (var i = 0; i < chunkTokens.Length; i++)
            {
                var sliceIndex = chunkTokens[i].SliceIndex;
                tokenValues[i] = chunkTokens[i].Token;

                if (sliceIndex != lastSliceIndex)
                {
                    documentIds.Add(slices[sliceIndex].DocumentId);
                    lastSliceIndex = sliceIndex;
                }
            }

            var decoded = tokenizer.Decode(new ArraySegment<int>(tokenValues, 0, chunkTokens.Length));
            results.Add(new TextChunk(documentIds.ToList(), decoded, chunkTokens.Length));

            ArrayPool<int>.Shared.Return(tokenValues);

            if (end >= flattened.Count)
            {
                break;
            }

            start = Math.Max(start + chunkSize - overlap, start + 1);
        }

        return results;
    }
}
