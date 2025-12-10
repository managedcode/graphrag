using System.Buffers;
using System.Text;
using GraphRag.Config;
using GraphRag.Tokenization;
using Microsoft.ML.Tokenizers;

namespace GraphRag.Chunking;

public sealed class MarkdownTextChunker : ITextChunker
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
        var options = new MarkdownChunkerOptions
        {
            MaxTokensPerChunk = Math.Max(MinChunkSize, config.Size),
            Overlap = Math.Max(0, config.Overlap)
        };

        var results = new List<TextChunk>();

        foreach (var slice in slices)
        {
            var fragments = Split(slice.Text, options, tokenizer);
            foreach (var fragment in fragments)
            {
                var tokenCount = tokenizer.CountTokens(fragment.AsSpan());
                if (tokenCount == 0)
                {
                    continue;
                }

                results.Add(new TextChunk([slice.DocumentId], fragment, tokenCount));
            }
        }

        return results;
    }

    private List<string> Split(string text, MarkdownChunkerOptions options, Tokenizer tokenizer)
    {
        text = NormalizeNewlines(text);
        var firstChunkDone = false;
        var primarySize = options.MaxTokensPerChunk;
        var secondarySize = Math.Max(MinChunkSize, options.MaxTokensPerChunk - options.Overlap);

        var rawChunkRanges = RecursiveSplitRanges(
            text, 0..text.Length,
            primarySize, secondarySize,
            SeparatorType.ExplicitSeparator, tokenizer, ref firstChunkDone);

        List<string> rawChunks;

        if (options.Overlap > 0 && rawChunkRanges.Count > 1)
        {
            rawChunks = new List<string>(rawChunkRanges.Count);

            var firstChunkText = text[rawChunkRanges[0]];
            rawChunks.Add(firstChunkText);
            var previousTokens = tokenizer.EncodeToIds(firstChunkText.AsSpan());

            for (var i = 1; i < rawChunkRanges.Count; i++)
            {
                var currentChunkText = text[rawChunkRanges[i]];
                var skipCount = Math.Max(0, previousTokens.Count - options.Overlap);
                var overlapText = tokenizer.Decode(previousTokens.Skip(skipCount));

                rawChunks.Add(string.Concat(overlapText, currentChunkText));
                previousTokens = tokenizer.EncodeToIds(currentChunkText.AsSpan());
            }
        }
        else
        {
            // No overlap - simple range to string conversion
            rawChunks = new List<string>(rawChunkRanges.Count);
            foreach (var range in rawChunkRanges)
            {
                rawChunks.Add(text[range]);
            }
        }

        return MergeImageChunks(rawChunks);
    }

    private List<string> RecursiveSplit(
        string text,
        int maxChunk1Size,
        int maxChunkNSize,
        SeparatorType separatorType,
        Tokenizer tokenizer,
        ref bool firstChunkDone)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return [];
        }

        var maxChunkSize = firstChunkDone ? maxChunkNSize : maxChunk1Size;
        if (tokenizer.CountTokens(text) <= maxChunkSize)
        {
            return [text];
        }

        var fragments = separatorType switch
        {
            SeparatorType.ExplicitSeparator => SplitToFragments(text, ExplicitSeparators),
            SeparatorType.PotentialSeparator => SplitToFragments(text, PotentialSeparators),
            SeparatorType.WeakSeparator1 => SplitToFragments(text, WeakSeparators1),
            SeparatorType.WeakSeparator2 => SplitToFragments(text, WeakSeparators2),
            SeparatorType.WeakSeparator3 => SplitToFragments(text, WeakSeparators3),
            SeparatorType.NotASeparator => SplitToFragments(text, null),
            _ => throw new ArgumentOutOfRangeException(nameof(separatorType), separatorType, null)
        };

        return GenerateChunks(text, fragments, maxChunk1Size, maxChunkNSize, separatorType, tokenizer, ref firstChunkDone);
    }

    private List<string> GenerateChunks(
        string text,
        List<FragmentRange> fragments,
        int maxChunk1Size,
        int maxChunkNSize,
        SeparatorType separatorType,
        Tokenizer tokenizer,
        ref bool firstChunkDone)
    {
        if (fragments.Count == 0)
        {
            return [];
        }

        var chunks = new List<string>();
        var builder = new ChunkBuilder();
        var textSpan = text.AsSpan();

        foreach (var fragment in fragments)
        {
            builder.NextSentence.Append(textSpan[fragment.Range]);

            if (!fragment.IsSeparator)
            {
                continue;
            }

            var nextSentence = builder.NextSentence.ToString();
            var nextSentenceSize = tokenizer.CountTokens(nextSentence);
            var maxChunkSize = firstChunkDone ? maxChunkNSize : maxChunk1Size;
            var chunkEmpty = builder.FullContent.Length == 0;
            var sentenceTooLong = nextSentenceSize > maxChunkSize;

            if (chunkEmpty && !sentenceTooLong)
            {
                builder.FullContent.Append(nextSentence);
                builder.NextSentence.Clear();
                continue;
            }

            if (chunkEmpty)
            {
                var moreChunks = RecursiveSplit(nextSentence, maxChunk1Size, maxChunkNSize, NextSeparatorType(separatorType), tokenizer, ref firstChunkDone);
                chunks.AddRange(moreChunks.Take(moreChunks.Count - 1));
                builder.NextSentence.Clear().Append(moreChunks.Last());
                continue;
            }

            var chunkPlusSentence = builder.FullContent.ToString() + builder.NextSentence;
            if (!sentenceTooLong && tokenizer.CountTokens(chunkPlusSentence) <= maxChunkSize)
            {
                builder.FullContent.Append(builder.NextSentence);
                builder.NextSentence.Clear();
                continue;
            }

            AddChunk(chunks, builder.FullContent, ref firstChunkDone);

            if (sentenceTooLong)
            {
                var moreChunks = RecursiveSplit(nextSentence, maxChunk1Size, maxChunkNSize, NextSeparatorType(separatorType), tokenizer, ref firstChunkDone);
                chunks.AddRange(moreChunks.Take(moreChunks.Count - 1));
                builder.NextSentence.Clear().Append(moreChunks.Last());
            }
            else
            {
                builder.FullContent.Clear().Append(builder.NextSentence);
                builder.NextSentence.Clear();
            }
        }

        var fullSentenceLeft = builder.FullContent.ToString();
        var nextSentenceLeft = builder.NextSentence.ToString();
        var remainingMax = firstChunkDone ? maxChunkNSize : maxChunk1Size;

        if (tokenizer.CountTokens(fullSentenceLeft + nextSentenceLeft) <= remainingMax)
        {
            if (fullSentenceLeft.Length > 0 || nextSentenceLeft.Length > 0)
            {
                AddChunk(chunks, fullSentenceLeft + nextSentenceLeft, ref firstChunkDone);
            }

            return chunks;
        }

        if (fullSentenceLeft.Length > 0)
        {
            AddChunk(chunks, fullSentenceLeft, ref firstChunkDone);
        }

        if (nextSentenceLeft.Length == 0)
        {
            return chunks;
        }

        if (tokenizer.CountTokens(nextSentenceLeft) <= remainingMax)
        {
            AddChunk(chunks, nextSentenceLeft, ref firstChunkDone);
        }
        else
        {
            var moreChunks = RecursiveSplit(nextSentenceLeft, maxChunk1Size, maxChunkNSize, NextSeparatorType(separatorType), tokenizer, ref firstChunkDone);
            chunks.AddRange(moreChunks);
        }

        return chunks;
    }

    internal static List<FragmentRange> SplitToFragments(string text, SeparatorTrie? separators)
    {
        if (separators is null)
        {
            // Character-level fallback
            var charFragments = new List<FragmentRange>(text.Length);
            for (var i = 0; i < text.Length; i++)
            {
                charFragments.Add(new FragmentRange(i..(i + 1), true));
            }

            return charFragments;
        }

        if (text.Length == 0 || separators.Length == 0)
        {
            return [];
        }

        var span = text.AsSpan();
        var fragments = new List<FragmentRange>();
        var contentStart = 0;
        var index = 0;

        while (index < span.Length)
        {
            // Use SearchValues for vectorized skip to next potential separator
            var remaining = span[index..];
            var nextPotential = remaining.IndexOfAny(separators.FirstChars);

            if (nextPotential < 0)
            {
                // No more potential separators - rest is content
                break;
            }

            index += nextPotential;

            // Try to match a separator at this position
            var matchLength = separators.MatchLongestOptimized(span, index);

            if (matchLength > 0)
            {
                // Emit content fragment if any
                if (index > contentStart)
                {
                    fragments.Add(new FragmentRange(contentStart..index, false));
                }

                // Emit separator fragment
                fragments.Add(new FragmentRange(index..(index + matchLength), true));
                index += matchLength;
                contentStart = index;
            }
            else
            {
                // Not a real separator, continue
                index++;
            }
        }

        // Emit remaining content
        if (contentStart < text.Length)
        {
            fragments.Add(new FragmentRange(contentStart..text.Length, false));
        }

        return fragments;
    }

    internal static List<string> MergeImageChunks(List<string> chunks)
    {
        if (chunks.Count <= 1)
        {
            return chunks;
        }

        var merged = new List<string>();

        foreach (var chunk in chunks)
        {
            var trimmed = chunk.TrimStart();
            if (trimmed.StartsWith("![", StringComparison.Ordinal) && merged.Count > 0)
            {
                merged[^1] = string.Concat(merged[^1].TrimEnd(), "\n\n", chunk.TrimStart());
            }
            else
            {
                merged.Add(chunk);
            }
        }

        return merged;
    }

    private static void AddChunk(List<string> chunks, StringBuilder builder, ref bool firstChunkDone)
    {
        var chunk = builder.ToString();
        if (!string.IsNullOrWhiteSpace(chunk))
        {
            chunks.Add(chunk);
            firstChunkDone = true;
        }

        builder.Clear();
    }

    private static void AddChunk(List<string> chunks, string chunk, ref bool firstChunkDone)
    {
        if (!string.IsNullOrWhiteSpace(chunk))
        {
            chunks.Add(chunk);
            firstChunkDone = true;
        }
    }

    internal static string NormalizeNewlines(string input) => input.ReplaceLineEndings("\n");

    private List<Range> RecursiveSplitRanges(
        string originalText,
        Range workingRange,
        int maxChunk1Size,
        int maxChunkNSize,
        SeparatorType separatorType,
        Tokenizer tokenizer,
        ref bool firstChunkDone)
    {
        var offset = workingRange.Start.Value;
        var length = workingRange.End.Value - offset;

        if (length == 0 || originalText.AsSpan()[workingRange].IsWhiteSpace())
        {
            return [];
        }

        var maxChunkSize = firstChunkDone ? maxChunkNSize : maxChunk1Size;

        // CountTokens with Span - ZERO allocation!
        if (tokenizer.CountTokens(originalText.AsSpan()[workingRange]) <= maxChunkSize)
        {
            return [workingRange];
        }

        // Get fragments for the working range - this still needs a substring for SplitToFragments
        // but the fragments returned use ranges relative to this substring
        var workingText = originalText[workingRange];
        var fragments = separatorType switch
        {
            SeparatorType.ExplicitSeparator => SplitToFragments(workingText, ExplicitSeparators),
            SeparatorType.PotentialSeparator => SplitToFragments(workingText, PotentialSeparators),
            SeparatorType.WeakSeparator1 => SplitToFragments(workingText, WeakSeparators1),
            SeparatorType.WeakSeparator2 => SplitToFragments(workingText, WeakSeparators2),
            SeparatorType.WeakSeparator3 => SplitToFragments(workingText, WeakSeparators3),
            SeparatorType.NotASeparator => SplitToFragments(workingText, null),
            _ => throw new ArgumentOutOfRangeException(nameof(separatorType), separatorType, null)
        };

        return GenerateChunksRanges(originalText, offset, workingText, fragments, maxChunk1Size, maxChunkNSize, separatorType, tokenizer, ref firstChunkDone);
    }


    private List<Range> GenerateChunksRanges(
        string originalText,
        int textOffset,
        string workingText,
        List<FragmentRange> fragments,
        int maxChunk1Size,
        int maxChunkNSize,
        SeparatorType separatorType,
        Tokenizer tokenizer,
        ref bool firstChunkDone)
    {
        if (fragments.Count == 0)
        {
            return [];
        }

        var chunks = new List<Range>();
        var workingSpan = workingText.AsSpan();

        // Track positions as indices (in original text coordinates)
        var chunkStart = textOffset;
        var chunkEnd = textOffset;
        var sentenceStart = textOffset;

        foreach (var fragment in fragments)
        {
            var fragLocalEnd = fragment.Range.End.Value;
            var fragGlobalEnd = textOffset + fragLocalEnd;

            if (!fragment.IsSeparator)
            {
                continue;
            }

            // We have accumulated a sentence from sentenceStart to fragGlobalEnd
            var sentenceLocalStart = sentenceStart - textOffset;
            var sentenceLocalEnd = fragLocalEnd;

            var sentenceSpan = workingSpan[sentenceLocalStart..sentenceLocalEnd];
            var sentenceTokens = tokenizer.CountTokens(sentenceSpan);

            var maxChunkSize = firstChunkDone ? maxChunkNSize : maxChunk1Size;
            var chunkEmpty = chunkEnd <= chunkStart;
            var sentenceTooLong = sentenceTokens > maxChunkSize;

            if (chunkEmpty && !sentenceTooLong)
            {
                // First sentence in chunk, it fits
                chunkEnd = fragGlobalEnd;
                sentenceStart = fragGlobalEnd;
                continue;
            }

            if (chunkEmpty && sentenceTooLong)
            {
                // Sentence alone is too long - recursively split it
                var sentenceRange = sentenceStart..fragGlobalEnd;
                var moreRanges = RecursiveSplitRanges(
                    originalText, sentenceRange,
                    maxChunk1Size, maxChunkNSize,
                    NextSeparatorType(separatorType), tokenizer, ref firstChunkDone);

                if (moreRanges.Count > 0)
                {
                    // Add all but last as finalized chunks
                    for (var i = 0; i < moreRanges.Count - 1; i++)
                    {
                        chunks.Add(moreRanges[i]);
                    }

                    // Keep last range as the new chunk start
                    var lastRange = moreRanges[^1];
                    chunkStart = lastRange.Start.Value;
                    chunkEnd = lastRange.End.Value;
                }

                sentenceStart = fragGlobalEnd;
                continue;
            }

            // Check if chunk + sentence fits together
            var chunkLocalStart = chunkStart - textOffset;
            var combinedSpan = workingSpan[chunkLocalStart..sentenceLocalEnd];
            if (!sentenceTooLong && tokenizer.CountTokens(combinedSpan) <= maxChunkSize)
            {
                // Combined fits - extend chunk
                chunkEnd = fragGlobalEnd;
                sentenceStart = fragGlobalEnd;
                continue;
            }

            // Combined doesn't fit - finalize current chunk
            if (chunkEnd > chunkStart)
            {
                chunks.Add(chunkStart..chunkEnd);
                firstChunkDone = true;
            }

            if (sentenceTooLong)
            {
                // Recursively split the sentence
                var sentenceRange = sentenceStart..fragGlobalEnd;
                var moreRanges = RecursiveSplitRanges(
                    originalText, sentenceRange,
                    maxChunk1Size, maxChunkNSize,
                    NextSeparatorType(separatorType), tokenizer, ref firstChunkDone);

                if (moreRanges.Count > 0)
                {
                    for (var i = 0; i < moreRanges.Count - 1; i++)
                    {
                        chunks.Add(moreRanges[i]);
                    }

                    var lastRange = moreRanges[^1];
                    chunkStart = lastRange.Start.Value;
                    chunkEnd = lastRange.End.Value;
                }
                else
                {
                    chunkStart = fragGlobalEnd;
                    chunkEnd = fragGlobalEnd;
                }
            }
            else
            {
                // Start new chunk with this sentence
                chunkStart = sentenceStart;
                chunkEnd = fragGlobalEnd;
            }

            sentenceStart = fragGlobalEnd;
        }

        // Handle remaining content
        var lastFragEnd = textOffset + fragments[^1].Range.End.Value;

        if (chunkEnd > chunkStart || sentenceStart < lastFragEnd)
        {
            // Combine any remaining chunk content with leftover sentence
            var remainingStart = Math.Min(chunkStart, sentenceStart);
            var remainingEnd = Math.Max(chunkEnd, lastFragEnd);

            if (remainingEnd > remainingStart)
            {
                var remainingLocalStart = remainingStart - textOffset;
                var remainingLocalEnd = remainingEnd - textOffset;
                var remainingSpan = workingSpan[remainingLocalStart..remainingLocalEnd];
                var remainingMax = firstChunkDone ? maxChunkNSize : maxChunk1Size;

                if (tokenizer.CountTokens(remainingSpan) <= remainingMax)
                {
                    if (!remainingSpan.IsWhiteSpace())
                    {
                        chunks.Add(remainingStart..remainingEnd);
                        firstChunkDone = true;
                    }
                }
                else
                {
                    // Need to split remaining content
                    if (chunkEnd > chunkStart && !workingSpan[(chunkStart - textOffset)..(chunkEnd - textOffset)].IsWhiteSpace())
                    {
                        chunks.Add(chunkStart..chunkEnd);
                        firstChunkDone = true;
                    }

                    if (sentenceStart < lastFragEnd)
                    {
                        var leftoverSpan = workingSpan[(sentenceStart - textOffset)..(lastFragEnd - textOffset)];
                        if (!leftoverSpan.IsWhiteSpace())
                        {
                            if (tokenizer.CountTokens(leftoverSpan) <= remainingMax)
                            {
                                chunks.Add(sentenceStart..lastFragEnd);
                                firstChunkDone = true;
                            }
                            else
                            {
                                var moreRanges = RecursiveSplitRanges(
                                    originalText, sentenceStart..lastFragEnd,
                                    maxChunk1Size, maxChunkNSize,
                                    NextSeparatorType(separatorType), tokenizer, ref firstChunkDone);
                                chunks.AddRange(moreRanges);
                            }
                        }
                    }
                }
            }
        }

        return chunks;
    }

    private static SeparatorType NextSeparatorType(SeparatorType separatorType) => separatorType switch
    {
        SeparatorType.ExplicitSeparator => SeparatorType.PotentialSeparator,
        SeparatorType.PotentialSeparator => SeparatorType.WeakSeparator1,
        SeparatorType.WeakSeparator1 => SeparatorType.WeakSeparator2,
        SeparatorType.WeakSeparator2 => SeparatorType.WeakSeparator3,
        SeparatorType.WeakSeparator3 => SeparatorType.NotASeparator,
        _ => SeparatorType.NotASeparator
    };

    private const int MinChunkSize = 5;

    internal static readonly SeparatorTrie ExplicitSeparators = new([
        ".\n\n",
        "!\n\n",
        "!!\n\n",
        "!!!\n\n",
        "?\n\n",
        "??\n\n",
        "???\n\n",
        "\n\n",
        "\n#",
        "\n##",
        "\n###",
        "\n####",
        "\n#####",
        "\n---"
    ]);

    internal static readonly SeparatorTrie PotentialSeparators = new([
        "\n> ",
        "\n>- ",
        "\n>* ",
        "\n1. ",
        "\n2. ",
        "\n3. ",
        "\n4. ",
        "\n5. ",
        "\n6. ",
        "\n7. ",
        "\n8. ",
        "\n9. ",
        "\n10. ",
        "\n```"
    ]);

    internal static readonly SeparatorTrie WeakSeparators1 = new([
        "![",
        "[",
        "| ",
        " |\n",
        "-|\n",
        "\n: "
    ]);

    internal static readonly SeparatorTrie WeakSeparators2 = new([
        ". ", ".\t", ".\n",
        "? ", "?\t", "?\n",
        "! ", "!\t", "!\n",
        "⁉ ", "⁉\t", "⁉\n",
        "⁈ ", "⁈\t", "⁈\n",
        "⁇ ", "⁇\t", "⁇\n",
        "… ", "…\t", "…\n",
        "!!!!", "????", "!!!", "???", "?!?", "!?!", "!?", "?!", "!!", "??", "....", "...", "..",
        ".", "?", "!", "⁉", "⁈", "⁇", "…"
    ]);

    internal static readonly SeparatorTrie WeakSeparators3 = new([
        "; ", ";\t", ";\n", ";",
        "} ", "}\t", "}\n", "}",
        ") ", ")\t", ")\n",
        "] ", "]\t", "]\n",
        ")", "]",
        ": ", ":",
        ", ", ",",
        "\n"
    ]);

    private enum SeparatorType
    {
        ExplicitSeparator,
        PotentialSeparator,
        WeakSeparator1,
        WeakSeparator2,
        WeakSeparator3,
        NotASeparator
    }

    internal readonly record struct FragmentRange(Range Range, bool IsSeparator);

    private sealed class ChunkBuilder
    {
        public StringBuilder FullContent { get; } = new();
        public StringBuilder NextSentence { get; } = new();
    }

    private sealed class MarkdownChunkerOptions
    {
        public int MaxTokensPerChunk { get; init; }
        public int Overlap { get; init; }
    }

    internal sealed class SeparatorTrie
    {
        private readonly Dictionary<char, List<string>> _lookup = new();
        private readonly SearchValues<char> _firstChars;

        public int Length { get; }

        public SearchValues<char> FirstChars => _firstChars;

        public SeparatorTrie(IEnumerable<string> separators)
        {
            var list = separators.Where(static s => !string.IsNullOrEmpty(s)).ToList();
            Length = list.Count;

            foreach (var separator in list)
            {
                var key = separator[0];
                if (!_lookup.TryGetValue(key, out var bucket))
                {
                    bucket = [];
                    _lookup[key] = bucket;
                }

                bucket.Add(separator);
            }

            foreach (var bucket in _lookup.Values)
            {
                bucket.Sort((a, b) => b.Length.CompareTo(a.Length));
            }

            // Create SearchValues from first chars for vectorized lookup
            _firstChars = SearchValues.Create([.. _lookup.Keys]);
        }

        public string? MatchLongest(string text, int index)
        {
            if (index >= text.Length)
            {
                return null;
            }

            if (!_lookup.TryGetValue(text[index], out var candidates))
            {
                return null;
            }

            foreach (var candidate in candidates)
            {
                if (index + candidate.Length > text.Length)
                {
                    continue;
                }

                if (text.AsSpan(index, candidate.Length).SequenceEqual(candidate))
                {
                    return candidate;
                }
            }

            return null;
        }

        /// <summary>
        /// Returns the length of the longest matching separator at the given index, or 0 if no match.
        /// </summary>
        public int MatchLongestOptimized(ReadOnlySpan<char> text, int index)
        {
            if (index >= text.Length)
            {
                return 0;
            }

            if (!_lookup.TryGetValue(text[index], out var candidates))
            {
                return 0;
            }

            foreach (var candidate in candidates)
            {
                if (index + candidate.Length > text.Length)
                {
                    continue;
                }

                if (text.Slice(index, candidate.Length).SequenceEqual(candidate))
                {
                    return candidate.Length;
                }
            }

            return 0;
        }
    }
}
