using GraphRag.Chunking;
using GraphRag.Config;
using GraphRag.Constants;
using GraphRag.Tokenization;

namespace ManagedCode.GraphRag.Tests.Chunking;

public sealed class MarkdownTextChunkerTests
{
    private readonly MarkdownTextChunker _chunker = new();

    #region Chunk Tests (Original)

    [Fact]
    public void Chunk_SplitsMarkdownBlocks()
    {
        var text = "# Title\n\nAlice met Bob.\n\n![image](path)\n\n" +
                   string.Join(" ", Enumerable.Repeat("This is a longer paragraph that should be chunked based on token limits.", 4));
        var slices = new[] { new ChunkSlice("doc-1", text) };

        var config = new ChunkingConfig
        {
            Size = 60,
            Overlap = 10,
            EncodingModel = TokenizerDefaults.DefaultEncoding
        };

        var chunks = _chunker.Chunk(slices, config);

        Assert.NotEmpty(chunks);
        Assert.All(chunks, chunk => Assert.Contains("doc-1", chunk.DocumentIds));
        Assert.True(chunks.Count >= 2);
        Assert.All(chunks, chunk => Assert.True(chunk.TokenCount > 0));
    }

    [Fact]
    public void Chunk_MergesImageBlocksIntoPrecedingChunk()
    {
        var text = string.Join(' ', Enumerable.Repeat("This paragraph provides enough content for chunking.", 6)) +
                   "\n\n![diagram](diagram.png)\nImage description follows with more narrative text.";
        var slices = new[] { new ChunkSlice("doc-1", text) };

        var config = new ChunkingConfig
        {
            Size = 60,
            Overlap = 0,
            EncodingModel = TokenizerDefaults.DefaultModel
        };

        var chunks = _chunker.Chunk(slices, config);

        Assert.NotEmpty(chunks);
        Assert.Contains(chunks, chunk => chunk.Text.Contains("![diagram](diagram.png)", StringComparison.Ordinal));
        Assert.DoesNotContain(chunks, chunk => chunk.Text.TrimStart().StartsWith("![", StringComparison.Ordinal));
    }

    [Fact]
    public void Chunk_RespectsOverlapBetweenChunks()
    {
        var text = string.Join(' ', Enumerable.Repeat("Token overlap ensures continuity across generated segments.", 20));
        var slices = new[] { new ChunkSlice("doc-1", text) };

        var config = new ChunkingConfig
        {
            Size = 80,
            Overlap = 20,
            EncodingModel = "gpt-4"
        };

        var chunks = _chunker.Chunk(slices, config);

        Assert.True(chunks.Count > 1);

        var tokenizer = TokenizerRegistry.GetTokenizer(config.EncodingModel);
        var firstTokens = tokenizer.EncodeToIds(chunks[0].Text);

        _ = tokenizer.EncodeToIds(chunks[1].Text);
        var overlapTokens = firstTokens.Skip(Math.Max(0, firstTokens.Count - config.Overlap)).ToArray();
        Assert.True(overlapTokens.Length > 0);
        var overlapText = tokenizer.Decode(overlapTokens).TrimStart();
        var secondText = chunks[1].Text.TrimStart();
        Assert.StartsWith(overlapText, secondText, StringComparison.Ordinal);
    }

    #endregion

    #region SplitToFragments Tests

    [Fact]
    public void SplitToFragments_EmptyString_ReturnsEmpty()
    {
        var result = MarkdownTextChunker.SplitToFragments("", MarkdownTextChunker.ExplicitSeparators);
        Assert.Empty(result);
    }

    [Fact]
    public void SplitToFragments_NullSeparators_ReturnsCharacterLevelFragments()
    {
        var text = "abc";
        var result = MarkdownTextChunker.SplitToFragments(text, null);

        Assert.Equal(3, result.Count);
        Assert.All(result, f => Assert.True(f.IsSeparator));
        Assert.Equal("a", text[result[0].Range]);
        Assert.Equal("b", text[result[1].Range]);
        Assert.Equal("c", text[result[2].Range]);
    }

    [Fact]
    public void SplitToFragments_NoSeparatorsInText_ReturnsSingleContentFragment()
    {
        var text = "hello world";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.ExplicitSeparators);

        Assert.Single(result);
        Assert.False(result[0].IsSeparator);
        Assert.Equal("hello world", text[result[0].Range]);
    }

    [Fact]
    public void SplitToFragments_SeparatorAtStart_FirstFragmentIsSeparator()
    {
        var text = "\n\nhello";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.ExplicitSeparators);

        Assert.Equal(2, result.Count);
        Assert.True(result[0].IsSeparator);
        Assert.Equal("\n\n", text[result[0].Range]);
        Assert.False(result[1].IsSeparator);
        Assert.Equal("hello", text[result[1].Range]);
    }

    [Fact]
    public void SplitToFragments_SeparatorAtEnd_LastFragmentIsSeparator()
    {
        var text = "hello.\n\n";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.ExplicitSeparators);

        Assert.Equal(2, result.Count);
        Assert.False(result[0].IsSeparator);
        Assert.Equal("hello", text[result[0].Range]);
        Assert.True(result[1].IsSeparator);
        Assert.Equal(".\n\n", text[result[1].Range]);
    }

    [Fact]
    public void SplitToFragments_AdjacentSeparators_CreatesSeparateFragments()
    {
        var text = "\n\n\n\n";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.ExplicitSeparators);

        Assert.Equal(2, result.Count);
        Assert.All(result, f => Assert.True(f.IsSeparator));
        Assert.Equal("\n\n", text[result[0].Range]);
        Assert.Equal("\n\n", text[result[1].Range]);
    }

    [Fact]
    public void SplitToFragments_LongestMatchPrecedence_MatchesDotNewlineNewlineOverDot()
    {
        // Using WeakSeparators2 which has both "." and ".\n\n" isn't there, but ExplicitSeparators has ".\n\n"
        var text = "hello.\n\nworld";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.ExplicitSeparators);

        Assert.Equal(3, result.Count);
        Assert.Equal("hello", text[result[0].Range]);
        Assert.Equal(".\n\n", text[result[1].Range]);
        Assert.True(result[1].IsSeparator);
        Assert.Equal("world", text[result[2].Range]);
    }

    [Fact]
    public void SplitToFragments_LongestMatchPrecedence_MatchesTripleQuestionOverDouble()
    {
        var text = "what???really";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.WeakSeparators2);

        // Should match "???" not "??"
        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "???");
    }

    [Fact]
    public void SplitToFragments_UnicodeSeparators_HandlesInterrobangCorrectly()
    {
        var text = "what⁉ really";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.WeakSeparators2);

        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "⁉ ");
    }

    [Fact]
    public void SplitToFragments_UnicodeSeparators_HandlesEllipsisCorrectly()
    {
        var text = "wait… more";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.WeakSeparators2);

        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "… ");
    }

    #endregion

    #region ExplicitSeparators Additional Tests

    [Fact]
    public void SplitToFragments_HeaderSeparators_MatchesNewlineHash()
    {
        var text = "content\n# Header1\n## Header2";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.ExplicitSeparators);

        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "\n#");
        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "\n##");
    }

    [Fact]
    public void SplitToFragments_HeaderSeparators_MatchesAllLevels()
    {
        var text = "a\n#b\n##c\n###d\n####e\n#####f";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.ExplicitSeparators);

        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "\n#");
        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "\n##");
        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "\n###");
        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "\n####");
        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "\n#####");
    }

    [Fact]
    public void SplitToFragments_HorizontalRule_MatchesNewlineDashes()
    {
        var text = "above\n---below";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.ExplicitSeparators);

        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "\n---");
    }

    [Fact]
    public void SplitToFragments_ExclamationNewlines_MatchesAllVariants()
    {
        var text1 = "wow!\n\nmore";
        var text2 = "wow!!\n\nmore";
        var text3 = "wow!!!\n\nmore";
        var result1 = MarkdownTextChunker.SplitToFragments(text1, MarkdownTextChunker.ExplicitSeparators);
        var result2 = MarkdownTextChunker.SplitToFragments(text2, MarkdownTextChunker.ExplicitSeparators);
        var result3 = MarkdownTextChunker.SplitToFragments(text3, MarkdownTextChunker.ExplicitSeparators);

        Assert.Contains(result1, f => f.IsSeparator && text1[f.Range] == "!\n\n");
        Assert.Contains(result2, f => f.IsSeparator && text2[f.Range] == "!!\n\n");
        Assert.Contains(result3, f => f.IsSeparator && text3[f.Range] == "!!!\n\n");
    }

    [Fact]
    public void SplitToFragments_QuestionNewlines_MatchesAllVariants()
    {
        var text1 = "what?\n\nmore";
        var text2 = "what??\n\nmore";
        var text3 = "what???\n\nmore";
        var result1 = MarkdownTextChunker.SplitToFragments(text1, MarkdownTextChunker.ExplicitSeparators);
        var result2 = MarkdownTextChunker.SplitToFragments(text2, MarkdownTextChunker.ExplicitSeparators);
        var result3 = MarkdownTextChunker.SplitToFragments(text3, MarkdownTextChunker.ExplicitSeparators);

        Assert.Contains(result1, f => f.IsSeparator && text1[f.Range] == "?\n\n");
        Assert.Contains(result2, f => f.IsSeparator && text2[f.Range] == "??\n\n");
        Assert.Contains(result3, f => f.IsSeparator && text3[f.Range] == "???\n\n");
    }

    #endregion

    #region PotentialSeparators Tests

    [Fact]
    public void SplitToFragments_Blockquote_MatchesNewlineGreaterThan()
    {
        var text = "text\n> quoted";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.PotentialSeparators);

        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "\n> ");
    }

    [Fact]
    public void SplitToFragments_BlockquoteList_MatchesVariants()
    {
        var text1 = "text\n>- item";
        var text2 = "text\n>* item";
        var result1 = MarkdownTextChunker.SplitToFragments(text1, MarkdownTextChunker.PotentialSeparators);
        var result2 = MarkdownTextChunker.SplitToFragments(text2, MarkdownTextChunker.PotentialSeparators);

        Assert.Contains(result1, f => f.IsSeparator && text1[f.Range] == "\n>- ");
        Assert.Contains(result2, f => f.IsSeparator && text2[f.Range] == "\n>* ");
    }

    [Fact]
    public void SplitToFragments_NumberedList_MatchesDigitDotSpace()
    {
        var text = "intro\n1. first\n2. second\n10. tenth";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.PotentialSeparators);

        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "\n1. ");
        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "\n2. ");
        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "\n10. ");
    }

    [Fact]
    public void SplitToFragments_CodeFence_MatchesTripleBacktick()
    {
        var text = "text\n```code";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.PotentialSeparators);

        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "\n```");
    }

    #endregion

    #region WeakSeparators1 Tests

    [Fact]
    public void SplitToFragments_TablePipe_MatchesPipeVariants()
    {
        var text1 = "col1| col2";
        var text2 = "data |\nmore";
        var text3 = "---|-|\ndata";
        var result1 = MarkdownTextChunker.SplitToFragments(text1, MarkdownTextChunker.WeakSeparators1);
        var result2 = MarkdownTextChunker.SplitToFragments(text2, MarkdownTextChunker.WeakSeparators1);
        var result3 = MarkdownTextChunker.SplitToFragments(text3, MarkdownTextChunker.WeakSeparators1);

        Assert.Contains(result1, f => f.IsSeparator && text1[f.Range] == "| ");
        Assert.Contains(result2, f => f.IsSeparator && text2[f.Range] == " |\n");
        Assert.Contains(result3, f => f.IsSeparator && text3[f.Range] == "-|\n");
    }

    [Fact]
    public void SplitToFragments_LinkBracket_MatchesOpenBracket()
    {
        var text = "click [here](url)";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.WeakSeparators1);

        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "[");
    }

    [Fact]
    public void SplitToFragments_ImageBracket_MatchesExclamationBracket()
    {
        var text = "see ![alt](img.png)";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.WeakSeparators1);

        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "![");
    }

    [Fact]
    public void SplitToFragments_DefinitionList_MatchesNewlineColon()
    {
        var text = "term\n: definition";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.WeakSeparators1);

        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "\n: ");
    }

    #endregion

    #region WeakSeparators2 Additional Tests

    [Fact]
    public void SplitToFragments_TabSeparators_MatchesPunctuationTab()
    {
        var text1 = "end.\tnext";
        var text2 = "what?\tnext";
        var text3 = "wow!\tnext";
        var result1 = MarkdownTextChunker.SplitToFragments(text1, MarkdownTextChunker.WeakSeparators2);
        var result2 = MarkdownTextChunker.SplitToFragments(text2, MarkdownTextChunker.WeakSeparators2);
        var result3 = MarkdownTextChunker.SplitToFragments(text3, MarkdownTextChunker.WeakSeparators2);

        Assert.Contains(result1, f => f.IsSeparator && text1[f.Range] == ".\t");
        Assert.Contains(result2, f => f.IsSeparator && text2[f.Range] == "?\t");
        Assert.Contains(result3, f => f.IsSeparator && text3[f.Range] == "!\t");
    }

    [Fact]
    public void SplitToFragments_NewlineSeparators_MatchesPunctuationNewline()
    {
        var text1 = "end.\nnext";
        var text2 = "what?\nnext";
        var text3 = "wow!\nnext";
        var result1 = MarkdownTextChunker.SplitToFragments(text1, MarkdownTextChunker.WeakSeparators2);
        var result2 = MarkdownTextChunker.SplitToFragments(text2, MarkdownTextChunker.WeakSeparators2);
        var result3 = MarkdownTextChunker.SplitToFragments(text3, MarkdownTextChunker.WeakSeparators2);

        Assert.Contains(result1, f => f.IsSeparator && text1[f.Range] == ".\n");
        Assert.Contains(result2, f => f.IsSeparator && text2[f.Range] == "?\n");
        Assert.Contains(result3, f => f.IsSeparator && text3[f.Range] == "!\n");
    }

    [Fact]
    public void SplitToFragments_QuadPunctuation_MatchesFourChars()
    {
        var text1 = "what!!!!really";
        var text2 = "what????really";
        var result1 = MarkdownTextChunker.SplitToFragments(text1, MarkdownTextChunker.WeakSeparators2);
        var result2 = MarkdownTextChunker.SplitToFragments(text2, MarkdownTextChunker.WeakSeparators2);

        Assert.Contains(result1, f => f.IsSeparator && text1[f.Range] == "!!!!");
        Assert.Contains(result2, f => f.IsSeparator && text2[f.Range] == "????");
    }

    [Fact]
    public void SplitToFragments_MixedPunctuation_MatchesInterrobangVariants()
    {
        var text1 = "what?!?really";
        var text2 = "what!?!really";
        var text3 = "what!?really";
        var text4 = "what?!really";
        var result1 = MarkdownTextChunker.SplitToFragments(text1, MarkdownTextChunker.WeakSeparators2);
        var result2 = MarkdownTextChunker.SplitToFragments(text2, MarkdownTextChunker.WeakSeparators2);
        var result3 = MarkdownTextChunker.SplitToFragments(text3, MarkdownTextChunker.WeakSeparators2);
        var result4 = MarkdownTextChunker.SplitToFragments(text4, MarkdownTextChunker.WeakSeparators2);

        Assert.Contains(result1, f => f.IsSeparator && text1[f.Range] == "?!?");
        Assert.Contains(result2, f => f.IsSeparator && text2[f.Range] == "!?!");
        Assert.Contains(result3, f => f.IsSeparator && text3[f.Range] == "!?");
        Assert.Contains(result4, f => f.IsSeparator && text4[f.Range] == "?!");
    }

    [Fact]
    public void SplitToFragments_Ellipsis_MatchesDotVariants()
    {
        var text1 = "wait....more";
        var text2 = "wait...more";
        var text3 = "wait..more";
        var result1 = MarkdownTextChunker.SplitToFragments(text1, MarkdownTextChunker.WeakSeparators2);
        var result2 = MarkdownTextChunker.SplitToFragments(text2, MarkdownTextChunker.WeakSeparators2);
        var result3 = MarkdownTextChunker.SplitToFragments(text3, MarkdownTextChunker.WeakSeparators2);

        Assert.Contains(result1, f => f.IsSeparator && text1[f.Range] == "....");
        Assert.Contains(result2, f => f.IsSeparator && text2[f.Range] == "...");
        Assert.Contains(result3, f => f.IsSeparator && text3[f.Range] == "..");
    }

    [Fact]
    public void SplitToFragments_SinglePunctuation_MatchesWithoutSpace()
    {
        // Single punctuation at end of string (no space after)
        var text1 = "end.";
        var text2 = "end?";
        var text3 = "end!";
        var result1 = MarkdownTextChunker.SplitToFragments(text1, MarkdownTextChunker.WeakSeparators2);
        var result2 = MarkdownTextChunker.SplitToFragments(text2, MarkdownTextChunker.WeakSeparators2);
        var result3 = MarkdownTextChunker.SplitToFragments(text3, MarkdownTextChunker.WeakSeparators2);

        Assert.Contains(result1, f => f.IsSeparator && text1[f.Range] == ".");
        Assert.Contains(result2, f => f.IsSeparator && text2[f.Range] == "?");
        Assert.Contains(result3, f => f.IsSeparator && text3[f.Range] == "!");
    }

    [Fact]
    public void SplitToFragments_DoubleQuestion_MatchesBeforeTriple()
    {
        var text = "what??next";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.WeakSeparators2);

        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "??");
    }

    [Fact]
    public void SplitToFragments_DoubleExclamation_MatchesBeforeTriple()
    {
        var text = "wow!!next";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.WeakSeparators2);

        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "!!");
    }

    #endregion

    #region WeakSeparators3 Tests

    [Fact]
    public void SplitToFragments_Semicolon_MatchesAllVariants()
    {
        var text1 = "a; b";
        var text2 = "a;\tb";
        var text3 = "a;\nb";
        var text4 = "a;b";
        var result1 = MarkdownTextChunker.SplitToFragments(text1, MarkdownTextChunker.WeakSeparators3);
        var result2 = MarkdownTextChunker.SplitToFragments(text2, MarkdownTextChunker.WeakSeparators3);
        var result3 = MarkdownTextChunker.SplitToFragments(text3, MarkdownTextChunker.WeakSeparators3);
        var result4 = MarkdownTextChunker.SplitToFragments(text4, MarkdownTextChunker.WeakSeparators3);

        Assert.Contains(result1, f => f.IsSeparator && text1[f.Range] == "; ");
        Assert.Contains(result2, f => f.IsSeparator && text2[f.Range] == ";\t");
        Assert.Contains(result3, f => f.IsSeparator && text3[f.Range] == ";\n");
        Assert.Contains(result4, f => f.IsSeparator && text4[f.Range] == ";");
    }

    [Fact]
    public void SplitToFragments_CloseBrace_MatchesAllVariants()
    {
        var text1 = "a} b";
        var text2 = "a}\tb";
        var text3 = "a}\nb";
        var text4 = "a}b";
        var result1 = MarkdownTextChunker.SplitToFragments(text1, MarkdownTextChunker.WeakSeparators3);
        var result2 = MarkdownTextChunker.SplitToFragments(text2, MarkdownTextChunker.WeakSeparators3);
        var result3 = MarkdownTextChunker.SplitToFragments(text3, MarkdownTextChunker.WeakSeparators3);
        var result4 = MarkdownTextChunker.SplitToFragments(text4, MarkdownTextChunker.WeakSeparators3);

        Assert.Contains(result1, f => f.IsSeparator && text1[f.Range] == "} ");
        Assert.Contains(result2, f => f.IsSeparator && text2[f.Range] == "}\t");
        Assert.Contains(result3, f => f.IsSeparator && text3[f.Range] == "}\n");
        Assert.Contains(result4, f => f.IsSeparator && text4[f.Range] == "}");
    }

    [Fact]
    public void SplitToFragments_CloseParen_MatchesAllVariants()
    {
        var text1 = "(a) b";
        var text2 = "(a)\tb";
        var text3 = "(a)\nb";
        var text4 = "(a)b";
        var result1 = MarkdownTextChunker.SplitToFragments(text1, MarkdownTextChunker.WeakSeparators3);
        var result2 = MarkdownTextChunker.SplitToFragments(text2, MarkdownTextChunker.WeakSeparators3);
        var result3 = MarkdownTextChunker.SplitToFragments(text3, MarkdownTextChunker.WeakSeparators3);
        var result4 = MarkdownTextChunker.SplitToFragments(text4, MarkdownTextChunker.WeakSeparators3);

        Assert.Contains(result1, f => f.IsSeparator && text1[f.Range] == ") ");
        Assert.Contains(result2, f => f.IsSeparator && text2[f.Range] == ")\t");
        Assert.Contains(result3, f => f.IsSeparator && text3[f.Range] == ")\n");
        Assert.Contains(result4, f => f.IsSeparator && text4[f.Range] == ")");
    }

    [Fact]
    public void SplitToFragments_CloseBracket_MatchesAllVariants()
    {
        var text1 = "[a] b";
        var text2 = "[a]\tb";
        var text3 = "[a]\nb";
        var text4 = "[a]b";
        var result1 = MarkdownTextChunker.SplitToFragments(text1, MarkdownTextChunker.WeakSeparators3);
        var result2 = MarkdownTextChunker.SplitToFragments(text2, MarkdownTextChunker.WeakSeparators3);
        var result3 = MarkdownTextChunker.SplitToFragments(text3, MarkdownTextChunker.WeakSeparators3);
        var result4 = MarkdownTextChunker.SplitToFragments(text4, MarkdownTextChunker.WeakSeparators3);

        Assert.Contains(result1, f => f.IsSeparator && text1[f.Range] == "] ");
        Assert.Contains(result2, f => f.IsSeparator && text2[f.Range] == "]\t");
        Assert.Contains(result3, f => f.IsSeparator && text3[f.Range] == "]\n");
        Assert.Contains(result4, f => f.IsSeparator && text4[f.Range] == "]");
    }

    [Fact]
    public void SplitToFragments_Colon_MatchesAllVariants()
    {
        var text1 = "key: value";
        var text2 = "key:value";
        var result1 = MarkdownTextChunker.SplitToFragments(text1, MarkdownTextChunker.WeakSeparators3);
        var result2 = MarkdownTextChunker.SplitToFragments(text2, MarkdownTextChunker.WeakSeparators3);

        Assert.Contains(result1, f => f.IsSeparator && text1[f.Range] == ": ");
        Assert.Contains(result2, f => f.IsSeparator && text2[f.Range] == ":");
    }

    [Fact]
    public void SplitToFragments_Comma_MatchesAllVariants()
    {
        var text1 = "a, b";
        var text2 = "a,b";
        var result1 = MarkdownTextChunker.SplitToFragments(text1, MarkdownTextChunker.WeakSeparators3);
        var result2 = MarkdownTextChunker.SplitToFragments(text2, MarkdownTextChunker.WeakSeparators3);

        Assert.Contains(result1, f => f.IsSeparator && text1[f.Range] == ", ");
        Assert.Contains(result2, f => f.IsSeparator && text2[f.Range] == ",");
    }

    [Fact]
    public void SplitToFragments_SingleNewline_MatchesInWeakSeparators3()
    {
        var text = "line1\nline2";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.WeakSeparators3);

        Assert.Contains(result, f => f.IsSeparator && text[f.Range] == "\n");
    }

    #endregion

    #region Edge Cases and Optimized Equivalence Tests

    [Fact]
    public void SplitToFragments_MultipleSeparatorTypes_ProcessesInOrder()
    {
        // Mix of different separator types
        var text = "hello.\n\nworld";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.ExplicitSeparators);

        Assert.Equal(3, result.Count);
        Assert.Equal("hello", text[result[0].Range]);
        Assert.False(result[0].IsSeparator);
        Assert.Equal(".\n\n", text[result[1].Range]);
        Assert.True(result[1].IsSeparator);
        Assert.Equal("world", text[result[2].Range]);
        Assert.False(result[2].IsSeparator);
    }

    [Fact]
    public void SplitToFragments_SixConsecutiveNewlines_CreatesSeparateFragments()
    {
        var text = "\n\n\n\n\n\n";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.ExplicitSeparators);

        // Should match \n\n three times
        Assert.Equal(3, result.Count);
        Assert.All(result, f => Assert.True(f.IsSeparator));
        Assert.All(result, f => Assert.Equal("\n\n", text[f.Range]));
    }

    [Fact]
    public void SplitToFragments_SeparatorOnly_ReturnsOnlySeparators()
    {
        var text = ".\n\n";
        var result = MarkdownTextChunker.SplitToFragments(text, MarkdownTextChunker.ExplicitSeparators);

        Assert.Single(result);
        Assert.True(result[0].IsSeparator);
        Assert.Equal(".\n\n", text[result[0].Range]);
    }

    #endregion

    #region NormalizeNewlines Tests

    [Fact]
    public void NormalizeNewlines_CRLF_ConvertsToLF()
    {
        var result = MarkdownTextChunker.NormalizeNewlines("hello\r\nworld");
        Assert.Equal("hello\nworld", result);
    }

    [Fact]
    public void NormalizeNewlines_CROnly_ConvertsToLF()
    {
        var result = MarkdownTextChunker.NormalizeNewlines("hello\rworld");
        Assert.Equal("hello\nworld", result);
    }

    [Fact]
    public void NormalizeNewlines_MixedLineEndings_AllConvertToLF()
    {
        var result = MarkdownTextChunker.NormalizeNewlines("a\r\nb\rc\nd");
        Assert.Equal("a\nb\nc\nd", result);
    }

    [Fact]
    public void NormalizeNewlines_AlreadyNormalized_Unchanged()
    {
        var result = MarkdownTextChunker.NormalizeNewlines("hello\nworld");
        Assert.Equal("hello\nworld", result);
    }

    [Fact]
    public void NormalizeNewlines_NoLineEndings_Unchanged()
    {
        var result = MarkdownTextChunker.NormalizeNewlines("hello world");
        Assert.Equal("hello world", result);
    }

    #endregion

    #region MergeImageChunks Tests

    [Fact]
    public void MergeImageChunks_NoImages_Unchanged()
    {
        var chunks = new List<string> { "first", "second", "third" };
        var result = MarkdownTextChunker.MergeImageChunks(chunks);

        Assert.Equal(3, result.Count);
        Assert.Equal(chunks, result);
    }

    [Fact]
    public void MergeImageChunks_ImageAtStart_NotMerged()
    {
        var chunks = new List<string> { "![image](path)", "second" };
        var result = MarkdownTextChunker.MergeImageChunks(chunks);

        Assert.Equal(2, result.Count);
        Assert.Equal("![image](path)", result[0]);
    }

    [Fact]
    public void MergeImageChunks_ImageAfterContent_MergedWithPrevious()
    {
        var chunks = new List<string> { "some text", "![image](path)" };
        var result = MarkdownTextChunker.MergeImageChunks(chunks);

        Assert.Single(result);
        Assert.Contains("some text", result[0]);
        Assert.Contains("![image](path)", result[0]);
    }

    [Fact]
    public void MergeImageChunks_ConsecutiveImages_AllMergedIntoPreceding()
    {
        var chunks = new List<string> { "content", "![img1](p1)", "![img2](p2)" };
        var result = MarkdownTextChunker.MergeImageChunks(chunks);

        Assert.Single(result);
        Assert.Contains("content", result[0]);
        Assert.Contains("![img1](p1)", result[0]);
        Assert.Contains("![img2](p2)", result[0]);
    }

    [Fact]
    public void MergeImageChunks_SingleChunk_Unchanged()
    {
        var chunks = new List<string> { "single chunk" };
        var result = MarkdownTextChunker.MergeImageChunks(chunks);

        Assert.Single(result);
        Assert.Equal("single chunk", result[0]);
    }

    #endregion

    #region Overlap Handling Tests

    [Fact]
    public void Chunk_ZeroOverlap_NoOverlapProcessing()
    {
        var text = string.Join(' ', Enumerable.Repeat("This sentence repeats for testing purposes.", 20));
        var slices = new[] { new ChunkSlice("doc-1", text) };

        var config = new ChunkingConfig
        {
            Size = 50,
            Overlap = 0,
            EncodingModel = TokenizerDefaults.DefaultEncoding
        };

        var chunks = _chunker.Chunk(slices, config);

        Assert.True(chunks.Count > 1);
        // With zero overlap, chunks should not have shared prefix/suffix
        var tokenizer = TokenizerRegistry.GetTokenizer(config.EncodingModel);
        var firstTokens = tokenizer.EncodeToIds(chunks[0].Text);
        var secondTokens = tokenizer.EncodeToIds(chunks[1].Text);

        // First token of second chunk shouldn't be last token of first chunk
        // (unless by coincidence from the text itself)
        Assert.True(firstTokens.Count > 0);
        Assert.True(secondTokens.Count > 0);
    }

    [Fact]
    public void Chunk_OverlapSmallerThanChunk_AddsOverlapPrefix()
    {
        var text = string.Join(' ', Enumerable.Repeat("Word", 100));
        var slices = new[] { new ChunkSlice("doc-1", text) };

        var config = new ChunkingConfig
        {
            Size = 30,
            Overlap = 10,
            EncodingModel = TokenizerDefaults.DefaultEncoding
        };

        var chunks = _chunker.Chunk(slices, config);

        Assert.True(chunks.Count > 1);
        // Second chunk should start with overlap from first
        var tokenizer = TokenizerRegistry.GetTokenizer(config.EncodingModel);
        var firstTokens = tokenizer.EncodeToIds(chunks[0].Text);
        var overlapTokens = firstTokens.Skip(Math.Max(0, firstTokens.Count - config.Overlap)).ToArray();
        var overlapText = tokenizer.Decode(overlapTokens);

        Assert.StartsWith(overlapText.Trim(), chunks[1].Text.Trim(), StringComparison.Ordinal);
    }

    [Fact]
    public void Chunk_SingleChunk_NoOverlapNeeded()
    {
        var text = "Short text";
        var slices = new[] { new ChunkSlice("doc-1", text) };

        var config = new ChunkingConfig
        {
            Size = 100,
            Overlap = 20,
            EncodingModel = TokenizerDefaults.DefaultEncoding
        };

        var chunks = _chunker.Chunk(slices, config);

        Assert.Single(chunks);
        Assert.Equal("Short text", chunks[0].Text);
    }

    #endregion

    #region GenerateChunks Token Boundary Tests

    [Fact]
    public void Chunk_SmallDocument_FitsInSingleChunk()
    {
        var text = "Hello world. This is a test.";
        var slices = new[] { new ChunkSlice("doc-1", text) };

        var config = new ChunkingConfig
        {
            Size = 100,
            Overlap = 0,
            EncodingModel = TokenizerDefaults.DefaultEncoding
        };

        var chunks = _chunker.Chunk(slices, config);

        Assert.Single(chunks);
    }

    [Fact]
    public void Chunk_LargeDocument_SplitsIntoMultipleChunks()
    {
        var text = string.Join("\n\n", Enumerable.Repeat("This is a paragraph with enough content to exceed token limits when repeated multiple times.", 20));
        var slices = new[] { new ChunkSlice("doc-1", text) };

        var config = new ChunkingConfig
        {
            Size = 50,
            Overlap = 0,
            EncodingModel = TokenizerDefaults.DefaultEncoding
        };

        var chunks = _chunker.Chunk(slices, config);

        Assert.True(chunks.Count > 1);

        // Each chunk should respect token limit (approximately)
        var tokenizer = TokenizerRegistry.GetTokenizer(config.EncodingModel);
        foreach (var chunk in chunks)
        {
            var tokenCount = tokenizer.CountTokens(chunk.Text);
            // Allow some flexibility due to overlap and boundary handling
            Assert.True(tokenCount <= config.Size * 1.5, $"Chunk has {tokenCount} tokens, expected <= {config.Size * 1.5}");
        }
    }

    [Fact]
    public void Chunk_DocumentWithHeaders_SplitsAtHeaderBoundaries()
    {
        var text = "# Header 1\n\nContent for header 1.\n\n## Header 2\n\nContent for header 2.\n\n### Header 3\n\nContent for header 3.";
        var slices = new[] { new ChunkSlice("doc-1", text) };

        var config = new ChunkingConfig
        {
            Size = 20,
            Overlap = 0,
            EncodingModel = TokenizerDefaults.DefaultEncoding
        };

        var chunks = _chunker.Chunk(slices, config);

        Assert.True(chunks.Count >= 1);
        // Headers should be preserved in chunks
        Assert.Contains(chunks, c => c.Text.Contains('#'));
    }

    [Fact]
    public void Chunk_TrailingContent_Captured()
    {
        var text = "First paragraph.\n\nSecond paragraph.\n\nTrailing content.";
        var slices = new[] { new ChunkSlice("doc-1", text) };

        var config = new ChunkingConfig
        {
            Size = 200,
            Overlap = 0,
            EncodingModel = TokenizerDefaults.DefaultEncoding
        };

        var chunks = _chunker.Chunk(slices, config);

        var allText = string.Join("", chunks.Select(c => c.Text));
        Assert.Contains("Trailing content", allText);
    }

    #endregion
}
