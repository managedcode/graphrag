using GraphRag.Config;

namespace GraphRag.Query.Drift;

internal sealed record DriftSearchLimits(
    int Depth, int Followups, int Concurrency, int ModelCalls, int Results,
    int TotalTokens, int PrimerTokens, int LocalTokens, int OutputTokens)
{
    internal static DriftSearchLimits From(DriftSearchConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.Depth <= 0 || config.DriftFollowupCount < 0 || config.Concurrency <= 0 ||
            config.MaximumModelCalls < 2 || config.DataMaxTokens <= 0 || config.PrimerMaxTokens <= 0 ||
            config.LocalSearchMaxDataTokens <= 0 || config.LocalSearchTopKEntities <= 0 ||
            config.LocalSearchTopKRelationships < 0 || config.ReduceMaxCompletionTokens is <= 0 ||
            config.ReduceMaxTokens is <= 0 || config.LocalSearchMaxGeneratedTokens is <= 0)
        {
            throw new ArgumentException("DRIFT requires positive work bounds and at least two allowed model calls.", nameof(config));
        }

        return new(config.Depth, config.DriftFollowupCount, config.Concurrency, config.MaximumModelCalls,
            checked(config.LocalSearchTopKEntities + config.LocalSearchTopKRelationships),
            config.DataMaxTokens, Math.Min(config.DataMaxTokens, config.PrimerMaxTokens),
            Math.Min(config.DataMaxTokens, config.LocalSearchMaxDataTokens),
            config.ReduceMaxCompletionTokens ?? config.ReduceMaxTokens ?? config.PrimerMaxTokens);
    }
}
