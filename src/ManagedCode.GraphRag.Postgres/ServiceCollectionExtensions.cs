using System;
using System.Collections.Generic;
using System.Linq;
using GraphRag.Config;
using GraphRag.Graphs;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace GraphRag.Storage.Postgres;

public static class ServiceCollectionExtensions
{
    public static IServiceCollection AddPostgresGraphStore(this IServiceCollection services, string key, Action<PostgresGraphStoreOptions> configure, bool makeDefault = false)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentException.ThrowIfNullOrWhiteSpace(key);
        ArgumentNullException.ThrowIfNull(configure);

        var options = new PostgresGraphStoreOptions();
        configure(options);

        services.AddKeyedSingleton<PostgresGraphStoreOptions>(key, (_, _) => options);
        services.AddKeyedSingleton<PostgresGraphStore>(key, (sp, serviceKey) =>
        {
            var opts = sp.GetRequiredKeyedService<PostgresGraphStoreOptions>(serviceKey);
            var logger = sp.GetRequiredService<ILogger<PostgresGraphStore>>();
            return new PostgresGraphStore(opts, logger);
        });
        services.AddKeyedSingleton<IGraphStore>(key, (sp, serviceKey) => sp.GetRequiredKeyedService<PostgresGraphStore>(serviceKey));
        services.AddKeyedSingleton<PostgresExplainService>(key, (sp, serviceKey) =>
        {
            var store = sp.GetRequiredKeyedService<PostgresGraphStore>(serviceKey);
            var logger = sp.GetRequiredService<ILogger<PostgresExplainService>>();
            return new PostgresExplainService(store, logger);
        });

        if (makeDefault)
        {
            services.AddSingleton(sp => sp.GetRequiredKeyedService<PostgresGraphStore>(key));
            services.AddSingleton<IGraphStore>(sp => sp.GetRequiredKeyedService<PostgresGraphStore>(key));
            services.AddSingleton(sp => sp.GetRequiredKeyedService<PostgresExplainService>(key));
        }

        return services;
    }

    public static IServiceCollection AddPostgresGraphStores(this IServiceCollection services, GraphRagConfig graphRagConfig)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(graphRagConfig);

        var stores = graphRagConfig.GetPostgresGraphStores();
        if (stores.Count == 0)
        {
            return services;
        }

        foreach (var (key, storeConfig) in stores)
        {
            if (string.IsNullOrWhiteSpace(key) || storeConfig is null)
            {
                continue;
            }

            services.AddPostgresGraphStore(key, options =>
            {
                options.ConnectionString = storeConfig.ConnectionString;
                options.GraphName = storeConfig.GraphName;
                options.AutoCreateIndexes = storeConfig.AutoCreateIndexes;
                options.VertexPropertyIndexes = ClonePropertyIndexMap(storeConfig.VertexPropertyIndexes);
                options.EdgePropertyIndexes = ClonePropertyIndexMap(storeConfig.EdgePropertyIndexes);
            }, storeConfig.MakeDefault);
        }

        return services;
    }

    private static Dictionary<string, string[]> ClonePropertyIndexMap(Dictionary<string, string[]>? source)
    {
        if (source is null || source.Count == 0)
        {
            return new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase);
        }

        var clone = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase);
        foreach (var (label, properties) in source)
        {
            if (string.IsNullOrWhiteSpace(label) || properties is null)
            {
                continue;
            }

            var cleaned = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var property in properties)
            {
                if (string.IsNullOrWhiteSpace(property))
                {
                    continue;
                }

                cleaned.Add(property.Trim());
            }

            if (cleaned.Count > 0)
            {
                clone[label] = cleaned.ToArray();
            }
        }

        return clone;
    }
}

public sealed class PostgresGraphStoreOptions
{
    public string ConnectionString { get; set; } = string.Empty;

    public string GraphName { get; set; } = "graphrag";

    public bool AutoCreateIndexes { get; set; } = true;

    public Dictionary<string, string[]> VertexPropertyIndexes { get; set; } = new(StringComparer.OrdinalIgnoreCase);

    public Dictionary<string, string[]> EdgePropertyIndexes { get; set; } = new(StringComparer.OrdinalIgnoreCase);
}
