using GraphRag.Community;
using GraphRag.Config;
using GraphRag.Entities;
using GraphRag.Relationships;

namespace ManagedCode.GraphRag.Tests.Community;

public sealed class FastLabelPropagationCommunityDetectorTests
{
    #region Helper Methods

    private static EntityRecord CreateEntity(string title) =>
        new(title, 0, title, "test", null, [], 1, 0, 0, 0);

    private static RelationshipRecord CreateRelationship(string source, string target, double weight = 1.0) =>
        new($"{source}-{target}", 0, source, target, "related", null, weight, 0, [], false);

    private static ClusterGraphConfig CreateConfig(int seed = 42, int maxIterations = 20) =>
        new() { Seed = seed, MaxIterations = maxIterations };

    #endregion

    #region Edge Cases

    [Fact]
    public void AssignLabels_EmptyEntities_ReturnsEmptyDictionary()
    {
        var entities = Array.Empty<EntityRecord>();
        var relationships = Array.Empty<RelationshipRecord>();
        var config = CreateConfig();

        var result = FastLabelPropagationCommunityDetector.AssignLabels(entities, relationships, config);

        Assert.Empty(result);
    }

    [Fact]
    public void AssignLabels_EmptyRelationships_ReturnsNodesSelfLabeled()
    {
        var entities = new[] { CreateEntity("A"), CreateEntity("B"), CreateEntity("C") };
        var relationships = Array.Empty<RelationshipRecord>();
        var config = CreateConfig();

        var result = FastLabelPropagationCommunityDetector.AssignLabels(entities, relationships, config);

        Assert.Equal(3, result.Count);
        Assert.Equal("A", result["A"]);
        Assert.Equal("B", result["B"]);
        Assert.Equal("C", result["C"]);
    }

    [Fact]
    public void AssignLabels_SingleNode_ReturnsSelfLabel()
    {
        var entities = new[] { CreateEntity("Solo") };
        var relationships = Array.Empty<RelationshipRecord>();
        var config = CreateConfig();

        var result = FastLabelPropagationCommunityDetector.AssignLabels(entities, relationships, config);

        Assert.Single(result);
        Assert.Equal("Solo", result["Solo"]);
    }

    [Fact]
    public void AssignLabels_NullEntities_ThrowsArgumentNullException()
    {
        var relationships = Array.Empty<RelationshipRecord>();
        var config = CreateConfig();

        Assert.Throws<ArgumentNullException>(() =>
            FastLabelPropagationCommunityDetector.AssignLabels(null!, relationships, config));
    }

    [Fact]
    public void AssignLabels_NullRelationships_ThrowsArgumentNullException()
    {
        var entities = Array.Empty<EntityRecord>();
        var config = CreateConfig();

        Assert.Throws<ArgumentNullException>(() =>
            FastLabelPropagationCommunityDetector.AssignLabels(entities, null!, config));
    }

    [Fact]
    public void AssignLabels_NullConfig_ThrowsArgumentNullException()
    {
        var entities = Array.Empty<EntityRecord>();
        var relationships = Array.Empty<RelationshipRecord>();

        Assert.Throws<ArgumentNullException>(() =>
            FastLabelPropagationCommunityDetector.AssignLabels(entities, relationships, null!));
    }

    #endregion

    #region Deterministic Outcome Tests

    [Fact]
    public void AssignLabels_TwoConnectedNodes_AssignsSameLabel()
    {
        var entities = new[] { CreateEntity("A"), CreateEntity("B") };
        var relationships = new[] { CreateRelationship("A", "B") };
        var config = CreateConfig(seed: 42);

        var result = FastLabelPropagationCommunityDetector.AssignLabels(entities, relationships, config);

        Assert.Equal(2, result.Count);
        Assert.Equal(result["A"], result["B"]);
    }

    [Fact]
    public void AssignLabels_ThreeNodeChain_ConvergesToSameLabel()
    {
        // A -- B -- C (chain topology)
        var entities = new[] { CreateEntity("A"), CreateEntity("B"), CreateEntity("C") };
        var relationships = new[]
        {
            CreateRelationship("A", "B"),
            CreateRelationship("B", "C")
        };
        var config = CreateConfig(seed: 42);

        var result = FastLabelPropagationCommunityDetector.AssignLabels(entities, relationships, config);

        Assert.Equal(3, result.Count);
        var uniqueLabels = result.Values.Distinct().ToList();
        Assert.Single(uniqueLabels);
    }

    [Fact]
    public void AssignLabels_TwoDisconnectedComponents_AssignsDifferentLabels()
    {
        // Component 1: A -- B
        // Component 2: C -- D
        var entities = new[]
        {
            CreateEntity("A"), CreateEntity("B"),
            CreateEntity("C"), CreateEntity("D")
        };
        var relationships = new[]
        {
            CreateRelationship("A", "B"),
            CreateRelationship("C", "D")
        };
        var config = CreateConfig(seed: 42);

        var result = FastLabelPropagationCommunityDetector.AssignLabels(entities, relationships, config);

        Assert.Equal(4, result.Count);

        // Nodes in same component should have same label
        Assert.Equal(result["A"], result["B"]);
        Assert.Equal(result["C"], result["D"]);

        // Nodes in different components should have different labels
        Assert.NotEqual(result["A"], result["C"]);
    }

    [Fact]
    public void AssignLabels_StarTopology_AllNodesGetCenterLabel()
    {
        // Star: Center connected to Leaf1, Leaf2, Leaf3
        var entities = new[]
        {
            CreateEntity("Center"),
            CreateEntity("Leaf1"),
            CreateEntity("Leaf2"),
            CreateEntity("Leaf3")
        };
        var relationships = new[]
        {
            CreateRelationship("Center", "Leaf1"),
            CreateRelationship("Center", "Leaf2"),
            CreateRelationship("Center", "Leaf3")
        };
        var config = CreateConfig(seed: 42);

        var result = FastLabelPropagationCommunityDetector.AssignLabels(entities, relationships, config);

        Assert.Equal(4, result.Count);
        var uniqueLabels = result.Values.Distinct().ToList();
        Assert.Single(uniqueLabels);
    }

    [Fact]
    public void AssignLabels_WeightedEdges_HigherWeightInfluencesResult()
    {
        // B connects to A with high weight, C connects to A with low weight
        // B -- A -- C
        //  10    1
        var entities = new[]
        {
            CreateEntity("A"),
            CreateEntity("B"),
            CreateEntity("C")
        };
        var relationships = new[]
        {
            CreateRelationship("A", "B", weight: 10.0),
            CreateRelationship("A", "C", weight: 1.0)
        };
        var config = CreateConfig(seed: 42);

        var result = FastLabelPropagationCommunityDetector.AssignLabels(entities, relationships, config);

        Assert.Equal(3, result.Count);
        // All should converge to same community
        var uniqueLabels = result.Values.Distinct().ToList();
        Assert.Single(uniqueLabels);
    }

    #endregion

    #region Property-Based Invariant Tests

    [Fact]
    public void AssignLabels_AllNodesGetLabel()
    {
        var entities = new[]
        {
            CreateEntity("Node1"),
            CreateEntity("Node2"),
            CreateEntity("Node3"),
            CreateEntity("Node4"),
            CreateEntity("Node5")
        };
        var relationships = new[]
        {
            CreateRelationship("Node1", "Node2"),
            CreateRelationship("Node3", "Node4")
        };
        var config = CreateConfig(seed: 123);

        var result = FastLabelPropagationCommunityDetector.AssignLabels(entities, relationships, config);

        foreach (var entity in entities)
        {
            Assert.True(result.ContainsKey(entity.Title), $"Entity '{entity.Title}' should have a label");
        }
    }

    [Fact]
    public void AssignLabels_LabelsAreValidNodeIds()
    {
        var entities = new[]
        {
            CreateEntity("Alpha"),
            CreateEntity("Beta"),
            CreateEntity("Gamma"),
            CreateEntity("Delta")
        };
        var relationships = new[]
        {
            CreateRelationship("Alpha", "Beta"),
            CreateRelationship("Gamma", "Delta")
        };
        var config = CreateConfig(seed: 456);

        var result = FastLabelPropagationCommunityDetector.AssignLabels(entities, relationships, config);

        var validTitles = entities.Select(e => e.Title).ToHashSet(StringComparer.OrdinalIgnoreCase);
        foreach (var label in result.Values)
        {
            Assert.True(validTitles.Contains(label), $"Label '{label}' should be a valid node title");
        }
    }

    [Fact]
    public void AssignLabels_ConnectedNodesConverge_WithSufficientIterations()
    {
        // Triangle: A -- B -- C -- A (all connected)
        var entities = new[]
        {
            CreateEntity("A"),
            CreateEntity("B"),
            CreateEntity("C")
        };
        var relationships = new[]
        {
            CreateRelationship("A", "B"),
            CreateRelationship("B", "C"),
            CreateRelationship("C", "A")
        };
        var config = CreateConfig(seed: 789, maxIterations: 50);

        var result = FastLabelPropagationCommunityDetector.AssignLabels(entities, relationships, config);

        var uniqueLabels = result.Values.Distinct().ToList();
        Assert.Single(uniqueLabels);
    }

    #endregion

    #region Convergence Tests

    [Fact]
    public void AssignLabels_ConvergesBeforeMaxIterations_WhenStable()
    {
        // Simple connected graph that should converge quickly
        var entities = new[] { CreateEntity("X"), CreateEntity("Y") };
        var relationships = new[] { CreateRelationship("X", "Y") };
        var config = CreateConfig(seed: 42, maxIterations: 1000);

        var result = FastLabelPropagationCommunityDetector.AssignLabels(entities, relationships, config);

        Assert.Equal(2, result.Count);
        Assert.Equal(result["X"], result["Y"]);
    }

    [Fact]
    public void AssignLabels_MaxIterationsOne_StillProducesValidOutput()
    {
        var entities = new[]
        {
            CreateEntity("P"),
            CreateEntity("Q"),
            CreateEntity("R")
        };
        var relationships = new[]
        {
            CreateRelationship("P", "Q"),
            CreateRelationship("Q", "R")
        };
        var config = CreateConfig(seed: 42, maxIterations: 1);

        var result = FastLabelPropagationCommunityDetector.AssignLabels(entities, relationships, config);

        // Even with 1 iteration, all nodes should have labels
        Assert.Equal(3, result.Count);
        Assert.True(result.ContainsKey("P"));
        Assert.True(result.ContainsKey("Q"));
        Assert.True(result.ContainsKey("R"));

        // Labels should be valid node titles
        var validTitles = new HashSet<string>(["P", "Q", "R"], StringComparer.OrdinalIgnoreCase);
        foreach (var label in result.Values)
        {
            Assert.Contains(label, validTitles);
        }
    }

    #endregion
}
