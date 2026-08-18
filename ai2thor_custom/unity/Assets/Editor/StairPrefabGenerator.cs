using System.Collections.Generic;
using System.Linq;
using Thor.Procedural;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.SceneManagement;

/// <summary>
/// Builds the schema-2 stair prefab required by ProcTHOR multi-floor houses.
/// Values match integrations/ai2thor/stair-asset-contract.json in hadr-nav/procthor.
/// </summary>
public static class StairPrefabGenerator {
    public const string AssetId = "Staircase_Straight_3m_1m_4_5m";
    const string PrefabDir = "Assets/Physics/VerticalConnectors";
    const string PrefabPath = PrefabDir + "/" + AssetId + ".prefab";
    const string ProceduralScenePath = "Assets/Scenes/Procedural/Procedural.unity";

    const float Rise = 3.0f;
    const float FlightWidth = 1.0f;
    const float FlightRun = 4.5f;
    const float ReservedWidth = 1.2f;
    const float ReservedLength = 6.5f;
    const float LandingDepth = 1.0f;
    const float SlabThickness = 0.2f;

    static float LandingCenterZ => (ReservedLength - LandingDepth) / 2.0f;

    public static void Generate() {
        EnsurePrefabDirectory();
        var root = BuildPrefabHierarchy();
        var prefab = SavePrefab(root);
        Object.DestroyImmediate(root);
        RegisterInProceduralScene(prefab);
        AssetDatabase.SaveAssets();
        Debug.Log("StairPrefabGenerator: registered " + AssetId);
    }

    static void EnsurePrefabDirectory() {
        if (!AssetDatabase.IsValidFolder("Assets/Physics")) {
            AssetDatabase.CreateFolder("Assets", "Physics");
        }
        if (!AssetDatabase.IsValidFolder(PrefabDir)) {
            AssetDatabase.CreateFolder("Assets/Physics", "VerticalConnectors");
        }
    }

    static GameObject BuildPrefabHierarchy() {
        var root = new GameObject(AssetId);
        var marker = root.AddComponent<VerticalConnectorAsset>();
        marker.connectorType = "stairs";
        marker.referenceRise = Rise;
        marker.referenceRun = FlightRun;
        marker.referenceWidth = FlightWidth;
        marker.reservedEnvelopeWidth = ReservedWidth;
        marker.reservedEnvelopeLength = ReservedLength;

        var walkableSurface = new GameObject("walkableSurface");
        walkableSurface.transform.SetParent(root.transform, false);
        var rampMesh = LoadOrCreateRampMeshAsset();
        var rampCollider = walkableSurface.AddComponent<MeshCollider>();
        rampCollider.sharedMesh = rampMesh;
        rampCollider.convex = false;
        marker.walkableSurface = walkableSurface;

        var lowerLanding = CreateLanding(
            "lowerLandingSurface",
            walkableSurface.transform,
            new Vector3(0.0f, 0.0f, -LandingCenterZ),
            SlabThickness
        );
        marker.lowerLandingSurface = lowerLanding;
        marker.lowerLandingAnchor = lowerLanding.transform;

        var upperLanding = CreateLanding(
            "upperLandingSurface",
            walkableSurface.transform,
            new Vector3(0.0f, Rise, LandingCenterZ),
            SlabThickness
        );
        marker.upperLandingSurface = upperLanding;
        marker.upperLandingAnchor = upperLanding.transform;

        SetNavMeshArea(root, NavMesh.GetAreaFromName("Not Walkable"));
        SetNavMeshArea(walkableSurface, NavMesh.GetAreaFromName("Walkable"));
        return root;
    }

    static GameObject CreateLanding(
        string name,
        Transform parent,
        Vector3 localPosition,
        float slabThickness
    ) {
        var landing = new GameObject(name);
        landing.transform.SetParent(parent, false);
        landing.transform.localPosition = localPosition;
        landing.transform.localRotation = Quaternion.identity;
        landing.transform.localScale = Vector3.one;
        var box = landing.AddComponent<BoxCollider>();
        box.size = new Vector3(ReservedWidth, slabThickness, LandingDepth);
        box.center = new Vector3(0.0f, -slabThickness / 2.0f, 0.0f);
        return landing;
    }

    static Mesh CreateRampMesh() {
        var halfWidth = FlightWidth / 2.0f;
        var halfRun = FlightRun / 2.0f;
        var mesh = new Mesh { name = "StairRampCollider" };
        mesh.vertices = new[] {
            new Vector3(-halfWidth, 0.0f, -halfRun),
            new Vector3(halfWidth, 0.0f, -halfRun),
            new Vector3(-halfWidth, Rise, halfRun),
            new Vector3(halfWidth, Rise, halfRun),
            new Vector3(-halfWidth, 0.0f, -halfRun),
            new Vector3(halfWidth, 0.0f, -halfRun),
            new Vector3(-halfWidth, 0.0f, halfRun),
            new Vector3(halfWidth, 0.0f, halfRun),
        };
        mesh.triangles = new[] {
            0, 2, 1, 1, 2, 3,
            4, 5, 6, 5, 7, 6,
            0, 1, 4, 1, 5, 4,
            2, 3, 6, 3, 7, 6,
            0, 4, 2, 2, 4, 6,
            1, 3, 5, 3, 7, 5,
        };
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        return mesh;
    }

    static Mesh LoadOrCreateRampMeshAsset() {
        const string meshPath = PrefabDir + "/StairRampCollider.asset";
        var existing = AssetDatabase.LoadAssetAtPath<Mesh>(meshPath);
        if (existing != null) {
            Object.DestroyImmediate(existing, true);
        }
        var mesh = CreateRampMesh();
        AssetDatabase.CreateAsset(mesh, meshPath);
        return mesh;
    }

    static void SetNavMeshArea(GameObject go, int area) {
        var modifier = go.GetComponent<NavMeshModifier>();
        if (modifier == null) {
            modifier = go.AddComponent<NavMeshModifier>();
        }
        modifier.overrideArea = true;
        modifier.area = area;
    }

    static GameObject SavePrefab(GameObject root) {
        var existing = AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath);
        if (existing != null) {
            PrefabUtility.SaveAsPrefabAsset(root, PrefabPath);
            return AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath);
        }
        return PrefabUtility.SaveAsPrefabAsset(root, PrefabPath);
    }

    static void RegisterInProceduralScene(GameObject prefab) {
        var scene = EditorSceneManager.OpenScene(ProceduralScenePath, OpenSceneMode.Single);
        var database = Object.FindObjectOfType<ProceduralAssetDatabase>();
        if (database == null) {
            throw new System.InvalidOperationException(
                "ProceduralAssetDatabase not found in " + ProceduralScenePath
            );
        }
        if (database.prefabs == null) {
            database.prefabs = new List<GameObject>();
        }
        if (!database.prefabs.Any(p => p != null && p.name == AssetId)) {
            database.prefabs.Add(prefab);
            EditorUtility.SetDirty(database);
        }
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
    }
}
