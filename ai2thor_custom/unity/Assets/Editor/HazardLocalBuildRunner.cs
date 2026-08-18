using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using System;

/// <summary>
/// Invoked from the command line to produce a FloorPlan1 + Procedural local build
/// without -batchmode (works with Unity Personal entitlement licensing).
/// </summary>
public static class HazardLocalBuildRunner {
    private static readonly string[] AlwaysIncludedShaderNames = {
        "Legacy Shaders/Particles/Additive",
        "Legacy Shaders/Particles/Alpha Blended",
        "Sprites/Default",
    };

    public static void Run() {
        EnsureParticleShadersIncluded();
        StairPrefabGenerator.Generate();
        Environment.SetEnvironmentVariable("BUILD_SCENES", "FloorPlan1_physics,Procedural/Procedural");
        string buildDir = Environment.GetEnvironmentVariable("HAZARD_UNITY_BUILD_DIR");
        if (string.IsNullOrEmpty(buildDir)) {
            buildDir = "builds/thor-OSXIntel64-local/thor-OSXIntel64-local";
        }
        Environment.SetEnvironmentVariable("UNITY_BUILD_NAME", buildDir);
        Build.HazardLocalBuild();
        EditorApplication.Exit(0);
    }

    private static void EnsureParticleShadersIncluded() {
        UnityEngine.Object graphicsSettings = GraphicsSettings.GetGraphicsSettings();
        SerializedObject so = new SerializedObject(graphicsSettings);
        SerializedProperty shaders = so.FindProperty("m_AlwaysIncludedShaders");
        if (shaders == null || !shaders.isArray) {
            Debug.LogWarning("HazardLocalBuildRunner: could not find m_AlwaysIncludedShaders");
            return;
        }

        foreach (string shaderName in AlwaysIncludedShaderNames) {
            Shader shader = Shader.Find(shaderName);
            if (shader == null) {
                Debug.LogWarning("HazardLocalBuildRunner: shader not found: " + shaderName);
                continue;
            }
            bool alreadyIncluded = false;
            for (int i = 0; i < shaders.arraySize; i++) {
                SerializedProperty entry = shaders.GetArrayElementAtIndex(i);
                if (entry.objectReferenceValue == shader) {
                    alreadyIncluded = true;
                    break;
                }
            }
            if (!alreadyIncluded) {
                shaders.InsertArrayElementAtIndex(shaders.arraySize);
                shaders.GetArrayElementAtIndex(shaders.arraySize - 1).objectReferenceValue = shader;
            }
        }
        so.ApplyModifiedPropertiesWithoutUndo();
        AssetDatabase.SaveAssets();
    }
}
