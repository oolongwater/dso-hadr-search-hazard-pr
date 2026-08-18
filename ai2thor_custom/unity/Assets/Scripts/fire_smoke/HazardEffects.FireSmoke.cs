using System.Collections.Generic;
using UnityEngine;

namespace UnityStandardAssets.Characters.FirstPerson {
    public partial class HazardEffects {
        private const float FlamePrefabScale = 8.0f;

        private static Texture2D softCircleTexture;
        private static Material fireParticleMaterial;
        private static Material smokeParticleMaterial;
        private static GameObject hazardFlamePrefab;

        private class FlameInstance {
            public GameObject root;
            public Light light;
            public ParticleSystem smoke;
            public float severity;
        }

        private readonly List<FlameInstance> flames = new List<FlameInstance>();
        private float globalSmokeDensity;
        private bool fogInitialized;
        private GameObject roomSmokeRoot;
        private ParticleSystem roomSmoke;

        public void StartFireAt(Transform target, float severity) {
            float clampedSeverity = Mathf.Clamp01(severity);
            Vector3 position = target != null
                ? target.position + Vector3.up * 0.05f
                : transform.position;

            FlameInstance flame = CreateFlameInstance(position, clampedSeverity);
            flames.Add(flame);
        }

        public void StopFire() {
            for (int i = flames.Count - 1; i >= 0; i--) {
                if (flames[i].root != null) {
                    Destroy(flames[i].root);
                }
            }
            flames.Clear();
        }

        public void SetSmokeDensity(float density) {
            globalSmokeDensity = Mathf.Clamp01(density);
            EnsureFog();
            EnsureRoomSmoke();

            float smokeRate = 6.0f + globalSmokeDensity * 28.0f;
            for (int i = 0; i < flames.Count; i++) {
                FlameInstance flame = flames[i];
                if (flame.smoke == null) {
                    continue;
                }
                ParticleSystem.EmissionModule emission = flame.smoke.emission;
                emission.rateOverTime = smokeRate * flame.severity;
            }

            if (roomSmoke != null) {
                ParticleSystem.EmissionModule roomEmission = roomSmoke.emission;
                roomEmission.rateOverTime = 4.0f + globalSmokeDensity * 18.0f;
            }

            RenderSettings.fog = globalSmokeDensity > 0.01f;
            RenderSettings.fogMode = FogMode.Exponential;
            RenderSettings.fogColor = Color.Lerp(
                Color.white,
                new Color(0.65f, 0.65f, 0.68f),
                globalSmokeDensity
            );
            RenderSettings.fogDensity = 0.02f + globalSmokeDensity * 0.28f;
        }

        private void FixedUpdateFireSmoke() {
            for (int i = 0; i < flames.Count; i++) {
                FlameInstance flame = flames[i];
                if (flame.light == null) {
                    continue;
                }
                float flicker = 0.85f + 0.3f * Mathf.PerlinNoise(Time.time * 8.0f, i * 0.37f);
                flame.light.intensity = 1.6f * flame.severity * flicker;
            }
        }

        private FlameInstance CreateFlameInstance(Vector3 position, float severity) {
            GameObject root = new GameObject("HazardFlameRoot");
            root.transform.position = position;

            Light light = null;
            GameObject prefab = GetHazardFlamePrefab();
            if (prefab != null) {
                GameObject flameVisual = Instantiate(prefab, root.transform);
                flameVisual.name = "HazardFlameVisual";
                flameVisual.transform.localPosition = Vector3.zero;
                flameVisual.transform.localRotation = Quaternion.identity;
                flameVisual.transform.localScale = Vector3.one * FlamePrefabScale;

                light = flameVisual.GetComponentInChildren<Light>();
                if (light != null) {
                    light.intensity = 1.6f * severity;
                    light.range = 3.5f;
                }

                ParticleSystem[] prefabParticles = flameVisual.GetComponentsInChildren<ParticleSystem>();
                for (int i = 0; i < prefabParticles.Length; i++) {
                    ParticleSystem.MainModule main = prefabParticles[i].main;
                    main.startSizeMultiplier = main.startSizeMultiplier * severity;
                }
            } else {
                CreateFireEmitter(root.transform);
                GameObject lightGo = new GameObject("HazardFireLight");
                lightGo.transform.SetParent(root.transform, false);
                light = lightGo.AddComponent<Light>();
                light.type = LightType.Point;
                light.color = new Color(1.0f, 0.55f, 0.15f);
                light.intensity = 1.6f * severity;
                light.range = 3.5f;
            }

            GameObject smokeGo = new GameObject("HazardFlameSmoke");
            smokeGo.transform.SetParent(root.transform, false);
            smokeGo.transform.localPosition = new Vector3(0.0f, 0.15f, 0.0f);
            ParticleSystem smoke = CreateFlameSmokeEmitter(smokeGo.transform, severity);

            return new FlameInstance {
                root = root,
                light = light,
                smoke = smoke,
                severity = severity,
            };
        }

        private static GameObject GetHazardFlamePrefab() {
            if (hazardFlamePrefab != null) {
                return hazardFlamePrefab;
            }
            hazardFlamePrefab = Resources.Load<GameObject>("HazardFlame");
            return hazardFlamePrefab;
        }

        private void EnsureFog() {
            if (fogInitialized) {
                return;
            }
            fogInitialized = true;
            RenderSettings.fog = false;
        }

        private void EnsureRoomSmoke() {
            if (roomSmokeRoot != null) {
                return;
            }
            roomSmokeRoot = new GameObject("HazardRoomSmoke");
            roomSmokeRoot.transform.SetParent(transform, false);
            roomSmokeRoot.transform.localPosition = new Vector3(0.0f, 2.2f, 0.0f);
            roomSmoke = CreateRoomSmokeEmitter(roomSmokeRoot.transform);
        }

        private static ParticleSystem CreateRoomSmokeEmitter(Transform parent) {
            ParticleSystem ps = parent.gameObject.AddComponent<ParticleSystem>();
            ps.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);

            ParticleSystem.MainModule main = ps.main;
            main.duration = 5.0f;
            main.loop = true;
            main.startLifetime = new ParticleSystem.MinMaxCurve(2.0f, 4.5f);
            main.startSpeed = new ParticleSystem.MinMaxCurve(0.05f, 0.25f);
            main.startSize = new ParticleSystem.MinMaxCurve(0.35f, 1.1f);
            main.startColor = new Color(0.58f, 0.58f, 0.6f, 0.22f);
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            main.maxParticles = 600;

            ParticleSystem.EmissionModule emission = ps.emission;
            emission.rateOverTime = 4.0f;

            ParticleSystem.ShapeModule shape = ps.shape;
            shape.shapeType = ParticleSystemShapeType.Box;
            shape.scale = new Vector3(4.5f, 0.35f, 4.5f);

            ParticleSystem.VelocityOverLifetimeModule velocity = ps.velocityOverLifetime;
            velocity.enabled = true;
            velocity.y = new ParticleSystem.MinMaxCurve(0.08f, 0.22f);

            ConfigureParticleRenderer(ps.GetComponent<ParticleSystemRenderer>(), GetSmokeParticleMaterial());

            ps.Play();
            return ps;
        }

        private static Texture2D GetSoftCircleTexture() {
            if (softCircleTexture != null) {
                return softCircleTexture;
            }
            const int size = 64;
            softCircleTexture = new Texture2D(size, size, TextureFormat.RGBA32, false);
            softCircleTexture.name = "HazardSoftCircle";
            softCircleTexture.wrapMode = TextureWrapMode.Clamp;
            softCircleTexture.filterMode = FilterMode.Bilinear;
            float center = (size - 1) * 0.5f;
            float radius = center;
            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    float dx = (x - center) / radius;
                    float dy = (y - center) / radius;
                    float dist = Mathf.Sqrt(dx * dx + dy * dy);
                    float alpha = Mathf.Clamp01(1.0f - dist);
                    alpha = alpha * alpha;
                    softCircleTexture.SetPixel(x, y, new Color(1.0f, 1.0f, 1.0f, alpha));
                }
            }
            softCircleTexture.Apply();
            return softCircleTexture;
        }

        private static Material GetFireParticleMaterial() {
            if (fireParticleMaterial != null) {
                return fireParticleMaterial;
            }
            Shader shader = Shader.Find("Legacy Shaders/Particles/Additive");
            if (shader == null) {
                shader = Shader.Find("Particles/Additive");
            }
            if (shader == null) {
                shader = Shader.Find("Sprites/Default");
            }
            fireParticleMaterial = new Material(shader);
            fireParticleMaterial.mainTexture = GetSoftCircleTexture();
            return fireParticleMaterial;
        }

        private static Material GetSmokeParticleMaterial() {
            if (smokeParticleMaterial != null) {
                return smokeParticleMaterial;
            }
            Shader shader = Shader.Find("Legacy Shaders/Particles/Alpha Blended");
            if (shader == null) {
                shader = Shader.Find("Particles/Alpha Blended");
            }
            if (shader == null) {
                shader = Shader.Find("Sprites/Default");
            }
            smokeParticleMaterial = new Material(shader);
            smokeParticleMaterial.mainTexture = GetSoftCircleTexture();
            return smokeParticleMaterial;
        }

        private static void ConfigureParticleRenderer(ParticleSystemRenderer renderer, Material material) {
            renderer.renderMode = ParticleSystemRenderMode.Billboard;
            renderer.material = material;
        }

        private static ParticleSystem CreateFireEmitter(Transform parent) {
            ParticleSystem ps = parent.gameObject.AddComponent<ParticleSystem>();
            ps.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);

            ParticleSystem.MainModule main = ps.main;
            main.duration = 5.0f;
            main.loop = true;
            main.startLifetime = new ParticleSystem.MinMaxCurve(0.35f, 0.9f);
            main.startSpeed = new ParticleSystem.MinMaxCurve(0.8f, 1.8f);
            main.startSize = new ParticleSystem.MinMaxCurve(0.08f, 0.22f);
            main.startColor = new ParticleSystem.MinMaxGradient(
                new Color(1.0f, 0.45f, 0.05f, 0.95f),
                new Color(1.0f, 0.85f, 0.15f, 0.85f)
            );
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            main.maxParticles = 400;
            main.gravityModifier = -0.15f;

            ParticleSystem.EmissionModule emission = ps.emission;
            emission.rateOverTime = 55.0f;

            ParticleSystem.ShapeModule shape = ps.shape;
            shape.shapeType = ParticleSystemShapeType.Cone;
            shape.angle = 18.0f;
            shape.radius = 0.08f;

            ParticleSystem.VelocityOverLifetimeModule velocity = ps.velocityOverLifetime;
            velocity.enabled = true;
            velocity.y = new ParticleSystem.MinMaxCurve(1.2f, 2.4f);

            ConfigureParticleRenderer(ps.GetComponent<ParticleSystemRenderer>(), GetFireParticleMaterial());

            ps.Play();
            return ps;
        }

        private static ParticleSystem CreateFlameSmokeEmitter(Transform parent, float severity) {
            ParticleSystem ps = parent.gameObject.AddComponent<ParticleSystem>();
            ps.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);

            ParticleSystem.MainModule main = ps.main;
            main.duration = 5.0f;
            main.loop = true;
            main.startLifetime = new ParticleSystem.MinMaxCurve(1.2f, 2.8f);
            main.startSpeed = new ParticleSystem.MinMaxCurve(0.15f, 0.55f);
            main.startSize = new ParticleSystem.MinMaxCurve(0.12f, 0.35f);
            main.startColor = new Color(0.55f, 0.55f, 0.58f, 0.35f);
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            main.maxParticles = 300;

            ParticleSystem.EmissionModule emission = ps.emission;
            emission.rateOverTime = 8.0f * severity;

            ParticleSystem.ShapeModule shape = ps.shape;
            shape.shapeType = ParticleSystemShapeType.Cone;
            shape.angle = 22.0f;
            shape.radius = 0.05f;

            ParticleSystem.VelocityOverLifetimeModule velocity = ps.velocityOverLifetime;
            velocity.enabled = true;
            velocity.y = new ParticleSystem.MinMaxCurve(0.4f, 1.0f);

            ConfigureParticleRenderer(ps.GetComponent<ParticleSystemRenderer>(), GetSmokeParticleMaterial());

            ps.Play();
            return ps;
        }
    }
}
