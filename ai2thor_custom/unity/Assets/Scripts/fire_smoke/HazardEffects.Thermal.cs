using System.Collections.Generic;
using UnityEngine;

namespace UnityStandardAssets.Characters.FirstPerson {
    public partial class HazardEffects {
        private float thermalCenterX;
        private float thermalCenterZ;
        private float thermalHalfExtentX;
        private float thermalHalfExtentZ;
        private int thermalNx;
        private int thermalNz;
        private float thermalAmbientC = 22.0f;
        private float thermalFlameCoreC = 600.0f;
        private float thermalDiffusivity = 0.15f;
        private float thermalConvectionGain = 2.0f;
        private float thermalCoolingRate = 0.05f;
        private float thermalFlameRadiusM = 1.2f;
        private float thermalHotThresholdC = 70.0f;
        private float thermalCellDx;
        private float thermalCellDz;
        private float[] thermalTemperatureC;
        private float[] thermalScratchC;
        private bool thermalInitialized;

        public void SetThermalParams(
            float centerX = 0.0f,
            float centerZ = 0.0f,
            float halfExtentX = 3.0f,
            float halfExtentZ = 3.0f,
            int resolutionX = 64,
            int resolutionZ = 48,
            float ambientC = 22.0f,
            float flameCoreC = 600.0f,
            float diffusivity = 0.15f,
            float convectionGain = 2.0f,
            float coolingRate = 0.05f,
            float flameRadiusM = 1.2f,
            float hotThresholdC = 70.0f
        ) {
            thermalCenterX = centerX;
            thermalCenterZ = centerZ;
            thermalHalfExtentX = Mathf.Max(0.5f, halfExtentX);
            thermalHalfExtentZ = Mathf.Max(0.5f, halfExtentZ);
            thermalNx = Mathf.Max(4, resolutionX);
            thermalNz = Mathf.Max(4, resolutionZ);
            thermalAmbientC = ambientC;
            thermalFlameCoreC = flameCoreC;
            thermalDiffusivity = Mathf.Max(0.01f, diffusivity);
            thermalConvectionGain = Mathf.Max(0.0f, convectionGain);
            thermalCoolingRate = Mathf.Max(0.0f, coolingRate);
            thermalFlameRadiusM = Mathf.Max(0.1f, flameRadiusM);
            thermalHotThresholdC = hotThresholdC;
            thermalCellDx = (2.0f * thermalHalfExtentX) / thermalNx;
            thermalCellDz = (2.0f * thermalHalfExtentZ) / thermalNz;

            int count = thermalNx * thermalNz;
            if (thermalTemperatureC == null || thermalTemperatureC.Length != count) {
                thermalTemperatureC = new float[count];
                thermalScratchC = new float[count];
            }
            for (int i = 0; i < count; i++) {
                thermalTemperatureC[i] = thermalAmbientC;
                thermalScratchC[i] = thermalAmbientC;
            }
            thermalInitialized = true;
        }

        public Dictionary<string, object> AdvanceHeatField(float deltaTime) {
            if (!thermalInitialized) {
                return BuildHeatFieldReturn(0.0f, 0.0f);
            }
            float dt = Mathf.Max(0.0f, deltaTime);
            if (dt <= 0.0f) {
                SampleObjects();
                return BuildHeatFieldReturn(0.0f, dt);
            }

            float totalSeverity = 0.0f;
            for (int i = 0; i < flames.Count; i++) {
                totalSeverity += flames[i].severity;
            }
            float alphaEff = thermalDiffusivity * (1.0f + thermalConvectionGain * totalSeverity);
            float hMin = Mathf.Min(thermalCellDx, thermalCellDz);
            float cflDt = (hMin * hMin) / (4.0f * alphaEff);
            int subSteps = Mathf.Max(4, Mathf.CeilToInt(dt / cflDt));
            float subDt = dt / subSteps;

            for (int step = 0; step < subSteps; step++) {
                ApplyFlameInjection(subDt);
                DiffuseTemperature(subDt, alphaEff);
                ApplyAmbientCooling(subDt);
                ClampTemperatures();
            }

            SampleObjects();
            return BuildHeatFieldReturn(totalSeverity, dt);
        }

        private void ApplyFlameInjection(float dt) {
            if (flames.Count == 0) {
                return;
            }
            float injectRate = 90.0f;
            for (int iz = 0; iz < thermalNz; iz++) {
                for (int ix = 0; ix < thermalNx; ix++) {
                    int idx = iz * thermalNx + ix;
                    Vector3 cellCenter = CellCenterWorld(ix, iz);
                    for (int f = 0; f < flames.Count; f++) {
                        FlameInstance flame = flames[f];
                        if (flame.root == null) {
                            continue;
                        }
                        Vector3 flamePos = flame.root.transform.position;
                        float dx = cellCenter.x - flamePos.x;
                        float dz = cellCenter.z - flamePos.z;
                        float dist = Mathf.Sqrt(dx * dx + dz * dz);
                        if (dist > thermalFlameRadiusM) {
                            continue;
                        }
                        float falloff = 1.0f - (dist / thermalFlameRadiusM);
                        float rate = injectRate * flame.severity * falloff * dt;
                        thermalTemperatureC[idx] += rate * (thermalFlameCoreC - thermalTemperatureC[idx]);
                    }
                }
            }
        }

        private void DiffuseTemperature(float dt, float alphaEff) {
            float invDx2 = 1.0f / (thermalCellDx * thermalCellDx);
            float invDz2 = 1.0f / (thermalCellDz * thermalCellDz);

            for (int iz = 0; iz < thermalNz; iz++) {
                for (int ix = 0; ix < thermalNx; ix++) {
                    int idx = iz * thermalNx + ix;
                    float center = thermalTemperatureC[idx];
                    float left = ix > 0 ? thermalTemperatureC[idx - 1] : center;
                    float right = ix < thermalNx - 1 ? thermalTemperatureC[idx + 1] : center;
                    float down = iz > 0 ? thermalTemperatureC[idx - thermalNx] : center;
                    float up = iz < thermalNz - 1 ? thermalTemperatureC[idx + thermalNx] : center;
                    float laplacian =
                        (right - 2.0f * center + left) * invDx2
                        + (up - 2.0f * center + down) * invDz2;
                    thermalScratchC[idx] = center + alphaEff * dt * laplacian;
                }
            }

            float[] swap = thermalTemperatureC;
            thermalTemperatureC = thermalScratchC;
            thermalScratchC = swap;
        }

        private void ApplyAmbientCooling(float dt) {
            for (int i = 0; i < thermalTemperatureC.Length; i++) {
                float delta = thermalTemperatureC[i] - thermalAmbientC;
                thermalTemperatureC[i] -= thermalCoolingRate * delta * dt;
            }
        }

        private void ClampTemperatures() {
            float maxAllowed = thermalFlameCoreC * 1.05f;
            for (int i = 0; i < thermalTemperatureC.Length; i++) {
                thermalTemperatureC[i] = Mathf.Clamp(
                    thermalTemperatureC[i],
                    thermalAmbientC,
                    maxAllowed
                );
            }
        }

        private Vector3 CellCenterWorld(int ix, int iz) {
            float x = thermalCenterX - thermalHalfExtentX + (ix + 0.5f) * thermalCellDx;
            float z = thermalCenterZ - thermalHalfExtentZ + (iz + 0.5f) * thermalCellDz;
            return new Vector3(x, 0.0f, z);
        }

        private float SampleTemperatureAt(float worldX, float worldZ) {
            if (!thermalInitialized || thermalTemperatureC == null) {
                return thermalAmbientC;
            }
            float normX = (worldX - (thermalCenterX - thermalHalfExtentX)) / (2.0f * thermalHalfExtentX);
            float normZ = (worldZ - (thermalCenterZ - thermalHalfExtentZ)) / (2.0f * thermalHalfExtentZ);
            if (normX < 0.0f || normX > 1.0f || normZ < 0.0f || normZ > 1.0f) {
                return thermalAmbientC;
            }

            float gx = normX * (thermalNx - 1);
            float gz = normZ * (thermalNz - 1);
            int ix0 = Mathf.Clamp(Mathf.FloorToInt(gx), 0, thermalNx - 2);
            int iz0 = Mathf.Clamp(Mathf.FloorToInt(gz), 0, thermalNz - 2);
            float tx = gx - ix0;
            float tz = gz - iz0;

            float t00 = thermalTemperatureC[iz0 * thermalNx + ix0];
            float t10 = thermalTemperatureC[iz0 * thermalNx + ix0 + 1];
            float t01 = thermalTemperatureC[(iz0 + 1) * thermalNx + ix0];
            float t11 = thermalTemperatureC[(iz0 + 1) * thermalNx + ix0 + 1];
            float t0 = Mathf.Lerp(t00, t10, tx);
            float t1 = Mathf.Lerp(t01, t11, tx);
            return Mathf.Lerp(t0, t1, tz);
        }

        private void SampleObjects() {
            SimObjPhysics[] simObjects = GameObject.FindObjectsOfType<SimObjPhysics>();
            foreach (SimObjPhysics sop in simObjects) {
                if (sop == null || sop.gameObject == null) {
                    continue;
                }
                Vector3 pos = sop.transform.position;
                float tempC = SampleTemperatureAt(pos.x, pos.z);
                if (tempC < thermalHotThresholdC) {
                    continue;
                }

                sop.CurrentTemperature = Temperature.Hot;
                if (sop.HowManySecondsUntilRoomTemp != sop.GetTimerResetValue()) {
                    sop.HowManySecondsUntilRoomTemp = sop.GetTimerResetValue();
                }
                sop.SetStartRoomTempTimer(false);

                if (sop.DoesThisObjectHaveThisSecondaryProperty(SimObjSecondaryProperty.CanBeCooked)) {
                    CookObject cook = sop.GetComponent<CookObject>();
                    if (cook != null && !cook.IsCooked()) {
                        cook.Cook();
                    }
                }
            }
        }

        private Dictionary<string, object> BuildHeatFieldReturn(float totalSeverity, float deltaTime) {
            float maxC = thermalAmbientC;
            float sumC = 0.0f;
            if (thermalTemperatureC != null) {
                for (int i = 0; i < thermalTemperatureC.Length; i++) {
                    float t = thermalTemperatureC[i];
                    sumC += t;
                    if (t > maxC) {
                        maxC = t;
                    }
                }
            }
            float meanC = thermalTemperatureC != null && thermalTemperatureC.Length > 0
                ? sumC / thermalTemperatureC.Length
                : thermalAmbientC;

            List<Dictionary<string, object>> flameList = new List<Dictionary<string, object>>();
            for (int i = 0; i < flames.Count; i++) {
                if (flames[i].root == null) {
                    continue;
                }
                Vector3 p = flames[i].root.transform.position;
                flameList.Add(new Dictionary<string, object> {
                    ["x"] = p.x,
                    ["z"] = p.z,
                    ["severity"] = flames[i].severity,
                });
            }

            List<Dictionary<string, object>> objectList = new List<Dictionary<string, object>>();
            SimObjPhysics[] simObjects = GameObject.FindObjectsOfType<SimObjPhysics>();
            foreach (SimObjPhysics sop in simObjects) {
                if (sop == null || sop.gameObject == null) {
                    continue;
                }
                Vector3 pos = sop.transform.position;
                float tempC = SampleTemperatureAt(pos.x, pos.z);
                objectList.Add(new Dictionary<string, object> {
                    ["objectId"] = sop.ObjectID,
                    ["x"] = pos.x,
                    ["z"] = pos.z,
                    ["temperatureC"] = tempC,
                    ["thorTemperature"] = sop.CurrentTemperature.ToString(),
                });
            }

            float[] tempsCopy = new float[thermalTemperatureC.Length];
            System.Array.Copy(thermalTemperatureC, tempsCopy, thermalTemperatureC.Length);

            return new Dictionary<string, object> {
                ["centerX"] = thermalCenterX,
                ["centerZ"] = thermalCenterZ,
                ["halfExtentX"] = thermalHalfExtentX,
                ["halfExtentZ"] = thermalHalfExtentZ,
                ["nx"] = thermalNx,
                ["nz"] = thermalNz,
                ["ambientC"] = thermalAmbientC,
                ["maxC"] = maxC,
                ["meanC"] = meanC,
                ["totalFlameSeverity"] = totalSeverity,
                ["deltaTime"] = deltaTime,
                ["temperatures"] = tempsCopy,
                ["flames"] = flameList,
                ["objects"] = objectList,
            };
        }
    }
}
