using System.Collections.Generic;
using UnityEngine;

namespace UnityStandardAssets.Characters.FirstPerson {
    public partial class PhysicsRemoteFPSAgentController {
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
            EnsureHazardEffects().SetThermalParams(
                centerX,
                centerZ,
                halfExtentX,
                halfExtentZ,
                resolutionX,
                resolutionZ,
                ambientC,
                flameCoreC,
                diffusivity,
                convectionGain,
                coolingRate,
                flameRadiusM,
                hotThresholdC
            );
            actionFinished(true);
        }

        public void AdvanceHeatField(float deltaTime = 0.2f) {
            Dictionary<string, object> result = EnsureHazardEffects().AdvanceHeatField(deltaTime);
            actionFinished(success: true, actionReturn: result);
        }
    }
}
