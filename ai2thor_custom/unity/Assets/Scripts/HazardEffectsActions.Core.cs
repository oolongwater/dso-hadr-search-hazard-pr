using UnityEngine;

namespace UnityStandardAssets.Characters.FirstPerson {
    /// <summary>
    /// Shared wiring for custom hazard actions dispatched from Python.
    /// </summary>
    public partial class PhysicsRemoteFPSAgentController {
        private HazardEffects hazardEffects;

        private HazardEffects EnsureHazardEffects() {
            if (hazardEffects == null) {
                hazardEffects = GetComponent<HazardEffects>();
                if (hazardEffects == null) {
                    hazardEffects = gameObject.AddComponent<HazardEffects>();
                }
                hazardEffects.Initialize(m_Camera);
            }
            return hazardEffects;
        }
    }
}
