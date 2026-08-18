using UnityEngine;

namespace UnityStandardAssets.Characters.FirstPerson {
    /// <summary>
    /// Coordinator for custom hazard effects. Implementation is split across
    /// fire_smoke/ and earthquake/ partial class files.
    /// </summary>
    public partial class HazardEffects : MonoBehaviour {
        protected Camera agentCamera;
        protected Vector3 baseCameraLocalPos;
        protected Quaternion baseCameraLocalRot = Quaternion.identity;

        public void Initialize(Camera camera) {
            agentCamera = camera;
            if (agentCamera != null) {
                baseCameraLocalPos = agentCamera.transform.localPosition;
                baseCameraLocalRot = agentCamera.transform.localRotation;
            }
        }

        private void FixedUpdate() {
            FixedUpdateEarthquake();
            FixedUpdateFireSmoke();
        }
    }
}
