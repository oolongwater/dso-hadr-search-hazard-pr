using UnityEngine;

namespace UnityStandardAssets.Characters.FirstPerson {
    public partial class PhysicsRemoteFPSAgentController {
        public void StartHazardFire(ServerAction action) {
            if (string.IsNullOrEmpty(action.objectId)) {
                errorMessage = "StartHazardFire requires objectId";
                actionFinished(false);
                return;
            }

            SimObjPhysics target = getInteractableSimObjectFromId(
                objectId: action.objectId,
                forceAction: true
            );
            if (target == null) {
                errorMessage = "StartHazardFire: object not found";
                actionFinished(false);
                return;
            }

            float severity = action.moveMagnitude > 0.0f ? action.moveMagnitude : 0.7f;
            EnsureHazardEffects().StartFireAt(target.transform, severity);
            actionFinished(true);
        }

        public void StopHazardFire() {
            EnsureHazardEffects().StopFire();
            actionFinished(true);
        }

        public void SetSmokeDensity(float density = 0.0f) {
            EnsureHazardEffects().SetSmokeDensity(density);
            actionFinished(true);
        }
    }
}
