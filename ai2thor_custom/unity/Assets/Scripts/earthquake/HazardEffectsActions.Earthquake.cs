namespace UnityStandardAssets.Characters.FirstPerson {
    public partial class PhysicsRemoteFPSAgentController {
        public void StartEarthquake(float magnitude = 2.0f, float frequencyHz = 2.0f) {
            EnsureHazardEffects().StartEarthquake(magnitude, frequencyHz);
            actionFinished(true);
        }

        public void StopEarthquake() {
            EnsureHazardEffects().StopEarthquake();
            actionFinished(true);
        }
    }
}
