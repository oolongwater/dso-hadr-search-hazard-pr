using System.Collections.Generic;
using UnityEngine;

namespace UnityStandardAssets.Characters.FirstPerson {
    public partial class HazardEffects {
        private const float Gravity = 9.81f;
        private const float MuStatic = 0.4f;
        private const float MuKinetic = 0.35f;
        private const float WavePeak = 1.8f;
        private const float VerticalMotionScale = 0.4f;
        private const float MaxHorizontalSpeed = 12.0f;
        private const float GroundRayPadding = 0.08f;
        private const float SlipSpeedEps = 0.05f;
        private const float RampInSeconds = 1.0f;

        private struct BodyState {
            public Rigidbody rb;
            public bool wasKinematic;
            public CollisionDetectionMode collisionMode;
        }

        private bool earthquakeActive;
        private float earthquakePga = 4.0f;
        private float earthquakeFrequencyHz = 2.5f;
        private float elapsedPhysicsTime;
        private Vector3 cameraShakeOffset = Vector3.zero;
        private readonly List<BodyState> earthquakeBodies = new List<BodyState>();

        public void StartEarthquake(float magnitude, float frequencyHz) {
            earthquakeActive = true;
            earthquakePga = Mathf.Max(0.5f, magnitude);
            earthquakeFrequencyHz = Mathf.Max(0.5f, frequencyHz);
            elapsedPhysicsTime = 0.0f;
            CacheEarthquakeBodies();
        }

        public void StopEarthquake() {
            earthquakeActive = false;
            foreach (BodyState state in earthquakeBodies) {
                if (state.rb == null) {
                    continue;
                }
                state.rb.isKinematic = state.wasKinematic;
                state.rb.collisionDetectionMode = state.collisionMode;
            }
            earthquakeBodies.Clear();
            cameraShakeOffset = Vector3.zero;
            if (agentCamera != null) {
                agentCamera.transform.localPosition = baseCameraLocalPos;
                agentCamera.transform.localRotation = baseCameraLocalRot;
            }
        }

        private bool IsSimulatable(SimObjPhysics sop) {
            if (sop == null) {
                return false;
            }
            if (
                sop.Type == SimObjType.Towel
                || sop.Type == SimObjType.HandTowel
                || sop.Type == SimObjType.ToiletPaper
            ) {
                if (sop.GetComponentInParent<ObjectSpecificReceptacle>() != null) {
                    return false;
                }
            }
            return sop.PrimaryProperty == SimObjPrimaryProperty.CanPickup
                || sop.PrimaryProperty == SimObjPrimaryProperty.Moveable;
        }

        private void CacheEarthquakeBodies() {
            earthquakeBodies.Clear();
            SimObjPhysics[] simObjects = GameObject.FindObjectsOfType<SimObjPhysics>();
            foreach (SimObjPhysics sop in simObjects) {
                if (sop == null || sop.gameObject == null || !IsSimulatable(sop)) {
                    continue;
                }
                Rigidbody rb = sop.GetComponent<Rigidbody>();
                if (rb == null) {
                    continue;
                }
                BodyState state = new BodyState {
                    rb = rb,
                    wasKinematic = rb.isKinematic,
                    collisionMode = rb.collisionDetectionMode,
                };
                rb.isKinematic = false;
                rb.collisionDetectionMode = CollisionDetectionMode.ContinuousSpeculative;
                rb.WakeUp();
                earthquakeBodies.Add(state);
            }
        }

        private void FixedUpdateEarthquake() {
            if (!earthquakeActive) {
                return;
            }
            ApplyEarthquakeStep(Time.fixedDeltaTime);
        }

        private float AxisWave(float t, float baseFreq, float phaseOffset) {
            float f = baseFreq;
            return Mathf.Sin(2.0f * Mathf.PI * f * t + phaseOffset)
                + 0.55f * Mathf.Sin(2.0f * Mathf.PI * f * 1.7f * t + phaseOffset * 1.3f)
                + 0.25f * Mathf.Sin(2.0f * Mathf.PI * f * 0.43f * t + phaseOffset * 0.7f);
        }

        private Vector3 ComputeGroundAcceleration(float t) {
            float f = earthquakeFrequencyHz;
            float ramp = Mathf.Clamp01(t / RampInSeconds);
            float pga = earthquakePga * ramp / WavePeak;
            float ax = AxisWave(t, f, 0.0f) * pga;
            float az = AxisWave(t, f, 1.7f) * pga * 0.85f;
            float ay = AxisWave(t, f * 1.9f, 2.4f) * pga * VerticalMotionScale;
            return new Vector3(ax, ay, az);
        }

        private Vector3 ComputeGroundDisplacement(float t) {
            float f = earthquakeFrequencyHz;
            float ramp = Mathf.Clamp01(t / RampInSeconds);
            float amp = earthquakePga * 0.004f * ramp / WavePeak;
            float dx = AxisWave(t, f, 0.0f) * amp;
            float dz = AxisWave(t, f, 1.7f) * amp * 0.85f;
            float dy = AxisWave(t, f * 1.9f, 2.4f) * amp * 0.35f;
            return new Vector3(dx, dy, dz);
        }

        private void ApplyEarthquakeStep(float dt) {
            elapsedPhysicsTime += dt;
            Vector3 groundAccel = ComputeGroundAcceleration(elapsedPhysicsTime);
            ApplyContactCoupledForces(groundAccel);

            Vector3 disp = ComputeGroundDisplacement(elapsedPhysicsTime);
            float pitch = disp.z * 8.0f;
            float roll = disp.x * 10.0f;
            cameraShakeOffset = new Vector3(disp.x, disp.y, 0.0f);
            if (agentCamera != null) {
                agentCamera.transform.localPosition = baseCameraLocalPos + cameraShakeOffset;
                agentCamera.transform.localRotation = baseCameraLocalRot
                    * Quaternion.Euler(pitch, 0.0f, roll);
            }
        }

        private bool IsGrounded(Rigidbody rb, out Vector3 basePoint) {
            basePoint = Vector3.zero;
            Collider col = rb.GetComponent<Collider>();
            if (col == null) {
                return false;
            }
            Bounds bounds = col.bounds;
            basePoint = new Vector3(bounds.center.x, bounds.min.y, bounds.center.z);
            float rayLen = bounds.extents.y + GroundRayPadding;
            RaycastHit[] hits = Physics.RaycastAll(
                bounds.center,
                Vector3.down,
                rayLen,
                ~0,
                QueryTriggerInteraction.Ignore
            );
            foreach (RaycastHit hit in hits) {
                if (hit.rigidbody == rb) {
                    continue;
                }
                return true;
            }
            return false;
        }

        private void ClampHorizontalSpeed(Rigidbody rb) {
            Vector3 vel = rb.velocity;
            Vector2 horizontal = new Vector2(vel.x, vel.z);
            if (horizontal.sqrMagnitude <= MaxHorizontalSpeed * MaxHorizontalSpeed) {
                return;
            }
            horizontal = horizontal.normalized * MaxHorizontalSpeed;
            rb.velocity = new Vector3(horizontal.x, vel.y, horizontal.y);
        }

        private Vector3 ComputeFrictionAccel(
            Rigidbody rb,
            Vector3 basePoint,
            Vector3 fictitiousAccel
        ) {
            Vector3 fictH = new Vector3(fictitiousAccel.x, 0.0f, fictitiousAccel.z);
            float normalAccel = Mathf.Max(0.0f, Gravity - fictitiousAccel.y);
            if (normalAccel <= 0.01f) {
                return Vector3.zero;
            }

            Vector3 velH = new Vector3(rb.velocity.x, 0.0f, rb.velocity.z);
            if (velH.sqrMagnitude <= SlipSpeedEps * SlipSpeedEps) {
                float fictMag = fictH.magnitude;
                float stickLimit = MuStatic * normalAccel;
                if (fictMag <= stickLimit) {
                    return -fictH;
                }
                return -fictH.normalized * stickLimit;
            }

            return -velH.normalized * (MuKinetic * normalAccel);
        }

        private void ApplyContactCoupledForces(Vector3 groundAccel) {
            Vector3 fictitiousAccel = -groundAccel;

            foreach (BodyState state in earthquakeBodies) {
                Rigidbody rb = state.rb;
                if (rb == null || rb.isKinematic) {
                    continue;
                }
                rb.WakeUp();

                rb.AddForce(fictitiousAccel * rb.mass, ForceMode.Force);

                Vector3 basePoint;
                if (IsGrounded(rb, out basePoint)) {
                    Vector3 frictionAccel = ComputeFrictionAccel(rb, basePoint, fictitiousAccel);
                    if (frictionAccel.sqrMagnitude > 0.0f) {
                        rb.AddForceAtPosition(
                            frictionAccel * rb.mass,
                            basePoint,
                            ForceMode.Force
                        );
                    }
                }

                ClampHorizontalSpeed(rb);
            }
        }
    }
}
