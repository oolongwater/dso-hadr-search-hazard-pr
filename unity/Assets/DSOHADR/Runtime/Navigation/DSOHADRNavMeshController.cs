using System;
using System.Collections.Generic;
using System.Linq;
using Thor.Procedural;
using UnityEngine;
using UnityEngine.AI;

namespace UnityStandardAssets.Characters.FirstPerson {
    [Serializable]
    public class DSOHADRNavMeshTriangulation {
        public int agentTypeId;
        public Vector3[] vertices;
        public int[] indices;
        public int[] areas;
        public int[] adjacency;
    }

    internal struct DSOHADRQuantizedVertex : IEquatable<DSOHADRQuantizedVertex> {
        private const float Scale = 100000.0f;
        private readonly int x;
        private readonly int y;
        private readonly int z;

        public DSOHADRQuantizedVertex(Vector3 value) {
            x = Mathf.RoundToInt(value.x * Scale);
            y = Mathf.RoundToInt(value.y * Scale);
            z = Mathf.RoundToInt(value.z * Scale);
        }

        public bool Equals(DSOHADRQuantizedVertex other) {
            return x == other.x && y == other.y && z == other.z;
        }

        public override bool Equals(object value) {
            return value is DSOHADRQuantizedVertex
                && Equals((DSOHADRQuantizedVertex)value);
        }

        public override int GetHashCode() {
            unchecked {
                var hash = 17;
                hash = hash * 31 + x;
                hash = hash * 31 + y;
                hash = hash * 31 + z;
                return hash;
            }
        }
    }

    internal struct DSOHADRNavMeshEdge : IEquatable<DSOHADRNavMeshEdge> {
        private readonly int first;
        private readonly int second;

        public DSOHADRNavMeshEdge(int vertexA, int vertexB) {
            first = Math.Min(vertexA, vertexB);
            second = Math.Max(vertexA, vertexB);
        }

        public bool Equals(DSOHADRNavMeshEdge other) {
            return first == other.first && second == other.second;
        }

        public override bool Equals(object value) {
            return value is DSOHADRNavMeshEdge
                && Equals((DSOHADRNavMeshEdge)value);
        }

        public override int GetHashCode() {
            unchecked {
                return first * 397 ^ second;
            }
        }
    }

    public partial class PhysicsRemoteFPSAgentController : BaseFPSAgentController {
        private static int[] DSOHADRNavMeshAdjacency(
            NavMeshTriangulation triangulation,
            int agentTypeId
        ) {
            var canonicalByPosition =
                new Dictionary<DSOHADRQuantizedVertex, int>();
            var canonicalByRawVertex = new int[triangulation.vertices.Length];
            for (var rawVertex = 0; rawVertex < triangulation.vertices.Length; rawVertex++) {
                var position = new DSOHADRQuantizedVertex(
                    triangulation.vertices[rawVertex]
                );
                int canonicalVertex;
                if (!canonicalByPosition.TryGetValue(position, out canonicalVertex)) {
                    canonicalVertex = canonicalByPosition.Count;
                    canonicalByPosition[position] = canonicalVertex;
                }
                canonicalByRawVertex[rawVertex] = canonicalVertex;
            }

            var trianglesByEdge = new Dictionary<DSOHADRNavMeshEdge, List<int>>();
            var triangleCount = triangulation.indices.Length / 3;
            for (var triangle = 0; triangle < triangleCount; triangle++) {
                var offset = triangle * 3;
                var vertices = new[] {
                    canonicalByRawVertex[triangulation.indices[offset]],
                    canonicalByRawVertex[triangulation.indices[offset + 1]],
                    canonicalByRawVertex[triangulation.indices[offset + 2]]
                };
                for (var edge = 0; edge < 3; edge++) {
                    var key = new DSOHADRNavMeshEdge(
                        vertices[edge],
                        vertices[(edge + 1) % 3]
                    );
                    List<int> triangles;
                    if (!trianglesByEdge.TryGetValue(key, out triangles)) {
                        triangles = new List<int>();
                        trianglesByEdge[key] = triangles;
                    }
                    triangles.Add(triangle);
                }
            }

            var queryFilter = new NavMeshQueryFilter() {
                agentTypeID = agentTypeId,
                areaMask = NavMesh.AllAreas
            };
            var accepted = new List<int[]>();
            var seen = new HashSet<long>();
            foreach (var triangles in trianglesByEdge.Values) {
                for (var first = 0; first < triangles.Count; first++) {
                    for (var second = first + 1; second < triangles.Count; second++) {
                        var triangleA = Math.Min(triangles[first], triangles[second]);
                        var triangleB = Math.Max(triangles[first], triangles[second]);
                        var pairKey = ((long)triangleA << 32) | (uint)triangleB;
                        if (!seen.Add(pairKey)) {
                            continue;
                        }
                        var offsetA = triangleA * 3;
                        var offsetB = triangleB * 3;
                        var centroidA = (
                            triangulation.vertices[triangulation.indices[offsetA]]
                            + triangulation.vertices[triangulation.indices[offsetA + 1]]
                            + triangulation.vertices[triangulation.indices[offsetA + 2]]
                        ) / 3.0f;
                        var centroidB = (
                            triangulation.vertices[triangulation.indices[offsetB]]
                            + triangulation.vertices[triangulation.indices[offsetB + 1]]
                            + triangulation.vertices[triangulation.indices[offsetB + 2]]
                        ) / 3.0f;
                        var normalA = Vector3.Cross(
                            triangulation.vertices[triangulation.indices[offsetA + 1]]
                                - triangulation.vertices[triangulation.indices[offsetA]],
                            triangulation.vertices[triangulation.indices[offsetA + 2]]
                                - triangulation.vertices[triangulation.indices[offsetA]]
                        ).normalized;
                        var normalB = Vector3.Cross(
                            triangulation.vertices[triangulation.indices[offsetB + 1]]
                                - triangulation.vertices[triangulation.indices[offsetB]],
                            triangulation.vertices[triangulation.indices[offsetB + 2]]
                                - triangulation.vertices[triangulation.indices[offsetB]]
                        ).normalized;
                        if (
                            (Mathf.Abs(normalA.y) >= 0.999f)
                            != (Mathf.Abs(normalB.y) >= 0.999f)
                        ) {
                            continue;
                        }
                        var horizontalDistance = Vector2.Distance(
                            new Vector2(centroidA.x, centroidA.z),
                            new Vector2(centroidB.x, centroidB.z)
                        );
                        if (
                            Mathf.Abs(centroidB.y - centroidA.y)
                                > horizontalDistance * 0.8f + 0.05f
                        ) {
                            continue;
                        }
                        NavMeshHit hit;
                        if (!NavMesh.Raycast(
                            centroidA,
                            centroidB,
                            out hit,
                            queryFilter
                        )) {
                            accepted.Add(new[] { triangleA, triangleB });
                        }
                    }
                }
            }
            return accepted
                .OrderBy(pair => pair[0])
                .ThenBy(pair => pair[1])
                .SelectMany(pair => pair)
                .ToArray();
        }

        public void DSOHADRGetNavMeshTriangulation(int? navMeshId = null) {
            var navmeshSurfaces = GameObject.FindObjectsOfType<NavMeshSurfaceExtended>(
                includeInactive: true
            );
            if (navmeshSurfaces.Length == 0) {
                actionFinishedEmit(
                    success: false,
                    errorMessage: "No runtime NavMeshSurfaceExtended was found."
                );
                return;
            }

            try {
                var selectedSurface = ProceduralTools.activateOnlyNavmeshSurface(
                    navmeshSurfaces,
                    navMeshId
                );
                var triangulation = NavMesh.CalculateTriangulation();
                if (triangulation.vertices.Length == 0 || triangulation.indices.Length == 0) {
                    actionFinishedEmit(
                        success: false,
                        errorMessage: "The selected runtime navmesh has no triangulation."
                    );
                    return;
                }
                actionFinishedEmit(
                    success: true,
                    actionReturn: new DSOHADRNavMeshTriangulation {
                        agentTypeId = selectedSurface.agentTypeID,
                        vertices = triangulation.vertices,
                        indices = triangulation.indices,
                        areas = triangulation.areas,
                        adjacency = DSOHADRNavMeshAdjacency(
                            triangulation,
                            selectedSurface.agentTypeID
                        ),
                    }
                );
            } catch (Exception exception) {
                actionFinishedEmit(success: false, errorMessage: exception.Message);
            } finally {
                ProceduralTools.activateAllNavmeshSurfaces(navmeshSurfaces);
            }
        }
    }
}
