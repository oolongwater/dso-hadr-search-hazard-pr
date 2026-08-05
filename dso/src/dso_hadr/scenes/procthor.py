"""Load and validate the configured ProcTHOR scene corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CorpusConfig:
    """Corpus identity, selection, validation policy, and fixed split."""

    dataset_id: str
    generator: str
    expected_schema_by_floor_count: dict[int, str]
    first_scene_index: int
    scene_count: int
    filename_template: str
    visualization_test_scene_count: int
    visualization_test_scene_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.visualization_test_scene_ids) != self.visualization_test_scene_count:
            raise ValueError(
                "visualization_test_scene_ids length must match visualization_test_scene_count"
            )

    def filename_for_index(self, index: int) -> str:
        return self.filename_template.format(index=index)


@dataclass(frozen=True)
class SceneRecord:
    """Content-locked metadata for one generated ProcTHOR house."""

    scene_id: str
    filename: str
    sha256: str
    schema: str
    floor_count: int
    layout_spec_id: str
    room_count: int
    object_count: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "scene_id": self.scene_id,
            "filename": self.filename,
            "sha256": self.sha256,
            "schema": self.schema,
            "floor_count": self.floor_count,
            "layout_spec_id": self.layout_spec_id,
            "room_count": self.room_count,
            "object_count": self.object_count,
        }


@dataclass(frozen=True)
class SceneManifest:
    """Content lock and configured visualization/test subset."""

    dataset_id: str
    generator: str
    scenes: tuple[SceneRecord, ...]
    visualization_test_scene_ids: tuple[str, ...]

    @property
    def development_scene_ids(self) -> tuple[str, ...]:
        held_out = set(self.visualization_test_scene_ids)
        return tuple(scene.scene_id for scene in self.scenes if scene.scene_id not in held_out)


def load_corpus_config(path: Path) -> CorpusConfig:
    """Load corpus parameters and IDs from JSON."""

    document = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    source = document["source"]
    selection = document["scene_selection"]
    validation = document["validation"]
    split = document["split"]
    return CorpusConfig(
        dataset_id=document["dataset_id"],
        generator=source["generator"],
        expected_schema_by_floor_count={
            int(count): schema
            for count, schema in validation["expected_house_schema_by_floor_count"].items()
        },
        first_scene_index=selection["first_index"],
        scene_count=selection["count"],
        filename_template=selection["filename_template"],
        visualization_test_scene_count=split["visualization_test_count"],
        visualization_test_scene_ids=tuple(split["visualization_test_scene_ids"]),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scene_record(path: Path, config: CorpusConfig) -> SceneRecord:
    scene = json.loads(path.read_text(encoding="utf-8"))
    metadata = scene["metadata"]
    schema = metadata["schema"]

    single_floor_schema = config.expected_schema_by_floor_count[1]
    multi_floor_schemas = {
        expected_schema
        for floor_count, expected_schema in config.expected_schema_by_floor_count.items()
        if floor_count > 1
    }
    if schema == single_floor_schema:
        floor_count = 1
        layout_spec_id = metadata["roomSpecId"]
        if scene.get("verticalConnectors"):
            raise ValueError(f"{path}: a one-floor scene contains vertical connectors")
    elif schema in multi_floor_schemas:
        floor_count = metadata["numFloors"]
        layout_spec_id = metadata["houseSpecId"]
        if len(scene["floors"]) != floor_count:
            raise ValueError(f"{path}: floors length does not match metadata.numFloors")
        if len(scene["verticalConnectors"]) != floor_count - 1:
            raise ValueError(f"{path}: vertical connector count does not match floor count")
    else:
        raise ValueError(f"{path}: unsupported ProcTHOR schema {schema!r}")

    if floor_count not in config.expected_schema_by_floor_count:
        raise ValueError(f"{path}: unsupported floor count {floor_count}")
    expected_schema = config.expected_schema_by_floor_count[floor_count]
    if schema != expected_schema:
        raise ValueError(f"{path}: expected ProcTHOR schema {expected_schema!r}, got {schema!r}")

    rooms = scene["rooms"]
    objects = scene["objects"]
    return SceneRecord(
        scene_id=path.stem,
        filename=path.name,
        sha256=_sha256(path),
        schema=schema,
        floor_count=floor_count,
        layout_spec_id=layout_spec_id,
        room_count=len(rooms),
        object_count=len(objects),
    )


def _validate_split(manifest: SceneManifest, config: CorpusConfig) -> None:
    if manifest.visualization_test_scene_ids != config.visualization_test_scene_ids:
        raise ValueError("manifest visualization/test IDs do not match the corpus config")
    scene_ids = {scene.scene_id for scene in manifest.scenes}
    unknown = set(manifest.visualization_test_scene_ids) - scene_ids
    if unknown:
        raise ValueError(f"visualization/test split references unknown scenes: {sorted(unknown)}")


def create_manifest(scenes_dir: Path, config: CorpusConfig) -> SceneManifest:
    """Create a content lock from the configured scene files."""

    resolved_dir = scenes_dir.expanduser().resolve(strict=True)
    scenes = tuple(
        _scene_record(resolved_dir / config.filename_for_index(index), config)
        for index in range(
            config.first_scene_index,
            config.first_scene_index + config.scene_count,
        )
    )
    manifest = SceneManifest(
        dataset_id=config.dataset_id,
        generator=config.generator,
        scenes=scenes,
        visualization_test_scene_ids=config.visualization_test_scene_ids,
    )
    _validate_split(manifest, config)
    return manifest


def write_manifest(manifest: SceneManifest, path: Path) -> None:
    """Serialize a manifest deterministically."""

    document = {
        "dataset_id": manifest.dataset_id,
        "generator": manifest.generator,
        "scene_count": len(manifest.scenes),
        "visualization_test_scene_ids": list(manifest.visualization_test_scene_ids),
        "scenes": [scene.to_dict() for scene in manifest.scenes],
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path, config: CorpusConfig) -> SceneManifest:
    """Load the content lock and compare it with the corpus config."""

    document = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    scenes = tuple(
        SceneRecord(
            scene_id=record["scene_id"],
            filename=record["filename"],
            sha256=record["sha256"],
            schema=record["schema"],
            floor_count=record["floor_count"],
            layout_spec_id=record["layout_spec_id"],
            room_count=record["room_count"],
            object_count=record["object_count"],
        )
        for record in document["scenes"]
    )
    declared_count = document["scene_count"]
    if declared_count != len(scenes) or declared_count != config.scene_count:
        raise ValueError("manifest scene count does not match its records and corpus config")

    manifest = SceneManifest(
        dataset_id=document["dataset_id"],
        generator=document["generator"],
        scenes=scenes,
        visualization_test_scene_ids=tuple(document["visualization_test_scene_ids"]),
    )
    if manifest.dataset_id != config.dataset_id:
        raise ValueError("manifest dataset_id does not match the corpus config")
    if manifest.generator != config.generator:
        raise ValueError("manifest generator does not match the corpus config")
    _validate_split(manifest, config)
    return manifest


def validate_corpus(
    manifest: SceneManifest,
    scenes_dir: Path,
    config: CorpusConfig,
) -> tuple[SceneRecord, ...]:
    """Compare each configured scene with its content lock."""

    resolved_dir = scenes_dir.expanduser().resolve(strict=True)
    actual_records = tuple(
        _scene_record(resolved_dir / expected.filename, config) for expected in manifest.scenes
    )
    for expected, actual in zip(manifest.scenes, actual_records, strict=True):
        if actual != expected:
            raise ValueError(f"{actual.filename} does not match the content manifest")
    return actual_records


def subset_records(manifest: SceneManifest, subset: str) -> tuple[SceneRecord, ...]:
    held_out = set(manifest.visualization_test_scene_ids)
    subsets = {
        "all": manifest.scenes,
        "development": tuple(scene for scene in manifest.scenes if scene.scene_id not in held_out),
        "visualization-test": tuple(
            scene for scene in manifest.scenes if scene.scene_id in held_out
        ),
    }
    return subsets[subset]
