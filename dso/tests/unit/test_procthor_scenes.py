from __future__ import annotations

import json
from pathlib import Path

import pytest

from dso_hadr.scenes.procthor import (
    CorpusConfig,
    SceneManifest,
    create_manifest,
    load_corpus_config,
    load_manifest,
    subset_records,
    validate_corpus,
    write_manifest,
)


def _config(
    *,
    visualization_test_scene_count: int = 2,
    visualization_test_scene_ids: tuple[str, ...] = ("fixture_01", "fixture_04"),
) -> CorpusConfig:
    return CorpusConfig(
        dataset_id="fixture-corpus",
        generator="ProcTHOR",
        expected_schema_by_floor_count={1: "1.0.0", 2: "2.0.0", 3: "2.0.0"},
        first_scene_index=1,
        scene_count=5,
        filename_template="fixture_{index:02d}.json",
        visualization_test_scene_count=visualization_test_scene_count,
        visualization_test_scene_ids=visualization_test_scene_ids,
    )


def _scene(scene_id: str, room_count: int = 1) -> dict[str, object]:
    rooms = []
    for index in range(room_count):
        offset = float(index * 2)
        rooms.append(
            {
                "id": f"room|{index + 1}",
                "roomType": "LivingRoom",
                "floorPolygon": [
                    {"x": offset, "y": 0.0, "z": 0.0},
                    {"x": offset + 1.0, "y": 0.0, "z": 0.0},
                    {"x": offset + 1.0, "y": 0.0, "z": 1.0},
                ],
            }
        )
    return {
        "id": scene_id,
        "metadata": {
            "schema": "1.0.0",
            "roomSpecId": "test-room-spec",
            "warnings": {},
            "agent": {"position": {}, "rotation": {}},
        },
        "rooms": rooms,
        "walls": [],
        "doors": [],
        "windows": [],
        "objects": [],
    }


def _write_scenes(path: Path, config: CorpusConfig) -> None:
    path.mkdir()
    for index in range(
        config.first_scene_index,
        config.first_scene_index + config.scene_count,
    ):
        filename = config.filename_for_index(index)
        (path / filename).write_text(
            json.dumps(_scene(Path(filename).stem, room_count=(index % 3) + 1)),
            encoding="utf-8",
        )


def test_loads_all_corpus_parameters_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / "corpus.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_id": "fixture-corpus",
                "source": {"generator": "ProcTHOR"},
                "scene_selection": {
                    "first_index": 3,
                    "count": 2,
                    "filename_template": "house-{index}.json",
                },
                "validation": {
                    "expected_house_schema_by_floor_count": {
                        "1": "1.0.0",
                        "2": "2.0.0",
                        "3": "2.0.0",
                    },
                },
                "split": {
                    "visualization_test_count": 1,
                    "visualization_test_scene_ids": ["house-4"],
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_corpus_config(config_path)

    assert config.first_scene_index == 3
    assert config.scene_count == 2
    assert config.filename_for_index(4) == "house-4.json"
    assert config.visualization_test_scene_ids == ("house-4",)


def test_create_write_load_and_validate_manifest(tmp_path: Path) -> None:
    config = _config()
    scenes_dir = tmp_path / "scenes"
    _write_scenes(scenes_dir, config)
    manifest = create_manifest(scenes_dir, config)
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest, manifest_path)

    loaded = load_manifest(manifest_path, config)
    validated = validate_corpus(loaded, scenes_dir, config)

    assert loaded == manifest
    assert len(validated) == 5
    assert len(loaded.development_scene_ids) == 3
    assert loaded.visualization_test_scene_ids == ("fixture_01", "fixture_04")
    assert len(subset_records(loaded, "development")) == 3
    assert len(subset_records(loaded, "visualization-test")) == 2


def test_records_multifloor_schema_and_layout_spec(tmp_path: Path) -> None:
    config = _config()
    scenes_dir = tmp_path / "scenes"
    _write_scenes(scenes_dir, config)
    scene_path = scenes_dir / "fixture_01.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    scene["metadata"] = {
        "schema": "2.0.0",
        "numFloors": 2,
        "houseSpecId": "test-house-spec",
    }
    scene["floors"] = [{}, {}]
    scene["verticalConnectors"] = [{}]
    scene_path.write_text(json.dumps(scene), encoding="utf-8")

    manifest = create_manifest(scenes_dir, config)

    assert manifest.scenes[0].floor_count == 2
    assert manifest.scenes[0].layout_spec_id == "test-house-spec"


def test_validate_rejects_changed_scene_content(tmp_path: Path) -> None:
    config = _config()
    scenes_dir = tmp_path / "scenes"
    _write_scenes(scenes_dir, config)
    manifest = create_manifest(scenes_dir, config)
    changed_path = scenes_dir / "fixture_01.json"
    changed = json.loads(changed_path.read_text(encoding="utf-8"))
    changed["metadata"]["roomSpecId"] = "changed"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the content manifest"):
        validate_corpus(manifest, scenes_dir, config)


def test_create_rejects_vertical_connector_in_one_floor_scene(tmp_path: Path) -> None:
    config = _config()
    scenes_dir = tmp_path / "scenes"
    _write_scenes(scenes_dir, config)
    scene_path = scenes_dir / "fixture_01.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    scene["verticalConnectors"] = [{}]
    scene_path.write_text(json.dumps(scene), encoding="utf-8")

    with pytest.raises(ValueError, match="one-floor scene contains vertical connectors"):
        create_manifest(scenes_dir, config)


def test_split_count_and_id_list_must_match() -> None:
    with pytest.raises(ValueError, match="length must match"):
        _config(
            visualization_test_scene_count=1,
            visualization_test_scene_ids=("fixture_01", "fixture_04"),
        )


def test_subset_rejects_unknown_name() -> None:
    manifest = SceneManifest(
        dataset_id="test",
        generator="ProcTHOR",
        scenes=(),
        visualization_test_scene_ids=(),
    )

    with pytest.raises(KeyError):
        subset_records(manifest, "training")
