import importlib
import sys


def test_package_imports() -> None:
    package = importlib.import_module("dso_hadr")

    assert package.__version__ == "0.0.0"


def test_import_does_not_import_ai2thor() -> None:
    sys.modules.pop("dso_hadr", None)
    sys.modules.pop("ai2thor", None)

    package = importlib.import_module("dso_hadr")

    assert package is not None
    assert "ai2thor" not in sys.modules
