"""FastAPI service tests — TestClient with a stub HierarchicalPredictor.

Skipped when the ``serve`` extra isn't installed (fastapi / uvicorn /
python-multipart). Fast to run; all endpoints exercise the predictor
through ``app.dependency_overrides`` rather than real checkpoints.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

fastapi = pytest.importorskip("fastapi")
starlette_testclient = pytest.importorskip("starlette.testclient")

from fastapi.testclient import TestClient  # noqa: E402
from mm_cxr_diag import __version__  # noqa: E402
from mm_cxr_diag.data import PATHOLOGY_LABELS  # noqa: E402
from mm_cxr_diag.inference import HierarchicalPrediction  # noqa: E402
from mm_cxr_diag.service.app import create_app  # noqa: E402
from mm_cxr_diag.service.dependencies import get_predictor  # noqa: E402


class StubStage:
    """Stand-in for SingleStagePredictor with just the attributes the
    service touches."""

    def __init__(self, name: str, num_classes: int):
        self.model_name = name
        self.num_classes = num_classes
        self._transform = None

    def predict_batch(self, images, tabular):
        import numpy as np

        if self.num_classes == 1:
            return np.array([[0.75]], dtype=np.float32)
        return np.full((1, self.num_classes), 0.4, dtype=np.float32)


class StubPredictor:
    """Stand-in for HierarchicalPredictor."""

    def __init__(self):
        self.stage1 = StubStage("densenet121", 1)
        self.stage2 = StubStage("densenet201", len(PATHOLOGY_LABELS))
        self.pathology_labels = PATHOLOGY_LABELS

    def predict(self, pil, tabular):
        return HierarchicalPrediction(
            abnormal_prob=0.75,
            abnormal=True,
            stage1_threshold=0.5,
            no_finding_prob=0.25,
            pathologies={lbl: 0.3 for lbl in PATHOLOGY_LABELS},
            pathology_thresholds={lbl: 0.5 for lbl in PATHOLOGY_LABELS},
            pathology_labels=PATHOLOGY_LABELS,
            skipped_stage2=False,
            stage1_model="densenet121",
            stage2_model="densenet201",
            latency_ms={"stage1": 1.0, "stage2": 2.0, "total": 3.0},
        )


@pytest.fixture
def client_with_predictor():
    app = create_app()
    # TestClient's context manager runs the lifespan, which sets
    # app.state.predictor = None (no checkpoints configured). We install
    # the stub AFTER that so it survives for the duration of the tests.
    with TestClient(app) as c:
        app.state.predictor = StubPredictor()
        app.state.device = "cpu"
        app.dependency_overrides[get_predictor] = lambda: app.state.predictor
        yield c


@pytest.fixture
def client_without_predictor():
    app = create_app()
    with TestClient(app) as c:
        app.state.device = "cpu"
        yield c


def _png_bytes(size: int = 32) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color=(128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


def _tabular_json() -> str:
    return '{"patientAge":0.6,"patientGender":1,"viewPosition":0,"followUpNumber":0.1}'


def test_health_ok_with_predictor(client_with_predictor):
    r = client_with_predictor.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["stage1_loaded"] is True
    assert body["stage2_loaded"] is True


def test_health_reports_unloaded(client_without_predictor):
    r = client_without_predictor.get("/health")
    assert r.status_code == 200  # process liveness; loaded flags tell the story
    body = r.json()
    assert body["stage1_loaded"] is False
    assert body["stage2_loaded"] is False


def test_version_returns_package_and_models(client_with_predictor):
    r = client_with_predictor.get("/version")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == __version__
    assert body["stage1_model"] == "densenet121"
    assert body["stage2_model"] == "densenet201"


def test_version_503_when_no_predictor(client_without_predictor):
    r = client_without_predictor.get("/version")
    assert r.status_code == 503


def test_predict_happy_path(client_with_predictor):
    files = {"image": ("x.png", _png_bytes(), "image/png")}
    data = {"tabular": _tabular_json()}
    r = client_with_predictor.post("/predict", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["abnormal_prob"] == pytest.approx(0.75)
    assert body["abnormal"] is True
    assert body["model_versions"] == {
        "stage1": "densenet121",
        "stage2": "densenet201",
    }
    assert len(body["pathologies"]) == len(PATHOLOGY_LABELS)


def test_predict_missing_image_422(client_with_predictor):
    r = client_with_predictor.post("/predict", data={"tabular": _tabular_json()})
    assert r.status_code == 422


def test_predict_missing_tabular_422(client_with_predictor):
    files = {"image": ("x.png", _png_bytes(), "image/png")}
    r = client_with_predictor.post("/predict", files=files)
    assert r.status_code == 422


def test_predict_bad_tabular_422(client_with_predictor):
    files = {"image": ("x.png", _png_bytes(), "image/png")}
    data = {"tabular": "not json"}
    r = client_with_predictor.post("/predict", files=files, data=data)
    assert r.status_code == 422
    assert "valid JSON" in r.json()["detail"]


def test_predict_unknown_tabular_key_422(client_with_predictor):
    files = {"image": ("x.png", _png_bytes(), "image/png")}
    data = {"tabular": '{"patientAge":0.6}'}  # missing keys → Pydantic fails
    r = client_with_predictor.post("/predict", files=files, data=data)
    assert r.status_code == 422


def test_predict_bad_image_422(client_with_predictor):
    files = {"image": ("x.png", b"not an image", "image/png")}
    data = {"tabular": _tabular_json()}
    r = client_with_predictor.post("/predict", files=files, data=data)
    assert r.status_code == 422


def test_predict_stage1_returns_gate_only(client_with_predictor):
    files = {"image": ("x.png", _png_bytes(), "image/png")}
    data = {"tabular": _tabular_json()}
    r = client_with_predictor.post("/predict/stage1", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["abnormal_prob"] == pytest.approx(0.75)
    assert body["stage1_model"] == "densenet121"
    assert "pathologies" not in body


def test_predict_stage2_returns_pathologies_only(client_with_predictor):
    files = {"image": ("x.png", _png_bytes(), "image/png")}
    data = {"tabular": _tabular_json()}
    r = client_with_predictor.post("/predict/stage2", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["pathologies"]) == set(PATHOLOGY_LABELS)
    assert body["stage2_model"] == "densenet201"
    assert "abnormal_prob" not in body


def test_predict_503_when_no_predictor(client_without_predictor):
    files = {"image": ("x.png", _png_bytes(), "image/png")}
    data = {"tabular": _tabular_json()}
    r = client_without_predictor.post("/predict", files=files, data=data)
    assert r.status_code == 503
