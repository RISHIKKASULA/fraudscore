"""API contract tests (architecture.md §8), against an artifact trained on the fixture."""

from __future__ import annotations

import json
import math

import pytest
from fastapi.testclient import TestClient

from fraudscore.serve import create_app


@pytest.fixture(scope="module")
def client(trained_artifact_dir) -> TestClient:
    return TestClient(create_app(trained_artifact_dir))


@pytest.fixture(scope="module")
def fraud_like_v(synthetic_df) -> list[float]:
    """V vector of the first fraud row in the fixture — scores high under the model."""
    row = synthetic_df[synthetic_df["Class"] == 1].iloc[0]
    return [float(row[f"V{i}"]) for i in range(1, 29)]


def valid_payload(**overrides) -> dict:
    payload = {"amount": 149.62, "time": 40632.0, "v": [0.0] * 28}
    payload.update(overrides)
    return payload


class TestScoreHappyPath:
    def test_schema_and_decision_quantity(self, client):
        resp = client.post("/score", json=valid_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {
            "fraud_probability", "expected_fraud_cost", "decision", "decision_rule",
            "c_review", "model_version", "scored_at",
        }
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["expected_fraud_cost"] == pytest.approx(
            body["fraud_probability"] * 149.62
        )
        assert body["decision"] in {"approve", "review"}
        assert body["decision_rule"] == "expected_cost"
        assert body["c_review"] == 10.0
        assert body["scored_at"].endswith("Z")

    def test_decision_consistent_with_expected_cost(self, client):
        body = client.post("/score", json=valid_payload()).json()
        should_review = body["expected_fraud_cost"] >= body["c_review"]
        assert body["decision"] == ("review" if should_review else "approve")


class TestValidation:
    def test_missing_field_422(self, client):
        payload = valid_payload()
        del payload["amount"]
        resp = client.post("/score", json=payload)
        assert resp.status_code == 422
        assert any("amount" in str(err["loc"]) for err in resp.json()["detail"])

    def test_27_floats_422(self, client):
        assert client.post("/score", json=valid_payload(v=[0.0] * 27)).status_code == 422

    def test_29_floats_422(self, client):
        assert client.post("/score", json=valid_payload(v=[0.0] * 29)).status_code == 422

    def test_negative_amount_422(self, client):
        assert client.post("/score", json=valid_payload(amount=-1.0)).status_code == 422

    def test_negative_time_422(self, client):
        assert client.post("/score", json=valid_payload(time=-5.0)).status_code == 422

    def test_non_finite_values_422(self, client):
        # httpx won't serialize NaN/inf, so send raw JSON with the literal tokens —
        # exactly the malformed input a hostile client could deliver.
        def post_raw(payload: dict):
            body = json.dumps(payload)  # allow_nan=True -> emits NaN / Infinity
            return client.post("/score", content=body,
                               headers={"content-type": "application/json"})

        for bad in (math.nan, math.inf):
            assert post_raw(valid_payload(amount=bad)).status_code == 422
            v = [0.0] * 28
            v[7] = bad
            assert post_raw(valid_payload(v=v)).status_code == 422

    def test_extra_field_422(self, client):
        assert client.post("/score", json=valid_payload(extra=1)).status_code == 422


class TestDecisionBoundary:
    def test_decision_flips_across_c_review_boundary(self, client, fraud_like_v):
        """Same V vector; amounts straddling break-even flip approve <-> review.

        At amount $5 even p=1 gives expected cost $5 < $10 -> approve. At a large
        amount the same (high-scoring) transaction crosses c_review -> review.
        """
        small = client.post("/score", json=valid_payload(v=fraud_like_v, amount=5.0)).json()
        assert small["decision"] == "approve"

        large = client.post("/score", json=valid_payload(v=fraud_like_v, amount=50_000.0)).json()
        assert large["expected_fraud_cost"] >= large["c_review"]
        assert large["decision"] == "review"


class TestOperationalEndpoints:
    def test_health(self, client):
        assert client.get("/health").json() == {"status": "ok", "model_loaded": True}

    def test_model_info_matches_artifact(self, client, trained_artifact_dir):
        body = client.get("/model-info").json()
        card = json.loads((trained_artifact_dir / "model-card.json").read_text())
        assert body["version"] == card["version"]
        assert body["t_star"] == card["t_star"]  # baseline rule inspectable too
        assert body["cost_params"]["c_review"] == 10.0
