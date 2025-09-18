import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from real_world.rest_api.app import app as rest_app
from real_world.websocket_server.app import app as ws_app


def _mock_resolve_to_ollama(monkeypatch: pytest.MonkeyPatch):
  """Force factory to pick Ollama provider regardless of model_id."""
  from langextract.providers import ollama
  monkeypatch.setattr(
      "langextract.providers.router.resolve",
      lambda model_id: ollama.OllamaLanguageModel,
  )


class _Resp:
  def __init__(self, payload: dict[str, Any]):
    self.status_code = 200
    self._json = payload
    self.encoding = 'utf-8'

  def json(self) -> dict[str, Any]:
    return self._json


def _mock_requests_post(monkeypatch: pytest.MonkeyPatch):
  """Mock requests.post used by Ollama provider to return a JSON string body.

  The returned JSON simulates Ollama's `{ "response": "..." }` where the
  inner string is a JSON document containing {"extractions": [...]}.
  """

  def fake_post(url: str, headers: dict, json: dict, timeout: int):  # noqa: A002
    prompt = json.get("prompt", "")
    # Choose a token dependent on prompt content for aggregate tests
    token = "first" if "first" in prompt else ("second" if "second" in prompt else "tok")
    inner = {
        "extractions": [
            {"token": token, "token_attributes": {}},
        ]
    }
    return _Resp({"response": json_dumps(inner)})

  # Provide json.dumps locally to avoid shadowing
  def json_dumps(obj: Any) -> str:
    return json.dumps(obj)

  import requests  # type: ignore
  monkeypatch.setattr(requests, "post", fake_post)


@pytest.mark.parametrize("model_id", ["gemma2:2b", "meta-llama/Llama-3.2-1B-Instruct"])
def test_e2e_rest_given_messages_when_post_then_end_to_end(monkeypatch: pytest.MonkeyPatch, model_id: str):
  """E2E (REST): Given chat messages, When POST /extract, Then we get extractions.

  Given
    - REST app and Ollama provider forced via router.resolve
  When
    - POST /extract with messages and model_url
  Then
    - Response contains an extraction item parsed end-to-end
  """

  _mock_resolve_to_ollama(monkeypatch)
  _mock_requests_post(monkeypatch)

  client = TestClient(rest_app)
  resp = client.post(
      "/extract",
      json={
          "model_id": model_id,
          "model_url": "http://localhost:11434",
          "messages": [
              {"role": "user", "content": "first"},
              {"role": "assistant", "content": "second"},
          ],
      },
  )
  assert resp.status_code == 200
  data = resp.json()
  assert data["extractions"] and data["extractions"][0]["extraction_class"] == "token"
  assert data["extractions"][0]["extraction_text"] in ("first", "second", "tok")


@pytest.mark.parametrize("aggregate", [False, True])
def test_e2e_ws_given_frames_when_send_then_end_to_end(monkeypatch: pytest.MonkeyPatch, aggregate: bool):
  """E2E (WebSocket): Given frames, When send, Then parsed results stream back.

  Given
    - WebSocket app with Ollama provider forced via router.resolve
  When
    - Client sends one or two frames depending on `aggregate`
  Then
    - We receive status and then a result frame (type depends on `aggregate`)
  """

  _mock_resolve_to_ollama(monkeypatch)
  _mock_requests_post(monkeypatch)

  client = TestClient(ws_app)
  qs = f"?model_id=gemma2:2b&model_url=http://localhost:11434&aggregate={'true' if aggregate else 'false'}"
  with client.websocket_connect(f"/ws{qs}") as ws:
    # info
    info = json.loads(ws.receive_text())
    assert info["type"] == "info"

    # first frame
    ws.send_text(json.dumps({"role": "user", "content": "first"}))
    status = json.loads(ws.receive_text())
    assert status["type"] == "status"
    res1 = json.loads(ws.receive_text())
    assert res1["type"] in ("result", "aggregate_result")
    assert res1["extractions"] and res1["extractions"][0]["extraction_class"] == "token"

    if aggregate:
      # second frame influences prompt; mock returns token="second"
      ws.send_text(json.dumps({"role": "user", "content": "second"}))
      _ = json.loads(ws.receive_text())  # status
      res2 = json.loads(ws.receive_text())
      assert res2["type"] == "aggregate_result"
      assert any(item["extraction_text"] in ("first", "second") for item in res2["extractions"])


def test_e2e_ws_persistent_sessions_with_eot(monkeypatch: pytest.MonkeyPatch):
  """E2E (WebSocket): Keep connection open and delimit sessions with eot twice.

  Then we should observe three separate aggregated conversations over one socket.
  """
  _mock_resolve_to_ollama(monkeypatch)
  _mock_requests_post(monkeypatch)

  client = TestClient(ws_app)
  qs = "?model_id=gemma2:2b&model_url=http://localhost:11434&aggregate=true"
  with client.websocket_connect(f"/ws{qs}") as ws:
    _ = json.loads(ws.receive_text())  # info

    # Conversation 1
    ws.send_text(json.dumps({"role": "user", "content": "first-1"}))
    _ = json.loads(ws.receive_text())  # status
    c1 = json.loads(ws.receive_text())
    assert c1["type"] == "aggregate_result"
    assert "user: first-1" in c1["text"]

    # eot #1
    ws.send_text(json.dumps({"eot": True}))
    ack1 = json.loads(ws.receive_text())
    assert ack1["type"] == "eot_ack"

    # Conversation 2
    ws.send_text(json.dumps({"role": "user", "content": "second-1"}))
    _ = json.loads(ws.receive_text())  # status
    c2 = json.loads(ws.receive_text())
    assert c2["type"] == "aggregate_result"
    assert "user: second-1" in c2["text"]
    assert "first-1" not in c2["text"]  # history was reset

    # eot #2
    ws.send_text(json.dumps({"type": "eot"}))
    ack2 = json.loads(ws.receive_text())
    assert ack2["type"] == "eot_ack"

    # Conversation 3
    ws.send_text(json.dumps({"role": "user", "content": "third-1"}))
    _ = json.loads(ws.receive_text())  # status
    c3 = json.loads(ws.receive_text())
    assert c3["type"] == "aggregate_result"
    assert "user: third-1" in c3["text"]
    assert all(s not in c3["text"] for s in ("first-1", "second-1"))
