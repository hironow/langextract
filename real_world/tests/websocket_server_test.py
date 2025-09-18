import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from langextract.core import data as lx_data
from real_world.websocket_server.app import app


@pytest.fixture()
def client() -> TestClient:
  return TestClient(app)


def _doc_with_text(text: str) -> lx_data.AnnotatedDocument:
  ex = lx_data.Extraction(
      extraction_class="token",
      extraction_text=text[:5] or "",
      char_interval=lx_data.CharInterval(start_pos=0, end_pos=min(5, len(text))),
      attributes={"len": len(text)},
  )
  return lx_data.AnnotatedDocument(text=text, extractions=[ex])


@pytest.mark.parametrize("aggregate", [False, True])
def test_websocket_given_messages_when_send_then_streams_extractions(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, aggregate: bool
):
  """Given a WebSocket connection, When client sends messages, Then server streams results.

  Given
    - WebSocket connected with aggregate flag
  When
    - Client sends one (aggregate=False) or two (aggregate=True) frames
  Then
    - Server emits status and a result frame; type differs by aggregate flag
    - For aggregate=True, the text contains the conversation history
  """

  # Patch the internal async extractor to return text back for assertion
  async def fake_extract_single_message(text: str, *args, **kwargs):
    return _doc_with_text(text)

  monkeypatch.setattr(
      "real_world.websocket_server.app._extract_single_message",
      fake_extract_single_message,
  )

  qs = f"?model_id=gemini-2.5-flash&aggregate={'true' if aggregate else 'false'}"
  with client.websocket_connect(f"/ws{qs}") as ws:
    # receive info
    info = json.loads(ws.receive_text())
    assert info["type"] == "info"

    # send 1st message
    ws.send_text(json.dumps({"role": "user", "content": "first"}))
    status = json.loads(ws.receive_text())
    assert status["type"] == "status"
    result = json.loads(ws.receive_text())
    if aggregate:
      assert result["type"] == "aggregate_result"
      assert "user: first" in result["text"]
    else:
      assert result["type"] == "result"
      assert result["text"] == "first"

    if aggregate:
      # send 2nd message for aggregate mode
      ws.send_text(json.dumps({"role": "user", "content": "second"}))
      _ = json.loads(ws.receive_text())  # status
      agg2 = json.loads(ws.receive_text())
      assert agg2["type"] == "aggregate_result"
      # Conversation text should include both messages with roles
      assert "user: first" in agg2["text"] and "user: second" in agg2["text"]


def test_websocket_given_invalid_json_when_send_then_error_frame(
    client: TestClient,
):
  # Given: websocket connection
  with client.websocket_connect("/ws?model_id=gemini-2.5-flash") as ws:
    _ = ws.receive_text()  # info

    # When: sending invalid JSON
    ws.send_text("not a json string")
    res = json.loads(ws.receive_text())

    # Then: error frame
    assert res["type"] == "error"
    assert "invalid json" in res.get("message", "").lower()


def test_websocket_given_empty_content_when_send_then_warn_frame(
    client: TestClient,
):
  # Given: websocket connection
  with client.websocket_connect("/ws?model_id=gemini-2.5-flash") as ws:
    _ = ws.receive_text()  # info

    # When: send empty content
    ws.send_text(json.dumps({"content": ""}))
    res = json.loads(ws.receive_text())

    # Then: warn frame about empty content
    assert res["type"] == "warn"
    assert "empty content" in res.get("message", "").lower()


def test_websocket_given_no_role_in_aggregate_when_send_then_role_defaults_user(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
  # Patch model to echo text
  async def fake_extract_single_message(text: str, *_, **__):
    return _doc_with_text(text)

  monkeypatch.setattr(
      "real_world.websocket_server.app._extract_single_message",
      fake_extract_single_message,
  )

  with client.websocket_connect("/ws?model_id=gemini-2.5-flash&aggregate=true") as ws:
    _ = json.loads(ws.receive_text())  # info
    ws.send_text(json.dumps({"content": "hello"}))  # no role
    _ = json.loads(ws.receive_text())  # status
    result = json.loads(ws.receive_text())
    assert result["type"] == "aggregate_result"
    assert "user: hello" in result["text"]  # role defaults to user


def test_websocket_given_model_error_when_send_then_error_frame_and_continue(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
  # Given: first call fails, second succeeds
  state = {"count": 0}

  async def flaky_extract(text: str, *_, **__):
    state["count"] += 1
    if state["count"] == 1:
      raise RuntimeError("boom")
    return _doc_with_text(text)

  monkeypatch.setattr(
      "real_world.websocket_server.app._extract_single_message",
      flaky_extract,
  )

  with client.websocket_connect("/ws?model_id=gemini-2.5-flash") as ws:
    _ = json.loads(ws.receive_text())  # info
    # When: first message triggers error
    ws.send_text(json.dumps({"content": "first"}))
    _ = json.loads(ws.receive_text())  # status
    err = json.loads(ws.receive_text())
    # Then: error frame returned
    assert err["type"] == "error"
    assert "boom" in err.get("message", "")

    # When: second message should work
    ws.send_text(json.dumps({"role": "user", "content": "ok"}))
    _ = json.loads(ws.receive_text())  # status
    res = json.loads(ws.receive_text())
    assert res["type"] == "result"
    assert res["extractions"]


def test_websocket_given_query_params_when_send_then_forwarded_to_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
  recorded = {}

  async def recorder(text: str, model_id: str, use_schema_constraints: bool, model_url: str | None, temperature: float | None):
    recorded["text"] = text
    recorded["model_id"] = model_id
    recorded["use_schema_constraints"] = use_schema_constraints
    recorded["model_url"] = model_url
    recorded["temperature"] = temperature
    return _doc_with_text(text)

  monkeypatch.setattr(
      "real_world.websocket_server.app._extract_single_message",
      recorder,
  )

  with client.websocket_connect(
      "/ws?model_id=gemma2:2b&model_url=http://proxy:11434&temperature=0.7"
  ) as ws:
    _ = json.loads(ws.receive_text())  # info
    ws.send_text(json.dumps({"content": "hello"}))
    _ = json.loads(ws.receive_text())  # status
    _ = json.loads(ws.receive_text())  # result

    assert recorded["model_id"] == "gemma2:2b"
    assert recorded["model_url"] == "http://proxy:11434"
    assert recorded["temperature"] == 0.7
    assert recorded["use_schema_constraints"] is True
