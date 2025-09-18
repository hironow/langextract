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

