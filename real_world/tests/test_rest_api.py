import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

import langextract as lx
from langextract.core import data as lx_data
from real_world.rest_api.app import app


@pytest.fixture()
def client() -> TestClient:
  return TestClient(app)


def _fake_doc(text: str) -> lx_data.AnnotatedDocument:
  # Create a simple single-span extraction with attributes and char interval
  start = text.find("ibuprofen")
  end = start + len("ibuprofen") if start >= 0 else None
  extraction = lx_data.Extraction(
      extraction_class="medication",
      extraction_text="ibuprofen",
      char_interval=(lx_data.CharInterval(start_pos=start, end_pos=end) if start >= 0 else None),
      attributes={"unit": "mg"},
  )
  return lx_data.AnnotatedDocument(text=text, extractions=[extraction])


@pytest.mark.parametrize(
    "model_id,model_url,temperature",
    [
        ("gemini-2.5-flash", None, 0.0),
        ("gemma2:2b", "http://proxy:11434", 0.3),
    ],
)
def test_rest_extract_given_messages_when_post_then_returns_extractions(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, model_id: str, model_url: str | None, temperature: float
):
  """Given chat messages, When POST /extract, Then returns mapped extractions and forwards provider kwargs.

  Given
    - A list of chat-style messages (role, content)
  When
    - The client calls POST /extract with model_id/model_url/temperature
  Then
    - The service flattens messages and returns extractions mapped to JSON
    - The underlying extract() receives language_model_params incl. model_url/base_url when provided
  """

  recorded: dict[str, Any] = {}

  def fake_extract(**kwargs):  # type: ignore[no-redef]
    recorded.update(kwargs)
    text = kwargs["text_or_documents"]
    return _fake_doc(text)

  monkeypatch.setattr(lx, "extract", lambda **kw: fake_extract(**kw))

  body = {
      "model_id": model_id,
      "messages": [
          {"role": "user", "content": "Patient took 400 mg ibuprofen."},
          {"role": "assistant", "content": "Anything else?"},
      ],
      "use_schema_constraints": True,
      "max_char_buffer": 500,
      "temperature": temperature,
  }
  if model_url:
    body["model_url"] = model_url

  resp = client.post("/extract", json=body)
  assert resp.status_code == 200
  data = resp.json()

  # Then: JSON maps extractions with attributes and char positions
  assert data["model_id"] == model_id
  assert "extractions" in data and isinstance(data["extractions"], list)
  assert data["extractions"][0]["extraction_class"] == "medication"
  assert data["extractions"][0]["extraction_text"] == "ibuprofen"
  assert data["extractions"][0]["attributes"]["unit"] == "mg"

  # Then: provider kwargs forwarded
  lm_params = recorded.get("language_model_params", {})
  assert lm_params.get("temperature") == temperature
  if model_url:
    assert lm_params.get("model_url") == model_url
    assert lm_params.get("base_url") == model_url

