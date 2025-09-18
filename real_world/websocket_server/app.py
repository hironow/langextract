from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query

import langextract as lx
from langextract.core import data as lx_data
from real_world.common.extraction_config import (
    DEFAULT_PROMPT,
    DEFAULT_EXAMPLES,
    flatten_chat_messages,
)


app = FastAPI(title="LangExtract WebSocket Server")


def _env_api_key_for(model_id: str) -> dict[str, Any]:
  if model_id.startswith("gemini"):
    return {"api_key": os.getenv("GEMINI_API_KEY") or os.getenv("LANGEXTRACT_API_KEY")}
  if model_id.startswith(("gpt", "o1")):
    return {"api_key": os.getenv("OPENAI_API_KEY") or os.getenv("LANGEXTRACT_API_KEY")}
  return {"api_key": os.getenv("LANGEXTRACT_API_KEY")}


async def _extract_single_message(
    text: str,
    model_id: str,
    use_schema_constraints: bool,
    model_url: str | None,
    temperature: float | None,
) -> lx_data.AnnotatedDocument:
  provider_kwargs: dict[str, Any] = {
      **_env_api_key_for(model_id),
      "temperature": temperature,
  }
  if model_url:
    provider_kwargs["model_url"] = model_url
    provider_kwargs["base_url"] = model_url

  loop = asyncio.get_event_loop()
  # Run blocking extract() in a thread to keep the event loop responsive
  return await loop.run_in_executor(
      None,
      lambda: lx.extract(
          text_or_documents=text,
          prompt_description=DEFAULT_PROMPT,
          examples=DEFAULT_EXAMPLES,
          model_id=model_id,
          use_schema_constraints=use_schema_constraints,
          max_char_buffer=500,
          language_model_params=provider_kwargs,
          show_progress=False,
      ),
  )


@app.websocket("/ws")
async def ws_endpoint(
    websocket: WebSocket,
    model_id: str = Query(default=os.environ.get("DEFAULT_MODEL_ID", "gemini-2.5-flash")),
    use_schema_constraints: bool = Query(default=True),
    model_url: str | None = Query(default=None),
    temperature: float | None = Query(default=None),
    aggregate: bool = Query(default=False),
):
  await websocket.accept()
  conversation_messages: list[dict[str, str]] = []  # [{role, content}]
  try:
    await websocket.send_text(
        json.dumps({
            "type": "info",
            "message": f"connected; model_id={model_id}; aggregate={aggregate}",
        })
    )
    # Protocol: client sends frames like {"content": "message text"}
    while True:
      raw = await websocket.receive_text()
      try:
        data = json.loads(raw)
      except json.JSONDecodeError:
        await websocket.send_text(json.dumps({"type": "error", "message": "invalid json"}))
        continue

      # Handle end-of-turn/session delimiter without closing the socket.
      if data.get("eot") or data.get("type") == "eot":
        if aggregate:
          conversation_messages.clear()
        await websocket.send_text(json.dumps({"type": "eot_ack"}))
        continue

      role = (data.get("role") or "user").strip() or "user"
      content = (data.get("content") or "").strip()
      if not content:
        await websocket.send_text(json.dumps({"type": "warn", "message": "empty content"}))
        continue

      await websocket.send_text(json.dumps({"type": "status", "message": "processing"}))
      try:
        if aggregate:
          conversation_messages.append({"role": role, "content": content})
          text = flatten_chat_messages(conversation_messages)
          doc = await _extract_single_message(
              text, model_id, use_schema_constraints, model_url, temperature
          )
        else:
          doc = await _extract_single_message(
              content, model_id, use_schema_constraints, model_url, temperature
          )
        # Stream the result immediately for this message
        span_dicts: list[dict[str, Any]] = []
        for ext in doc.extractions or []:
          ci = ext.char_interval
          span_dicts.append({
              "extraction_class": ext.extraction_class,
              "extraction_text": ext.extraction_text,
              "char_start": (ci.start_pos if ci else None),
              "char_end": (ci.end_pos if ci else None),
              "attributes": ext.attributes,
          })
        await websocket.send_text(
            json.dumps({
                "type": "aggregate_result" if aggregate else "result",
                "text": doc.text,
                "extractions": span_dicts,
            })
        )
      except Exception as e:  # Return error for this specific message
        await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))

  except WebSocketDisconnect:
    # Client disconnected
    return
