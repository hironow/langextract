from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import langextract as lx
from langextract.core import data as lx_data
from real_world.common.extraction_config import (
    DEFAULT_PROMPT,
    DEFAULT_EXAMPLES,
    flatten_chat_messages,
)


app = FastAPI(title="LangExtract REST API")


class Message(BaseModel):
  role: str
  content: str


class ExtractRequest(BaseModel):
  messages: list[Message] = Field(default_factory=list)
  model_id: str = Field(
      default=os.environ.get("DEFAULT_MODEL_ID", "gemini-2.5-flash")
  )
  use_schema_constraints: bool = True
  max_char_buffer: int = 1000
  temperature: Optional[float] = None
  model_url: Optional[str] = None  # for Ollama/proxied backends


class ExtractionSpan(BaseModel):
  extraction_class: str
  extraction_text: str
  char_start: Optional[int] = None
  char_end: Optional[int] = None
  attributes: Optional[dict[str, Any]] = None


class ExtractResponse(BaseModel):
  model_id: str
  text: str
  extractions: list[ExtractionSpan]


def _env_api_key_for(model_id: str) -> dict[str, Any]:
  """Pick API key by provider from env; fallback to LANGEXTRACT_API_KEY.

  This mirrors factory behavior but lets callers override explicitly.
  """
  # Allow explicit fallback (factory also does this), but ok to pass here
  if model_id.startswith("gemini"):
    return {"api_key": os.getenv("GEMINI_API_KEY") or os.getenv("LANGEXTRACT_API_KEY")}
  if model_id.startswith(("gpt", "o1")):
    return {"api_key": os.getenv("OPENAI_API_KEY") or os.getenv("LANGEXTRACT_API_KEY")}
  # Ollama (local) typically needs no key; proxies might require one
  return {"api_key": os.getenv("LANGEXTRACT_API_KEY")}


@app.post("/extract", response_model=ExtractResponse)
def extract_endpoint(payload: ExtractRequest) -> ExtractResponse:
  if not payload.messages:
    raise HTTPException(status_code=400, detail="messages must be provided")

  text = flatten_chat_messages([m.model_dump() for m in payload.messages])

  # Build provider kwargs
  provider_kwargs: dict[str, Any] = {
      **_env_api_key_for(payload.model_id),
      "temperature": payload.temperature,
  }
  if payload.model_url:
    # Supports Ollama/proxy endpoints (mapped to both model_url and base_url)
    provider_kwargs["model_url"] = payload.model_url
    provider_kwargs["base_url"] = payload.model_url

  try:
    result: lx_data.AnnotatedDocument = lx.extract(
        text_or_documents=text,
        prompt_description=DEFAULT_PROMPT,
        examples=DEFAULT_EXAMPLES,
        model_id=payload.model_id,
        use_schema_constraints=payload.use_schema_constraints,
        max_char_buffer=payload.max_char_buffer,
        language_model_params=provider_kwargs,
        show_progress=False,
    )
  except Exception as e:  # Pydantic response aligns error handling above
    raise HTTPException(status_code=500, detail=str(e)) from e

  spans: list[ExtractionSpan] = []
  for ext in result.extractions or []:
    spans.append(
        ExtractionSpan(
            extraction_class=ext.extraction_class,
            extraction_text=ext.extraction_text,
            char_start=(ext.char_interval.start_pos if ext.char_interval else None),
            char_end=(ext.char_interval.end_pos if ext.char_interval else None),
            attributes=ext.attributes,
        )
    )

  return ExtractResponse(model_id=payload.model_id, text=result.text or text, extractions=spans)


@app.get("/health")
def health() -> dict[str, str]:
  return {"status": "ok"}

