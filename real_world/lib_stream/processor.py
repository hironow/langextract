from __future__ import annotations

"""Library-only streaming processor that uses LangExtract internally.

Scenario
  - Your library accepts an incoming stream of text chunks/messages (sync or async)
    and you want to produce incremental extractions while hiding the details
    (e.g., it internally uses Gemini via google-genai, but callers only see
    a stable Python API).

Design
  - Coalesce small chunks into a buffer and call LangExtract when: (a) the
    buffer grows beyond a threshold or (b) a flush() is requested by caller.
  - Merge extractions across calls (first-come wins on overlaps) and yield the
    new extractions since last yield.
  - Keep dependencies and provider config private to this implementation.
"""

import asyncio
from typing import Any, AsyncIterator, Iterator, Iterable, Optional

import langextract as lx
from langextract.core import data as lx_data
from real_world.common.extraction_config import (
    DEFAULT_PROMPT,
    DEFAULT_EXAMPLES,
)


class StreamExtractor:
  """Incremental extractor for streaming inputs.

  Parameters
    model_id: Provider/model selection. Defaults to Gemini flash.
    use_schema_constraints: Enables structured output for Gemini; safe for OpenAI/Ollama.
    max_char_buffer: Per-call max chunk size passed to LangExtract (affects latency).
    temperature: Provider sampling temperature (optional).
    model_url: For self-hosted endpoints (Ollama/proxy). Ignored if not needed.
    language_model_params: Extra provider kwargs (overrides env-based defaults).

  Notes
    - This class is synchronous by default; `arun_async` supports async streams.
    - For very high QPS, consider pooling or throttling upstream.
  """

  def __init__(
      self,
      *,
      model_id: str = "gemini-2.5-flash",
      use_schema_constraints: bool = True,
      max_char_buffer: int = 800,
      temperature: float | None = None,
      model_url: str | None = None,
      language_model_params: Optional[dict[str, Any]] = None,
  ) -> None:
    self._model_id = model_id
    self._use_schema = use_schema_constraints
    self._max_char_buffer = max_char_buffer
    self._temperature = temperature
    self._model_url = model_url
    self._lm_params = dict(language_model_params or {})
    if temperature is not None:
      self._lm_params.setdefault("temperature", temperature)
    if model_url:
      self._lm_params.setdefault("model_url", model_url)
      self._lm_params.setdefault("base_url", model_url)

    self._buffer: list[str] = []
    self._transcript: list[str] = []  # full conversation so far
    self._merged: list[lx_data.Extraction] = []

  def _extract_now(self, text: str) -> lx_data.AnnotatedDocument:
    return lx.extract(
        text_or_documents=text,
        prompt_description=DEFAULT_PROMPT,
        examples=DEFAULT_EXAMPLES,
        model_id=self._model_id,
        use_schema_constraints=self._use_schema,
        max_char_buffer=self._max_char_buffer,
        language_model_params=self._lm_params,
        show_progress=False,
    )

  @staticmethod
  def _merge(first: list[lx_data.Extraction], new: list[lx_data.Extraction]) -> list[lx_data.Extraction]:
    # Simple first-wins merge using char intervals (similar to internal helper)
    def overlaps(a: lx_data.Extraction, b: lx_data.Extraction) -> bool:
      if not a.char_interval or not b.char_interval:
        return False
      s1, e1 = a.char_interval.start_pos, a.char_interval.end_pos
      s2, e2 = b.char_interval.start_pos, b.char_interval.end_pos
      if s1 is None or e1 is None or s2 is None or e2 is None:
        return False
      return s1 < e2 and s2 < e1

    # Unique by class/text/interval; first occurrence wins
    def key(e: lx_data.Extraction):
      ci = e.char_interval
      s = ci.start_pos if ci else None
      t = ci.end_pos if ci else None
      return (e.extraction_class, e.extraction_text, s, t)

    merged = list(first)
    existing = {key(e) for e in merged}
    for ex in new:
      k = key(ex)
      if k in existing:
        continue
      if any(overlaps(ex, m) for m in merged):
        continue
      merged.append(ex)
      existing.add(k)
    return merged

  def push(self, chunk: str) -> Iterator[list[lx_data.Extraction]]:
    """Push a chunk synchronously and yield new extractions when threshold met.

    Yields
      A list of extractions added since last yield (empty lists may be skipped).
    """
    if not chunk:
      return iter(())
    self._buffer.append(chunk)
    self._transcript.append(chunk)
    if sum(len(c) for c in self._buffer) < self._max_char_buffer // 2:
      return iter(())

    # Aggregate extraction over entire conversation so far
    full_text = "\n".join(self._transcript)
    doc = self._extract_now(full_text)
    new_list = doc.extractions or []

    # Compute delta vs prior merged (by unique key)
    def k(e: lx_data.Extraction):
      ci = e.char_interval
      s = ci.start_pos if ci else None
      t = ci.end_pos if ci else None
      return (e.extraction_class, e.extraction_text, s, t)

    prev_keys = {k(e) for e in self._merged}
    delta = [e for e in new_list if k(e) not in prev_keys]
    self._merged = self._merge(self._merged, new_list)
    self._buffer.clear()
    return iter([delta] if delta else [])

  def flush(self) -> list[lx_data.Extraction]:
    """Force a call and return any new extractions since last emission."""
    if not self._buffer:
      return []
    full_text = "\n".join(self._transcript)
    doc = self._extract_now(full_text)
    new_list = doc.extractions or []
    def k(e: lx_data.Extraction):
      ci = e.char_interval
      s = ci.start_pos if ci else None
      t = ci.end_pos if ci else None
      return (e.extraction_class, e.extraction_text, s, t)
    prev_keys = {k(e) for e in self._merged}
    delta = [e for e in new_list if k(e) not in prev_keys]
    self._merged = self._merge(self._merged, new_list)
    self._buffer.clear()
    return delta

  async def arun_async(self, aiter: AsyncIterator[str]) -> AsyncIterator[list[lx_data.Extraction]]:
    """Process an async stream, yielding incremental new extractions."""
    async for chunk in aiter:
      # Reuse sync logic on a worker thread to avoid blocking
      delta_lists = await asyncio.get_event_loop().run_in_executor(None, lambda: list(self.push(chunk)))
      for delta in delta_lists:
        if delta:
          yield delta
    # Flush at the end
    final_delta = await asyncio.get_event_loop().run_in_executor(None, self.flush)
    if final_delta:
      yield final_delta
