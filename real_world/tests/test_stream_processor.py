from typing import Any, List

import pytest

import langextract as lx
from langextract.core import data as lx_data
from real_world.lib_stream.processor import StreamExtractor


def _mk_doc(full_text: str) -> lx_data.AnnotatedDocument:
  """Create extractions for the first occurrence of A/D/G in the text.

  This lets us simulate incremental discoveries as the transcript grows.
  """
  exts: List[lx_data.Extraction] = []
  for token in ["A", "D", "G"]:
    idx = full_text.find(token)
    if idx >= 0:
      exts.append(
          lx_data.Extraction(
              extraction_class="letter",
              extraction_text=token,
              char_interval=lx_data.CharInterval(start_pos=idx, end_pos=idx + 1),
          )
      )
  return lx_data.AnnotatedDocument(text=full_text, extractions=exts)


@pytest.mark.parametrize(
    "chunks,expected_deltas",
    [
        # Given 'ABC' then 'DEF' then flush => When threshold hit and flush => Then deltas [A,D], then [G]
        ( ["ABC", "DEF", "GHI"], [["A", "D"], ["G"]] ),
        # Given 'A' then 'A' again => When aggregate extraction reruns => Then only the new position appears once
        ( ["A", "A"], [["A"], []] ),
    ],
)
def test_stream_extractor_given_chunks_when_pushed_then_yields_incremental_deltas(
    monkeypatch: pytest.MonkeyPatch, chunks, expected_deltas
):
  """Given streaming chunks, When pushing and flushing, Then only new extractions are yielded.

  Given
    - A StreamExtractor with small buffer so pushes trigger extraction
  When
    - We push chunks and finally flush
  Then
    - The deltas contain only new (class,text,interval) not seen before
  """

  def fake_extract(**kwargs):  # type: ignore[no-redef]
    full_text = kwargs["text_or_documents"]
    return _mk_doc(full_text)

  monkeypatch.setattr(lx, "extract", lambda **kw: fake_extract(**kw))

  proc = StreamExtractor(model_id="gemini-2.5-flash", max_char_buffer=10)

  all_deltas: list[list[str]] = []
  # Push chunks; threshold is half of max_char_buffer (5)
  for c in chunks:
    for delta in proc.push(c):
      if delta:
        all_deltas.append([e.extraction_text for e in delta])

  # Ensure a flush to emit remaining
  flushed = proc.flush()
  if flushed:
    all_deltas.append([e.extraction_text for e in flushed])

  assert all_deltas == expected_deltas

