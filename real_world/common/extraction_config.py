"""Shared prompt + examples used by the integration samples.

Adjust these for your own domain. The examples are intentionally small for demo.
"""
from __future__ import annotations

import langextract as lx


# A concise extraction instruction. Keep it short for demo; expand for production.
DEFAULT_PROMPT = (
    "Extract medications, dosages, frequencies, durations, and conditions in"
    " the order they appear. Use exact text spans."
)


# Minimal few-shot examples to bootstrap the model behavior.
DEFAULT_EXAMPLES: list[lx.data.ExampleData] = [
    lx.data.ExampleData(
        text="Patient was given 250 mg IV Cefazolin TID for one week.",
        extractions=[
            lx.data.Extraction(extraction_class="dosage", extraction_text="250 mg"),
            lx.data.Extraction(extraction_class="route", extraction_text="IV"),
            lx.data.Extraction(
                extraction_class="medication", extraction_text="Cefazolin"
            ),
            lx.data.Extraction(
                extraction_class="frequency", extraction_text="TID"
            ),
            lx.data.Extraction(
                extraction_class="duration", extraction_text="one week"
            ),
        ],
    )
]


def flatten_chat_messages(messages: list[dict[str, str]]) -> str:
  """Convert chat-style messages to a single text blob with roles.

  Example input:
    [{"role": "user", "content": "hello"}, {"role": "assistant", ...}]

  Returns:
    A string like: "user: hello\nassistant: ...\nuser: ..."
  """
  lines: list[str] = []
  for m in messages or []:
    role = m.get("role") or "user"
    content = (m.get("content") or "").strip()
    if not content:
      continue
    lines.append(f"{role}: {content}")
  return "\n".join(lines)

