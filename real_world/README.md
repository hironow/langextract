Real-World Integration Samples (FastAPI + Streaming)

This folder shows three concrete integration patterns for using LangExtract in real apps:

- REST API: Accepts chat-style `messages` and returns extractions for the whole conversation.
- WebSocket: Streams extractions per incoming message (stream-like updates to the client).
- Library (internal stream): A reusable processor that accepts a text/message stream and yields incremental results, while internally using Gemini (google-genai) via LangExtract.

All examples share a small configuration helper in `real_world/common/extraction_config.py` that defines a prompt and few-shot examples. Adjust it to your domain.

Prerequisites
- Python 3.10+
- Install dependencies:
  - FastAPI and Uvicorn for servers: `pip install fastapi uvicorn`
  - A provider backend:
    - Gemini: `pip install google-genai` and set `GEMINI_API_KEY` or `LANGEXTRACT_API_KEY`
    - OpenAI: `pip install openai` and set `OPENAI_API_KEY`
    - Ollama: install Ollama locally and run `ollama serve`; for remote/proxy, set `OLLAMA_BASE_URL` or pass `model_url`

Provider Selection
- The examples accept a `model_id` in requests. LangExtract auto-selects a provider by pattern:
  - Gemini (e.g., `gemini-2.5-flash`)
  - OpenAI (e.g., `gpt-4o-mini`)
  - Ollama (e.g., `gemma2:2b`, `llama3`) — for local inference
  - HF-style IDs are also supported (see repo tests for patterns)

Security Notes
- Do not log raw API keys. The examples read keys from environment variables and pass them as kwargs.
- When using Ollama with localhost + API Key, LangExtract warns (still sends header). Prefer a remote/proxy when auth is required.

1) REST API (FastAPI)
Run
  - `uvicorn real_world.rest_api.app:app --reload --port 8080`

Request
  - POST `http://localhost:8080/extract`
  - JSON body:
    {
      "model_id": "gemini-2.5-flash",      // or "gpt-4o-mini", "gemma2:2b" etc.
      "messages": [
        {"role": "user", "content": "Patient took 400 mg ibuprofen."},
        {"role": "assistant", "content": "Anything else?"}
      ],
      "use_schema_constraints": true,       // default true for Gemini
      "max_char_buffer": 500
    }

Response (200)
  - JSON with the final AnnotatedDocument structure (document text and extractions).

2) WebSocket Streaming (FastAPI)
Run
  - `uvicorn real_world.websocket_server.app:app --reload --port 8081`

Protocol
  - 単発モード（低レイテンシ）:
    - Connect: `ws://localhost:8081/ws?model_id=gemini-2.5-flash&aggregate=false`
    - Client → Server: `{ "role": "user", "content": "message text" }`（role省略可）
    - Server → Client: `{ "type": "result", "text": "...", "extractions": [...] }`
  - 会話全体に対する集約抽出（aggregate=true）:
    - Connect: `ws://localhost:8081/ws?model_id=gemini-2.5-flash&aggregate=true`
    - サーバは受信フレームを履歴に蓄積し、毎回「履歴全体」を対象に抽出を実行
    - Server → Client: `{ "type": "aggregate_result", "text": "full transcript", "extractions": [...] }`

3) Library: Streaming Processor（会話全体の集約抽出）
Usage
  - See `real_world/lib_stream/processor.py` for `StreamExtractor` — 文字列チャンクを順次 `push()`（同期）または `arun_async()`（非同期）で渡すと、**会話全文**に対して抽出を行い、新規に見つかった抽出だけを増分で返します。
  - 内部ではトランスクリプト（全文）を保持し、しきい値や `flush()` に応じて LangExtract を呼び出し、既存抽出と重複しない新規抽出のみを返却します。

Notes on Formats and Schemas
- The samples default to JSON output. With Gemini + `use_schema_constraints=True`, LangExtract enforces a response schema and does not require code fences.
- For Ollama, we use JSON format with a format handler. If you override to YAML, fences will be required and handled by the resolver.

Extending Examples
- Replace `DEFAULT_PROMPT` and `DEFAULT_EXAMPLES` with your domain rules.
- Tune `max_char_buffer`, `batch_length`, `extraction_passes` based on your latency/recall tradeoffs.
