"""Anthropic /v1/messages -> OpenAI /v1/chat/completions translator.

Minimal translator so Claude Code can talk to a vLLM OpenAI-compatible server.
Supports: text, tool_use, tool_result (as text), streaming + non-streaming.
Does NOT support: images, prompt caching, computer-use, documents.

Env:
  UPSTREAM        OpenAI base URL, e.g. http://vllm:30000/v1
  MODEL_OVERRIDE  if set, forces this model id on upstream regardless of
                  what the Anthropic client asks for (useful to map any
                  claude-* model name to qwen3.5-122b-int4).
"""
import os, json, uuid, time, logging, sys
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

UPSTREAM = os.environ.get("UPSTREAM", "http://vllm:30000/v1").rstrip("/")
MODEL_OVERRIDE = os.environ.get("MODEL_OVERRIDE", "")

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cc-proxy")

app = FastAPI()


# ---------- request: anthropic -> openai ----------
def _flatten_content(content):
    """Anthropic content can be str or list of blocks. Return (text, tool_calls, tool_results)."""
    if isinstance(content, str):
        return content, [], []
    text_parts = []
    tool_calls = []
    tool_results = []
    for b in content or []:
        t = b.get("type")
        if t == "text":
            text_parts.append(b.get("text", ""))
        elif t == "tool_use":
            tool_calls.append({
                "id": b.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": b.get("name", ""),
                    "arguments": json.dumps(b.get("input", {}), ensure_ascii=False),
                },
            })
        elif t == "tool_result":
            c = b.get("content", "")
            if isinstance(c, list):
                c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
            tool_results.append({
                "tool_call_id": b.get("tool_use_id", ""),
                "content": c if isinstance(c, str) else json.dumps(c, ensure_ascii=False),
                "is_error": bool(b.get("is_error")),
            })
        # images/other blocks silently dropped
    return "\n".join(text_parts), tool_calls, tool_results


def anthropic_to_openai(req):
    out_messages = []

    # system can be str or list[{type:text,text:...}]
    sys_ = req.get("system")
    if isinstance(sys_, list):
        sys_ = "\n\n".join(b.get("text", "") for b in sys_ if b.get("type") == "text")
    if sys_:
        out_messages.append({"role": "system", "content": sys_})

    for m in req.get("messages") or []:
        role = m.get("role")
        text, tool_calls, tool_results = _flatten_content(m.get("content"))

        if role == "user":
            # tool_result blocks become separate tool-role messages
            for tr in tool_results:
                out_messages.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_call_id"],
                    "content": tr["content"],
                })
            if text:
                out_messages.append({"role": "user", "content": text})
        elif role == "assistant":
            msg = {"role": "assistant", "content": text or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out_messages.append(msg)

    out = {
        "model": MODEL_OVERRIDE or req.get("model"),
        "messages": out_messages,
        "stream": bool(req.get("stream")),
    }
    if "max_tokens" in req:
        out["max_tokens"] = req["max_tokens"]
    if "temperature" in req:
        out["temperature"] = req["temperature"]
    if "top_p" in req:
        out["top_p"] = req["top_p"]
    if "stop_sequences" in req:
        out["stop"] = req["stop_sequences"]

    tools = req.get("tools") or []
    if tools:
        out["tools"] = [{
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        } for t in tools]
        tc = req.get("tool_choice") or {}
        if isinstance(tc, dict):
            if tc.get("type") == "auto":
                out["tool_choice"] = "auto"
            elif tc.get("type") == "any":
                out["tool_choice"] = "required"
            elif tc.get("type") == "tool":
                out["tool_choice"] = {"type": "function",
                                       "function": {"name": tc.get("name", "")}}

    return out


# ---------- response: openai -> anthropic (non-stream) ----------
def openai_to_anthropic(resp, model):
    choice = resp["choices"][0]
    msg = choice.get("message") or {}
    blocks = []

    text = msg.get("content") or ""
    if text:
        blocks.append({"type": "text", "text": text})

    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            args = {"_raw": fn.get("arguments")}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
            "name": fn.get("name", ""),
            "input": args,
        })

    fr = choice.get("finish_reason")
    stop_reason = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "stop_sequence",
    }.get(fr, "end_turn")

    usage = resp.get("usage") or {}
    return {
        "id": resp.get("id") or f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ---------- streaming translator ----------
async def stream_openai_to_anthropic(upstream_body, model):
    """Consume OpenAI SSE, emit Anthropic SSE events."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    def sse(event, data):
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()

    yield sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    # Track block state: current index, whether it's text or tool_use, tool call id/name
    block_idx = -1
    current_block = None  # "text" | "tool_use"
    tool_state = {}  # idx -> {id, name, args_buf}
    stop_reason = "end_turn"
    usage_out = {"input_tokens": 0, "output_tokens": 0}

    def open_text_block():
        nonlocal block_idx, current_block
        block_idx += 1
        current_block = "text"
        return sse("content_block_start", {
            "type": "content_block_start",
            "index": block_idx,
            "content_block": {"type": "text", "text": ""},
        })

    def close_block():
        nonlocal current_block
        if current_block is None:
            return b""
        current_block = None
        return sse("content_block_stop", {"type": "content_block_stop", "index": block_idx})

    ping_sent = False
    try:
        async for raw in upstream_body:
            for line in raw.splitlines():
                line = line.strip()
                if not line or not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                try:
                    chunk = json.loads(payload)
                except Exception:
                    continue

                if not ping_sent:
                    yield sse("ping", {"type": "ping"})
                    ping_sent = True

                choices = chunk.get("choices") or []
                if not choices:
                    u = chunk.get("usage") or {}
                    if u:
                        usage_out = {
                            "input_tokens": u.get("prompt_tokens", 0),
                            "output_tokens": u.get("completion_tokens", 0),
                        }
                    continue
                delta = choices[0].get("delta") or {}

                # Text content
                if delta.get("content"):
                    if current_block != "text":
                        yield close_block()
                        yield open_text_block()
                    yield sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {"type": "text_delta", "text": delta["content"]},
                    })

                # Reasoning (Qwen3 via vLLM exposes as `reasoning` or `reasoning_content`)
                rtxt = delta.get("reasoning") or delta.get("reasoning_content")
                if rtxt:
                    # Emit as thinking block so CC can show it
                    if current_block != "thinking":
                        yield close_block()
                        block_idx += 1
                        current_block = "thinking"
                        yield sse("content_block_start", {
                            "type": "content_block_start",
                            "index": block_idx,
                            "content_block": {"type": "thinking", "thinking": ""},
                        })
                    yield sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_idx,
                        "delta": {"type": "thinking_delta", "thinking": rtxt},
                    })

                # Tool calls (may arrive split across chunks)
                for tc in delta.get("tool_calls") or []:
                    oa_idx = tc.get("index", 0)
                    st = tool_state.get(oa_idx)
                    if st is None:
                        # new tool_use block
                        yield close_block()
                        block_idx += 1
                        current_block = "tool_use"
                        st = {"id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                              "name": (tc.get("function") or {}).get("name", ""),
                              "anthropic_idx": block_idx,
                              "args_buf": ""}
                        tool_state[oa_idx] = st
                        yield sse("content_block_start", {
                            "type": "content_block_start",
                            "index": block_idx,
                            "content_block": {
                                "type": "tool_use", "id": st["id"],
                                "name": st["name"], "input": {},
                            },
                        })
                    fn = tc.get("function") or {}
                    if fn.get("name") and not st["name"]:
                        st["name"] = fn["name"]
                    args_piece = fn.get("arguments") or ""
                    if args_piece:
                        st["args_buf"] += args_piece
                        yield sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": st["anthropic_idx"],
                            "delta": {"type": "input_json_delta",
                                      "partial_json": args_piece},
                        })

                fr = choices[0].get("finish_reason")
                if fr:
                    stop_reason = {
                        "stop": "end_turn",
                        "length": "max_tokens",
                        "tool_calls": "tool_use",
                        "content_filter": "stop_sequence",
                    }.get(fr, "end_turn")
    finally:
        yield close_block()
        yield sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": usage_out,
        })
        yield sse("message_stop", {"type": "message_stop"})


# ---------- endpoints ----------
@app.post("/v1/messages")
async def messages(req: Request):
    body = await req.json()
    stream = bool(body.get("stream"))
    oa_body = anthropic_to_openai(body)
    model = body.get("model") or oa_body["model"]
    log.info("CC messages model=%s stream=%s msgs=%d tools=%d",
             oa_body["model"], stream, len(oa_body["messages"]),
             len(oa_body.get("tools") or []))

    url = f"{UPSTREAM}/chat/completions"
    if stream:
        async def gen():
            async with httpx.AsyncClient(timeout=600) as c:
                async with c.stream("POST", url, json=oa_body) as resp:
                    if resp.status_code >= 400:
                        text = await resp.aread()
                        log.warning("upstream %d: %s", resp.status_code, text[:500])
                        yield f"event: error\ndata: {json.dumps({'type':'error','error':{'type':'api_error','message':text.decode('utf-8','replace')[:500]}})}\n\n".encode()
                        return
                    async for chunk in stream_openai_to_anthropic(resp.aiter_bytes(), model):
                        yield chunk
        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=600) as c:
        r = await c.post(url, json=oa_body)
    if r.status_code >= 400:
        log.warning("upstream %d: %s", r.status_code, r.text[:500])
        return JSONResponse({"type": "error",
                             "error": {"type": "api_error", "message": r.text[:500]}},
                            status_code=r.status_code)
    return JSONResponse(openai_to_anthropic(r.json(), model))


@app.post("/v1/messages/count_tokens")
async def count_tokens(req: Request):
    body = await req.json()
    oa = anthropic_to_openai(body)
    payload = {"model": oa["model"], "messages": oa["messages"]}
    if oa.get("tools"):
        payload["tools"] = oa["tools"]
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(f"{UPSTREAM.rsplit('/v1', 1)[0]}/tokenize", json=payload)
            return JSONResponse({"input_tokens": r.json().get("count", 0)})
    except Exception as e:
        log.warning("count_tokens failed: %s", e)
        return JSONResponse({"input_tokens": 0})


@app.get("/health")
def health():
    return {"ok": True, "upstream": UPSTREAM}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8082")))
