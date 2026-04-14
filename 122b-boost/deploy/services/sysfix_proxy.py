"""Proxy: merge system messages + fix thinking/tool_call conflict for sglang/Qwen3.5.
When tools are present, disable thinking to prevent <tool_call> in reasoning from
terminating the response prematurely."""
import os, json, logging, sys, httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response
import uvicorn

UPSTREAM = os.environ.get("UPSTREAM", "http://localhost:8000")
app = FastAPI()

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("sysfix")


def _summarize_messages(messages):
    roles, total = {}, 0
    biggest = ("", 0)
    for m in messages:
        role = m.get("role", "?")
        c = m.get("content", "")
        if isinstance(c, list):
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        n = len(c or "")
        total += n
        roles[role] = roles.get(role, 0) + 1
        if n > biggest[1]:
            biggest = (role, n)
        # tool_calls / tool results can also be huge
        tcs = m.get("tool_calls") or []
        for t in tcs:
            a = (t.get("function") or {}).get("arguments", "") or ""
            total += len(a)
    return roles, total, biggest


async def _count_tokens(data):
    """Ask upstream /tokenize for the prompt token count. Returns int or None."""
    payload = {"model": data.get("model")}
    if data.get("messages") is not None:
        payload["messages"] = data["messages"]
        if data.get("tools"):
            payload["tools"] = data["tools"]
        cpk = data.get("chat_template_kwargs")
        if cpk:
            payload["chat_template_kwargs"] = cpk
    elif data.get("prompt") is not None:
        payload["prompt"] = data["prompt"]
    else:
        return None
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.post(f"{UPSTREAM}/tokenize", json=payload)
            if r.status_code == 200:
                return r.json().get("count")
    except Exception as e:
        log.debug("tokenize failed: %s", e)
    return None

def merge_system_messages(messages):
    system_parts = []
    others = []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(m.get("content", ""))
        else:
            others.append(m)
    result = []
    if system_parts:
        result.append({"role": "system", "content": "\n\n".join(system_parts)})
    result.extend(others)
    return result

def fix_tool_thinking_conflict(data):
    """When tools are present, disable thinking to avoid <tool_call> in reasoning."""
    if data.get("tools") or data.get("functions"):
        if "chat_template_kwargs" not in data:
            data["chat_template_kwargs"] = {}
        data["chat_template_kwargs"]["enable_thinking"] = False
    return data

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"])
async def proxy(path: str, request: Request):
    url = f"{UPSTREAM}/{path}"
    if request.query_params:
        url += f"?{request.query_params}"

    if request.method in ("GET", "HEAD", "OPTIONS"):
        async with httpx.AsyncClient(timeout=30) as c:
            resp = await c.request(request.method, url)
            return Response(content=resp.content, status_code=resp.status_code,
                          headers=dict(resp.headers), media_type=resp.headers.get("content-type"))

    body = await request.body()
    msgs_summary = None
    req_max_tokens = None
    data = None
    try:
        data = json.loads(body)
        if "messages" in data:
            msgs_summary = _summarize_messages(data["messages"])
            data["messages"] = merge_system_messages(data["messages"])
        req_max_tokens = data.get("max_tokens")
        data = fix_tool_thinking_conflict(data)
        body = json.dumps(data).encode()
    except Exception:
        pass

    if path.endswith("chat/completions") and msgs_summary is not None:
        roles, total, (big_role, big_n) = msgs_summary
        tok_count = await _count_tokens(data) if data is not None else None
        log.info("REQ path=%s roles=%s chars=%d biggest=%s(%d) prompt_tokens=%s max_tokens=%s body_bytes=%d",
                 path, roles, total, big_role, big_n,
                 tok_count if tok_count is not None else "?",
                 req_max_tokens, len(body))

    is_stream = False
    try:
        is_stream = json.loads(body).get("stream", False)
    except:
        pass

    if is_stream:
        async def stream():
            async with httpx.AsyncClient(timeout=600) as c:
                async with c.stream("POST", url, content=body,
                                   headers={"Content-Type": "application/json"}) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        return StreamingResponse(stream(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient(timeout=600) as c:
            resp = await c.post(url, content=body, headers={"Content-Type": "application/json"})
            if resp.status_code >= 400:
                log.warning("UPSTREAM %d on %s: %s", resp.status_code, path,
                            resp.text[:500])
            return Response(content=resp.content, status_code=resp.status_code,
                          media_type=resp.headers.get("content-type"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
