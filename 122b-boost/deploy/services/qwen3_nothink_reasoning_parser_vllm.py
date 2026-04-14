"""Custom vLLM reasoning parser plugin for Qwen3.

Behavior: if `<tool_call>` appears inside the `<think>` block (i.e. the model
jumps straight to a tool call without emitting `</think>`), treat it as the
end of reasoning. Text from `<tool_call>` onward is passed to the tool-call
parser (qwen3_coder) as normal content.

Activate with:
    --reasoning-parser-plugin /path/to/qwen3_nothink_reasoning_parser_vllm.py
    --reasoning-parser qwen3_nothink
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


TOOL_CALL_TOKEN = "<tool_call>"


@ReasoningParserManager.register_module("qwen3_nothink")
class Qwen3NoThinkReasoningParser(Qwen3ReasoningParser):
    """Qwen3 reasoning parser that treats `<tool_call>` inside thinking as end-of-reasoning."""

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        parts = model_output.partition(self.start_token)
        body = parts[2] if parts[1] else parts[0]

        end_idx = body.find(self.end_token)
        tool_idx = body.find(TOOL_CALL_TOKEN)

        if tool_idx != -1 and (end_idx == -1 or tool_idx < end_idx):
            reasoning = body[:tool_idx].rstrip()
            content = body[tool_idx:]
            # Drop any stray </think> still embedded in the content.
            content = content.replace(self.end_token, "")
            return (reasoning or None), (content or None)

        return super().extract_reasoning(model_output, request)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        # Once reasoning has ended (either via </think> or via a previous
        # <tool_call>), delegate to the base implementation.
        if self.end_token_id in previous_token_ids:
            return super().extract_reasoning_streaming(
                previous_text, current_text, delta_text,
                previous_token_ids, current_token_ids, delta_token_ids,
            )

        # We're still in the reasoning phase. Check whether <tool_call>
        # has just appeared — either fully inside delta_text, or spanning
        # the boundary between previous_text and delta_text.
        combined = previous_text + delta_text
        tool_idx = combined.find(TOOL_CALL_TOKEN)
        if tool_idx == -1:
            # Still thinking; also allow </think> handling by parent.
            if self.end_token_id in delta_token_ids:
                return super().extract_reasoning_streaming(
                    previous_text, current_text, delta_text,
                    previous_token_ids, current_token_ids, delta_token_ids,
                )
            # Guard against emitting a partial <tool_call> prefix as reasoning.
            for i in range(1, len(TOOL_CALL_TOKEN)):
                if delta_text.endswith(TOOL_CALL_TOKEN[:i]) and \
                   not delta_text.endswith(TOOL_CALL_TOKEN):
                    safe = delta_text[:-i]
                    return DeltaMessage(reasoning=safe) if safe else None
            return DeltaMessage(reasoning=delta_text) if delta_text else None

        # <tool_call> boundary is at absolute index `tool_idx` in `combined`.
        prev_len = len(previous_text)
        if tool_idx >= prev_len:
            # The token starts within this delta.
            split = tool_idx - prev_len
            reasoning_part = delta_text[:split]
            content_part = delta_text[split:]
        else:
            # The token already started in previous_text — emit the remaining
            # delta entirely as content.
            reasoning_part = ""
            content_part = delta_text

        content_part = content_part.replace(self.end_token, "")
        if not reasoning_part and not content_part:
            return None
        return DeltaMessage(
            reasoning=reasoning_part if reasoning_part else None,
            content=content_part if content_part else None,
        )
