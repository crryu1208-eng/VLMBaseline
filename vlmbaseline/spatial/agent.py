"""Question answering / guidance agent: the VLM calls the spatial tools, code computes the facts.

    agent = Agent(proc, model, SpatialIndex(scene))
    answer, trace = agent.ask("책상 O817에서 가장 가까운 화분까지 가는 길 알려줘")
"""
from __future__ import annotations

import json
import re

import torch

from vlmbaseline.spatial.tools import TOOLS, SpatialIndex

SYSTEM = """당신은 실내 공간 안내 도우미입니다. 이 공간의 3D 씬그래프를 도구로 조회할 수 있습니다.
- 물체의 개수, 위치, 거리, 경로, 크기 같은 공간 정보는 추측하지 말고 반드시 도구로 확인하세요.
- 도구의 label 인자는 아래 영어 라벨 중에서 고르세요: {labels}
- 물체 기호는 O와 숫자로 된 형식이며, 도구 결과에 나온 기호만 쓰세요. 답에 해당 물체의 기호를 함께 쓰세요.
- 길 안내 질문은 guide_to(목표가 라벨일 때) 또는 route(목표가 기호일 때)를 호출해, 걷는 거리와 지나가는 랜드마크 순서로 안내하세요.
- 물체를 기호 없이 말로 설명하면(예: '창가의 큰 화분', '앉을 곳') search로 후보를 찾고, 설명이 맞는 물체를 고르세요. 후보가 맞지 않으면 관련된 영어 라벨로 find를 한 번 더 해 보세요(예: 차가운 음료 -> refrigerator, small refrigerator).
- 할 일·작업 추천 질문은 inventory를 호출해 actionable 목록의 물체와 action만 근거로 3가지 이내로 추천하세요(기호 포함). 목록이 비어 있으면 '지금 촬영 기준으로 손볼 곳이 보이지 않습니다'라고 답하세요.
- 씬그래프에 없는 정보(콘센트, 와이파이, 온도, 운영 시간, 지금 있는 사람 수, 물건의 내부 상태 등)는 추측하지 말고
  '씬그래프로는 확인할 수 없습니다'라고 답하세요. 물체의 상태는 촬영 당시 기준입니다.
- 도구 결과의 영어 필드 이름(needs_check 등)이나 null을 답에 그대로 쓰지 마세요.
- 거리는 m 단위입니다. 도구 결과만 근거로 한국어로 짧게 답하세요."""

CALL = re.compile(r"<tool_call>\s*<function=([\w_]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM = re.compile(r"<parameter=([\w_]+)>\s*(.*?)\s*</parameter>", re.S)


def parse_calls(text):
    calls = []
    for name, body in CALL.findall(text):
        args = {}
        for k, v in PARAM.findall(body):
            v = v.strip()
            args[k] = int(v) if re.fullmatch(r"-?\d+", v) and k == "k" else v
        calls.append((name, args))
    return calls


class Agent:
    def __init__(self, proc, model, index: SpatialIndex, max_steps=5):
        self.proc, self.model, self.idx, self.max_steps = proc, model, index, max_steps
        self.system = SYSTEM.format(labels=", ".join(sorted(index.labels)))
        tok = proc.tokenizer
        ends = [t for t in ("<|im_end|>", "<|endoftext|>", "<end_of_turn>", "<eos>") if t in tok.get_vocab()]
        self.eos = [tok.convert_tokens_to_ids(t) for t in ends] or None

    def _gen(self, msgs, max_new_tokens=384, prefix=""):
        """Generate the next assistant turn; `prefix` is forced as its start (e.g. '<tool_call>')."""
        text = self.proc.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, tokenize=False,
                                             enable_thinking=False) + prefix
        inputs = self.proc.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      eos_token_id=self.eos)
        return prefix + self.proc.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def ask(self, question):
        msgs = [{"role": "system", "content": [{"type": "text", "text": self.system}]},
                {"role": "user", "content": [{"type": "text", "text": question}]}]
        trace, nudged, checked = [], False, False
        for _ in range(self.max_steps):
            text = self._gen(msgs)
            calls = parse_calls(text)
            if not calls and not trace:  # about to answer without looking: make this turn a tool call
                text = self._gen(msgs, prefix="<tool_call>\n<function=")
                calls = parse_calls(text)
            if not calls and not nudged and re.search(r"(보겠습니다|확인하겠습니다|찾아보겠습니다)\W*$", text):
                nudged = True  # promised to look and stopped: ask once to go on
                msgs += [{"role": "assistant", "content": [{"type": "text", "text": text}]},
                         {"role": "user", "content": [{"type": "text", "text": "도구를 호출해서 계속 확인하세요."}]}]
                continue
            if not calls:
                unknown = sorted({x for x in re.findall(r"O\d+", text) if x not in self.idx.obj})
                if unknown and not checked:  # cited objects that do not exist: ask once to answer from the tools
                    checked = True
                    msgs += [{"role": "assistant", "content": [{"type": "text", "text": text}]},
                             {"role": "user", "content": [{"type": "text", "text":
                              f"{', '.join(unknown)}은(는) 이 공간에 없는 기호입니다. 도구로 다시 확인해서 답하세요."}]}]
                    continue
                return text, trace
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            for name, args in calls:
                res = self.idx.call(name, args)
                trace.append({"tool": name, "args": args, "result": res})
                msgs.append({"role": "tool", "content": [{"type": "text", "text": json.dumps(res, ensure_ascii=False)}]})
        return self._gen(msgs + [{"role": "user", "content": [{"type": "text", "text": "도구 결과로 답만 하세요."}]}]), trace
