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
- 물체 기호는 O817 같은 형식입니다. 답에 해당 물체의 기호를 함께 쓰세요.
- 길 안내 질문은 guide_to(목표가 라벨일 때) 또는 route(목표가 기호일 때)를 호출해, 걷는 거리와 지나가는 랜드마크 순서로 안내하세요.
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

    def _gen(self, msgs, max_new_tokens=384):
        inputs = self.proc.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt", enable_thinking=False)
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      eos_token_id=self.eos)
        return self.proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def ask(self, question):
        msgs = [{"role": "system", "content": [{"type": "text", "text": self.system}]},
                {"role": "user", "content": [{"type": "text", "text": question}]}]
        trace = []
        for _ in range(self.max_steps):
            text = self._gen(msgs)
            calls = parse_calls(text)
            if not calls:
                return text, trace
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            for name, args in calls:
                res = self.idx.call(name, args)
                trace.append({"tool": name, "args": args, "result": res})
                msgs.append({"role": "tool", "content": [{"type": "text", "text": json.dumps(res, ensure_ascii=False)}]})
        return self._gen(msgs + [{"role": "user", "content": [{"type": "text", "text": "도구 결과로 답만 하세요."}]}]), trace
