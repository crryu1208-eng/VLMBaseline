"""Find objects from a free-text description: SigLIP2 text-image similarity + words of the VLM descriptions.

The object features in the export are SigLIP2-B/16 image embeddings (mean over views); the same model's text
tower embeds the query (Korean works too). VLM descriptions (이름, 용도, 설명, 상태, 색, 재질) add a
character-bigram match, so '앉을 곳' also finds objects whose uses say '앉기'.
"""
from __future__ import annotations

import re

import numpy as np

_MODEL = "hf-hub:timm/ViT-B-16-SigLIP2"


class TextEncoder:
    def __init__(self, device="cpu"):
        import open_clip
        import torch
        self.torch = torch
        self.model, _, _ = open_clip.create_model_and_transforms(_MODEL, device=device)
        self.model.eval()
        self.tok = open_clip.get_tokenizer(_MODEL)
        self.device = device

    def __call__(self, texts):
        with self.torch.no_grad():
            t = self.model.encode_text(self.tok(texts).to(self.device)).float()
        t = t / t.norm(dim=-1, keepdim=True)
        return t.cpu().numpy()


def _bigrams(s):
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)} | set(s) if s else set()


def vlm_text(o):
    v = o.vlm or {}
    uses = v.get("용도") or []
    return " ".join([str(v.get("이름", "")), " ".join(map(str, uses)) if isinstance(uses, list) else str(uses),
                     str(v.get("설명", "")), str(v.get("상태", "")), str(v.get("색", "")), str(v.get("재질", ""))])


def rank(objects, query, encoder=None, k=5):
    """Objects best matching the query: [(score, object, parts)], best first."""
    n = len(objects)
    sig = np.zeros(n)
    if encoder is not None:
        feats = [o.feature for o in objects]
        ok = np.array([f is not None for f in feats])
        if ok.any():
            F = np.stack([f / np.linalg.norm(f) if f is not None else np.zeros(768) for f in feats])
            t = encoder([query, f"a photo of {query}"]).mean(0)
            s = F @ (t / np.linalg.norm(t))
            mu, sd = s[ok].mean(), s[ok].std() + 1e-6
            sig = np.where(ok, (s - mu) / sd, 0.0)
    q = _bigrams(query)
    kw = np.array([len(q & _bigrams(vlm_text(o))) / max(len(q), 1) for o in objects]) if q else np.zeros(n)
    lab = np.array([1.0 if o.label in query.lower() else 0.0 for o in objects])
    score = sig + 3.0 * kw + 2.0 * lab
    order = np.argsort(-score)[:k]
    return [(float(score[i]), objects[i], {"image_text": round(float(sig[i]), 2), "description": round(float(kw[i]), 2)})
            for i in order]
