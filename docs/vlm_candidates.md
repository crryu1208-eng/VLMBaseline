# Local VLM candidates (checked 2026-11-19)

License = `license` field of the Hugging Face model card (huggingface.co/api/models/<id>), checked on 2026-11-19.
Params = safetensors total. GPU: RTX 3080 Ti Laptop, 16 GB (bf16 needs about 2 GB per 1B params, 4-bit about 0.6 GB).

| Model | Weights license | Params | Training data | 16 GB fit | Verdict |
|---|---|---|---|---|---|
| Qwen/Qwen3.5-4B | Apache-2.0 | 4.7B | not disclosed | bf16 | shortlist |
| Qwen/Qwen3.5-9B | Apache-2.0 | 9.7B | not disclosed | 8-bit / 4-bit | shortlist (main) |
| Qwen/Qwen3.5-27B, 35B-A3B; Qwen3.6-27B, 35B-A3B; Qwen3.8-27B | Apache-2.0 | 28-36B | not disclosed | 4-bit, too tight with images | later / server |
| Qwen/Qwen3.8-Flash-Next | other (custom) | 180B | - | no | exclude |
| Qwen/Qwen3-VL-4B / 8B-Instruct | Apache-2.0 | 4.4 / 8.8B | not disclosed | bf16 / 8-bit | fallback (older) |
| google/gemma-4-E4B-it | Apache-2.0 | 8.0B (E4B) | not disclosed (web, code, images, audio; cutoff 2025-01) | bf16 | shortlist |
| google/gemma-4-12B-it | Apache-2.0 | 12.0B | not disclosed | 8-bit / 4-bit (QAT q4_0 published) | shortlist |
| google/gemma-4-26B-A4B-it, 31B-it | Apache-2.0 | 26 / 31B | not disclosed | 4-bit, tight | later / server |
| google/gemma-3-* | Gemma Terms of Use (gated, prohibited-use policy) | - | - | - | exclude |
| allenai/Molmo2-8B | Apache-2.0 (base Qwen3-8B + SigLIP) | 8.7B | open (PixMo / Molmo2 data, released) | 8-bit / 4-bit | shortlist (pointing, open data) |
| allenai/Molmo2-O-7B | Apache-2.0 (base OLMo-3-7B + SigLIP) | 7.8B | open (fully open stack) | 8-bit | alternative to Molmo2-8B |
| OpenGVLab/InternVL3_5-8B | Apache-2.0 (card) | 8.5B | partly described | 8-bit | not shortlisted |
| microsoft/Phi-4-multimodal-instruct | MIT | 5.6B | partly described | bf16 | not shortlisted (older) |
| openbmb/MiniCPM-V-4_5 | Apache-2.0 (registration optional) | 8.7B | not disclosed | 8-bit | not shortlisted |
| mistralai/Mistral-Small-3.2-24B | Apache-2.0 | 24B | not disclosed | 4-bit, tight | not shortlisted |
| zai-org/GLM-4.1V-9B-Thinking | MIT | 10.3B | not disclosed | 8-bit | not shortlisted |
| zai-org/GLM-4.5V | MIT | 108B | - | no | exclude (size) |
| moonshotai/Kimi-VL-A3B | MIT | 16.4B | not disclosed | 8-bit | not shortlisted |
| HuggingFaceTB/SmolVLM2-2.2B | Apache-2.0 | 2.2B | open | bf16 | fast captioner only |
| meta-llama/Llama-3.2-11B-Vision | Llama 3.2 Community (gated; vision rights excluded for EU-based users, AUP) | 10.7B | - | - | exclude |

Notes
- "Apache-2.0 / MIT" covers code and weights. Except Molmo2 and SmolVLM2, the training data is not disclosed, so its
  licence cannot be checked; this is the remaining risk for every closed-data model (shared by all of them).
- Inherited licences: Molmo2-8B is fine-tuned from Qwen3-8B (Apache-2.0) and SigLIP (Apache-2.0).
- Before deployment, re-read the LICENSE file of the exact checkpoint (cards can change) and keep a copy.

Next: benchmark the shortlist on our exports (object attributes from keyframe crops, Korean QA, latency, VRAM).
