# Original Strata draft vocabulary subset: A3B portability assessment

This is a source-level assessment only. No A3B subset model, CUDA kernel, or
service run was built or benchmarked. It is a separate possible experiment
from the default-off packed Q8 output-head candidate in `ANALYSIS.md`.

## What original Strata actually does

Original Strata's `--draft-vocab` reads an int32 token-ID list in
`G:/Strata/src/core/mtp.cpp` (`MtpDrafter::bind`). It gathers those rows from
the already loaded native GPU output head into a smaller GPU head, performs
the MTP Q8/Q6 matrix-vector product against that smaller head, and maps the
sampled subset index back to the model's full token ID. Its coupled sampling
path also uses inverse ID mapping when applying token penalties. The original
`G:/Strata/tools/draft_vocab.py` derives the ID list from the GGUF tokenizer.
Original `--draft-vocab en`, `cyrillic`, `fr`, and CJK lists refer to the
Flash-Next tokenizer and cannot be reused as A3B ID lists without checking
every token mapping. The original [DETAILS.md](../../../../Strata/docs/DETAILS.md)
reports language-specific acceptance and speed tradeoffs, not an invariant
gain for every request.

## What the matching A3B MTP GGUF permits

The selected matching `mtp-Qwen3.6-35B-A3B-Q8_0.gguf` has a separate Q8_0
`output.weight` `[2048,248320]` and a separate Q8_0 `token_embd.weight` of
the same shape. Its `blk.40.nextn.shared_head_norm.weight` exists but no
`blk.40.nextn.shared_head_head.weight` exists, so the A3B MTP graph uses
`model.output` for its proposal logits. A Q8_0 output row is exactly
`2048 / 32 * 34 = 2176` bytes. The full head is 540,344,320 bytes, or
515.3125 MiB. Selecting whole Q8_0 rows preserves their quantized bytes:
it is the same downloaded MTP model's output head, with fewer eligible token
IDs, not a new quantization method. The full input token embedding must stay:
verified target tokens fed to the draft can be any of the 248,320 IDs.

Original Flash-Next subset *counts* give an illustrative scale only; A3B
needs a newly generated ID set from its own tokenizer and workload:

The local IQ3_S target and Q8_0 A3B tokenizer directories have byte-identical
`vocab.json`, `merges.txt`, and `token_type.json` SHA-256 hashes. That supports
building one A3B ID list for this target/draft pair, but a subset generator
must still validate the IDs against the matching MTP GGUF tokenizer before
use. It does not make Flash-Next IDs transferable.

| Illustrative selected IDs | Q8_0 output bytes | MiB | Share of full head |
|---:|---:|---:|---:|
| 40,525 | 88,182,400 | 84.10 | 16.32% |
| 58,963 | 128,303,488 | 122.36 | 23.74% |
| 106,299 | 231,306,624 | 220.59 | 42.81% |
| 248,320 | 540,344,320 | 515.31 | 100% |

This does **not** automatically save VRAM. The original Strata runtime
allocates a subset head **in addition to** its existing full native head.
Doing the same gather in the present A3B GGML model would speed the head
product if actually routed, but would raise resident VRAM by the selected
subset's size. To save VRAM, the full draft `output.weight` must be absent
from GPU after subset initialization, for example through a derived MTP GGUF
with only the subset output rows while preserving the full tokenizer and
`token_embd.weight`. That is a distinct, opt-in model artifact with explicit
provenance to the original matching MTP GGUF, never a silent replacement.
An alternative loader can stream/gather from the original GGUF before freeing
the full GPU output tensor, but GGML buffer ownership and graph capture make
that more invasive; it needs a separate design and lifetime test.

## Integration constraints in the current A3B engine

`G:/Strata-Moe/third_party/llama.cpp/src/models/qwen35moe.cpp` currently
loads `output.weight` with `{n_embd,n_vocab}` and its MTP graph multiplies
by this full head. The A3B `MtpDraft` samples the resulting full-vocabulary
logits using GPU greedy or the CPU fallback. It assumes a full target token
ID, including in the dependent device K2 chain. The GGML EAGLE3 and DFlash
graphs show an existing `d2t` precedent: load a shortened output head and
scatter its logits back to full-vocabulary positions. QWEN35MOE has no such
branch. An A3B version would need a validated d2t tensor, output-head shape
change limited to MTP-only loading, full-vocabulary logit shape restoration,
and a stable full token ID at every proposal and K2 handoff. It must not
shrink the target head or tokenizer.

There is an additional trap: EAGLE3/DFlash fill omitted logits with
`-INFINITY`, but `G:/Strata-Moe/a3b/speculative.cpp` rejects non-finite
logits in its CPU greedy path and computes `logits - logits` as a GPU finite
guard. Subtracting `-INFINITY` from itself produces NaN, so directly copying
their scatter branch would fail. A finite negative sentinel (with an
argmax/tie safety check) or subset-only sampling followed by device-side
d2t mapping is required. The latter may avoid a 248,320-logit fill but needs
sampler and K2 graph changes. The original Strata sampler's explicit
`sub_to_id`/`id_to_sub` path is the direct design reference.

## Likely benefit and validation boundary

The partial GPU0 Nsight capture in `ANALYSIS.md` attributes 119.057 ms to
122 full-head Q8_0 calls across 61 proposal windows, averaging 0.976 ms
per call. A smaller head could reduce those calls roughly in proportion to
rows **if** the kernel is dominated by reading the head; this is an
unmeasured hypothesis. Even an impossible elimination of all 119.057 ms
from this partial, profiler-perturbed capture is not a whole-request wall
time bound or a demonstrated path to 100 tok/s. A valid same-binary A/B must
measure complete 50K-input/256-output requests, not just a microkernel.

A newly generated A3B subset must include all special/end tokens, reflect
the intended English/code/CJK or multilingual workload, and report token
coverage over the exact prompts and target outputs. Any token outside the
subset can never be proposed, lowering acceptance for that distribution.
The current A3B service only emits tokens sampled by the full target during
verification, so a correct subset changes proposals and speed rather than
intentionally truncating target output choices. Still, proposal changes can
change verification batch shapes, floating-point results, or sampling RNG
consumption; output byte identity and quality cannot be presumed. Compare
target output hashes, accepted/offered rates, first HTTP byte, first model
token, prefill, output tok/s, and whole-card VRAM on 50K/256 requests in
English, code, Chinese, and the user's relevant languages. Keep the full
vocabulary as the default and preserve the current GGUF path as fallback.
