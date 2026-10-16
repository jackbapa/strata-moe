# A3B MTP draft vocabulary subset: opt-in implementation, not a speed result

This work ports the original Strata `--draft-vocab` *idea* to the A3B MTP
path. Original Strata gathers selected native-head rows on GPU and maps
sampled subset positions back to full token IDs. The A3B implementation uses
the same selected-row semantics in a separate derived GGUF: it copies Q8_0
`output.weight` rows without requantizing, retains the complete
`token_embd.weight`, and writes an I64 `d2t` tensor. The QWEN35MOE MTP graph
projects over the smaller head, scatters the logits into the full A3B
vocabulary, then gives full IDs to the existing CPU greedy, CUDA greedy and
dependent K2 proposal paths. Omitted logits use a finite negative sentinel
because A3B's samplers reject non-finite logits. The ordinary MTP GGUF has no
`d2t` and follows the same graph path as before.

This is a functional port of selected output rows, **not** the original
Strata subset sampler's final performance design: Strata samples the compact
logits directly and maps the chosen position with `sub_to_id`, while this
GGML-compatible branch fills/scatters a full 248,320-logit tensor each draft
step. Its cost must be included in the end-to-end A/B before claiming any
speedup.

Physical GGUF construction checks the downloaded Q8_0 MTP file's pinned
size and SHA-256 before copying; `build --dry-run` only checks structure and
the ID manifest. It does not modify either downloaded model.

`tools/a3b_draft_vocab.py` validates A3B target/draft tokenizer tokens,
merges, token types and pre-tokenizer, generates IDs from **this A3B
tokenizer**, preserves every special/end token, and refuses an ID binary
without its matching fingerprint manifest. The target and draft tokenizer
vocabularies match, although their chat template and padding-token metadata
differ. Flash-Next ID files are not accepted as unverified A3B input.

The broad `--ascii --add cjk` static candidate here selects 193,959 of
248,320 A3B IDs. The Q8_0 output rows go from 540,344,320 to 422,054,784
bytes, a **calculated weight reduction of 112.81 MiB**. A `d2t` tensor adds
1,551,672 bytes to the derived GGUF. This is neither measured card VRAM nor a
throughput result. The broad set is large enough that its benefit may be
small. Corpus-selected sets can be smaller but may miss output languages and
hurt MTP acceptance.

Re-tokenizing each of the two previously saved Chinese 256-token outputs
from the 2026-10-15 async-input A/B gives 12/256 pieces outside the broad
ASCII+CJK set. They are curly quotation marks and an em-dash token, not
Chinese ideographs. This is only static text coverage (not the model's exact
generated token trace or a measured MTP acceptance rate). The generator now
offers `--punct` to include standalone Unicode punctuation; a revised
candidate was generated as `a3b-ascii-cjk-punct.ids`: 194,208 of 248,320
rows. It adds 249 rows (541,824 Q8_0 bytes) to the first candidate, including
all three missing token IDs above. Re-tokenizing those two saved output texts
with this candidate gives 0/256 pieces outside the set for each. This is
still only text coverage on two samples, not draft acceptance or broad
language coverage.

Offline validation so far:

- `python tools/test_a3b_draft_vocab.py`: 3/3 synthetic GGUF tests. They check
  byte-exact selected Q8 rows, byte-exact full embedding, I64 `d2t`, intact
  metadata, output file SHA, required special tokens, duplicate-ID rejection,
  and tokenizer mismatch rejection.
- Real IQ3_S target plus matching Q8_0 draft: selection and `build --dry-run`
  passed for both candidates. The punctuation-inclusive derived Q8_0 MTP
  file was physically written at
  `models/a3b/mtp-Qwen3.6-35B-A3B-Q8_0-DV-ASCII-CJK-PUNCT.gguf`, size
  1,874,455,616 bytes, SHA-256
  `7a0bd6c1a1c8d5037262c52b50375d063e128a254f5f6348a0772543893512ae`.
  The original source's pinned SHA-256 was verified before copying, and every
  non-output tensor remains byte-identical in the generated GGUF.
- The C++ GGML CPU graph test `a3b/draft_vocab_map_test.cpp` built in the
  isolated `build-a3b-draft-vocab-cpu` directory and passed CTest. The full
  CPU `strata-moe.exe` also built with the modified QWEN35MOE loader/graph.
- The CUDA engine and strict speculative fixture compiled in isolated
  `build-a3b-draft-vocab-cuda` (VS2022, CUDA 12.6, sm75, CUDA graphs). The
  engine SHA-256 is
  `a6020a08638e937963679a888084f21b7a8a78b94244695b778e6ce107738d46`;
  vendor QWEN35MOE source SHA-256
  `9edcd5df8b52e070f169ab2755de762e320296cf3600495a7e1a6dc61a6be7ff`
  was unchanged before and after this build.
- Real-model strict fixture: full Q8 and subset Q8 ran with the **same**
  isolated fixture binary. Both passed all eight output limits
  (1/2/3/4/5/8/16/32), checking target-only sequential parity. All 16
  serialized draft/target state files match byte-for-byte between arms;
  29 proposal trace lines and all eight offered/accepted class lines match.
  Repeating both arms with `STRATA_A3B_MTP_DEVICE_CHAIN_K2=1` also passed
  those comparisons. The strict fixture itself does not print the number of
  successful dependent chains.
- A separate 64-output short CLI run of the subset with the new engine and
  dependent CUDA K2 enabled recorded 23 successful device-chain calls, zero
  fallbacks and zero inter-step context synchronizations. Its 125 token/s
  output rate is a one-arm short-prompt smoke result, **not** a measured
  full-head versus subset benefit or a 50K-context result.

The two ready A/B configs, `full.snapshot.json` and `subset.snapshot.json`,
keep the existing four optimization switches and selected 256K service flags.
Both use the same new engine binary; the only inference argument difference
is `--mtp-model`. Neither config enables dependent K2 chain, so the planned
50K/256 run isolates this vocabulary-head change.

The smaller head still uses full-vocabulary fill/scatter before GPU sampling.
A possible next opt-in step, **not implemented or measured here**, is to
leave compact logits in the GPU-sampler graph and map its sampled index to a
full A3B token ID before publishing `t_sampled` to the dependent K2 input.
GGML's CUDA `GET_ROWS` supports an I32 value table and I32 sampled index;
this derived file currently stores I64 `d2t` for `SET_ROWS`, so direct
sampling would require an I32 map in a new derived file. CPU sampling and
raw-logit diagnostics would keep the existing full-scatter fallback. That
change needs separate state/parity and 50K performance tests; it is not a
reason to infer a benefit from the current subset experiment.

The performance experiment must use a separately built binary and a derived
GGUF, keeping the current selected service/binary frozen. Compare ordinary
full Q8 draft and derived A3B subset on identical uncached 50,000-token
inputs, 256 output tokens, 262,144 configured context, GPU 0, K2 and target
IQ3_S. Run at least alternating repeated requests for English, code, Chinese
and any relevant workload. Record first HTTP byte, first *model* token,
prefill rate, output tok/s, peak whole-card VRAM, draft offered/accepted,
final output hash and model/version hashes. Proposal traces are expected to
differ when the subset removes a high-ranked draft token; this is a quality
tradeoff, not automatic correctness failure. Validate all proposal IDs are
full A3B IDs and that the final output is acceptable before considering the
feature for normal use. Do not turn it on by default without a measured gain.
The comparison must use the **same new CUDA binary** for both draft GGUFs;
the previous 83 token/s selected-service baseline is not a valid comparison
against a new binary. Keep the subset opt-in if the repeated 50K/256 A/B does
not show an output-speed improvement without a material first-token regression,
or if proposal acceptance degrades. A smaller Q8 head alone does not establish
an end-to-end win.
