# AGENTS.md

Strata-Moe is a local Qwen3.5/3.6-35B-A3B MoE inference project derived from
[Strata](https://github.com/Niko1221/Strata). Its A3B engine is in `a3b/`, with
the pinned llama.cpp/ggml source and A3B patches fetched at build time. The
Python API and web app are in `serve/`. The release path uses the Unsloth
Qwen3.6-35B-A3B UD-IQ3_S GGUF and, in the full-GPU and hybrid profiles, the matching
ggml-org Q8_0 MTP file. Other old Strata models are not part of the default install.

## Installing for a user

Read [README.md](README.md). On Windows run `SETUP.bat`; it creates `.venv`,
installs pinned Python/CMake/Ninja dependencies, compiles the CUDA or CPU A3B
engine, downloads only the selected model files with SHA-256 checks, exports the
tokenizer, and starts the local server. `START-HERE.bat` is the later-run entry.
Use `--mode cpu`, `--mode 6g`, `--mode hybrid-no-mtp`, `--mode hybrid`, or `--mode full` when explicitly
selecting a profile. Use `--reconfigure` to replace a previously generated
`strata-moe.json`. Never expose the server beyond `127.0.0.1` without an API key.

## Working on the code

- Do not modify the separate upstream `G:\Strata` checkout for this project.
- Keep the model files in ignored `models/`, build outputs in ignored `build*/`,
  and the generated local config out of Git. Do not commit benchmark state dumps.
- Attribute performance claims to a specific model, quantization, machine,
  actual input length, configured context limit, output length and configuration.
  Distinguish HTTP first byte from first model token, and a single trial from a
  repeated-trial peak or median. The fixed-256K input-length matrix is in
  [bench/results/2026-10-16-256k-context-matrix](bench/results/2026-10-16-256k-context-matrix/README.md);
  the controlled 50K/256 result is in
  [bench/results/2026-10-16-a3b-iq3s-combined-exact-r3](bench/results/2026-10-16-a3b-iq3s-combined-exact-r3/README.md).
- Do not claim every Strata kernel or the entire upstream execution pipeline
  has been ported. The A3B engine mixes adapted Strata kernels and GGML paths.
- Run `python tools/test_a3b_download.py` and `python tools/test_a3b_options.py`
  after changing model selection, download, or config behavior. Build the
  backend a changed native file touches before asking for review.
