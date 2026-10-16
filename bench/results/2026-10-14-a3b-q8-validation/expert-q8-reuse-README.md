# Experimental Q8 expert duplicate-ID reuse, 2026-10-14

The optional `STRATA_A3B_EXPERT_Q8_REUSE=1` kernel passed strict numerical checks, but this first implementation did **not** establish a speed benefit. Keep it disabled in the production configuration unless a later independent model A/B demonstrates a gain. The exact initial patch is preserved as [expert-q8-reuse-v1-negative.patch](expert-q8-reuse-v1-negative.patch), with its SHA-256 matching the comparison JSON. The active source may contain later experiments; their timings must use separate logs.

Hardware: GPU 0, NVIDIA RTX 2080 Ti, SM75, 22,527 MiB reported by CUDA, on Windows. The standalone fixture keeps its Q8 weights on that GPU. These measurements concern GPU graph calls, not model token throughput, API latency, PCIe expert uploads, or the 100 output token/s goal.

## Correctness

Both [off](expert-q8-reuse-off.log) and [on](expert-q8-reuse-on.log) passed **432** projection/full-FFN cases against independent ordinary one-channel GPU graphs. The one-channel oracle cannot enter the new kernel. The fixture covers K=2,048/512/256/768; 1/2/3/8/24/64/65 channels where applicable; unique, identical and mixed expert IDs; groups with more than eight duplicates; shared/per-lane activations; padded input strides; zero activations and varied scales. The complete FFN contains adjacent gate/up/SwiGLU nodes and the down projection.

This confirms the tested kernel's arithmetic. It does not prove end-to-end MTP state, rollback, cache lifetime or model output correctness; those require separate engine tests.

## Timing

Each timed cell ran five warmup graph calls followed by 100 **synchronous graph API** calls. There was one sequential off/on run pair, with no repeat-pair uncertainty estimate. The [comparison JSON](expert-q8-reuse-comparison.json) retains all 64 paired cells, exact values, source/log/binary fingerprints and caveats.

| Subset | Cells | Median paired on/off time | Cells faster with reuse |
| --- | ---: | ---: | ---: |
| Eligible 2–64 channels | 56 | 1.679 | 3 |
| Actual 2,048→512→2,048 FFN, eligible channels | 10 | 1.550 | 1 |
| Same FFN, 24/64 channels | 4 | 1.356 | 0 |
| Unchanged one-channel control | 8 | 0.495 | 7 |

A ratio above one means slower. The unchanged control ranged from **0.324 to 1.949**, despite bypassing the new kernel in both arms. This exposes substantial host/driver scheduling noise in these short wall-clock cells; the exact aggregate slowdown must not be treated as a stable kernel-only estimate. Nevertheless, the eligible cells consistently favor the old path, so these results do not justify enabling the optimization.

Representative complete-FFN cells, in microseconds:

| Channels | Fixture routing | Off | On |
| ---: | --- | ---: | ---: |
| 8 | Same expert | 102.088 | 155.683 |
| 8 | Mixed 11 experts | 108.945 | 334.252 |
| 24 | Same expert | 214.693 | 293.254 |
| 24 | Mixed 11 experts | 218.686 | 541.322 |
| 64 | Same expert | 488.196 | 657.592 |
| 64 | Mixed 11 experts | 497.368 | 663.105 |

The identical-ID and mixed-ID patterns stress grouping correctness. They may repeat an expert within the same eight-lane token and therefore are not representative legal top-8 routing traces. The unique-ID pattern passed correctness checks but was not timed. A production decision needs real-model routing and model throughput measurements.

## Implementation limits and possible follow-up

The opt-in dispatch applies only to SM75 Q8, the exact flattened one-pseudo-token layout with 2–64 channels, compatible input broadcast, and bias-free SwiGLU gate/up fusion. Other paths retain the previous dispatch. Every eighth duplicate starts another bounded group; singleton K partition, accumulation and reduction are retained.

Possible costs include repeated per-CTA ID scans, an eight-member accumulator footprint even for small groups, and the dynamically indexed per-thread member array. The build reported unused-variable warnings; its output did not include a register/spill resource report, so no claim about zero local-memory traffic follows from that absence. A shared per-CTA member plan or group-size specializations might improve this experiment, but neither has been implemented or measured here.

Reproduce after building `strata-moe-expert-q8-reuse-test`:

```powershell
$env:CUDA_VISIBLE_DEVICES = '0'
$env:STRATA_A3B_EXPERT_Q8_REUSE = '0'
build-a3b/bin/strata-moe-expert-q8-reuse-test.exe --benchmark 100
$env:STRATA_A3B_EXPERT_Q8_REUSE = '1'
build-a3b/bin/strata-moe-expert-q8-reuse-test.exe --benchmark 100
```

No original `G:\Strata` file was changed for this experiment.

## Second experiment: cap2 and shared planning

The second implementation reduces the group capacity from eight to two. Thread 0 builds the member plan in shared memory, replacing the repeated per-thread scans and dynamically indexed local member array. It retains the canonical arithmetic and the default-off switch. Its exact source patch is [expert-q8-reuse-v2-cap2.patch](expert-q8-reuse-v2-cap2.patch).

[The cap2 run](expert-q8-reuse-on-cap2.log) also passed all **432** strict cases. Its 100-call timing cells are compared to the earlier off and cap8 runs in [expert-q8-reuse-cap2-comparison.json](expert-q8-reuse-cap2-comparison.json). This was **not** a new interleaved off/on experiment; singleton control noise remains a material limitation.

| Subset | Cells | Median paired cap2/off | Median paired cap2/cap8 | Cap2 faster than off |
| --- | ---: | ---: | ---: | ---: |
| Eligible 2–64 channels | 56 | 1.238 | 0.834 | 9 |
| Actual 2,048→512→2,048 FFN, eligible channels | 10 | 1.213 | 0.897 | 3 |
| Same FFN, 24/64 channels | 4 | 1.057 | 0.782 | 2 |
| 2,048→512 projection, eligible channels | 10 | 1.458 | 0.951 | 2 |
| 512→2,048 projection, eligible channels | 10 | 1.303 | 0.767 | 0 |
| Unchanged one-channel control | 8 | 0.712 | 1.216 | 7 |

Cap2 improved on cap8 in 46 of 56 eligible cells, but still lost to the original path in 47 of 56. Representative actual expert-shape cells, in microseconds:

| Work | Channels | Fixture routing | Off | Cap8 | Cap2 |
| --- | ---: | --- | ---: | ---: | ---: |
| Full 2,048→512→2,048 FFN | 24 | Same expert | 214.693 | 293.254 | 211.616 |
| Full 2,048→512→2,048 FFN | 24 | Mixed 11 experts | 218.686 | 541.322 | 216.535 |
| Full 2,048→512→2,048 FFN | 64 | Same expert | 488.196 | 657.592 | 554.552 |
| Full 2,048→512→2,048 FFN | 64 | Mixed 11 experts | 497.368 | 663.105 | 558.794 |
| 2,048→512 projection | 24 | Same expert | 122.302 | 126.452 | 120.244 |
| 2,048→512 projection | 24 | Mixed 11 experts | 126.571 | 164.994 | 117.641 |
| 512→2,048 projection | 24 | Same expert | 105.185 | 148.761 | 135.479 |
| 512→2,048 projection | 24 | Mixed 11 experts | 102.078 | 145.414 | 134.601 |

The approximately 1–1.4% reduction in the 24-channel full-FFN cells is much smaller than the unchanged singleton control's spread (cap2/off **0.269–1.114**). It does not establish a repeatable speed gain. Cap2 reduces the first experiment's regression, but these microbenchmarks still do not justify enabling expert reuse in production. Real-model routing and API A/B results take precedence.
