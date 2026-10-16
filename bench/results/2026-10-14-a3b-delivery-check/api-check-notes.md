# API verification notes

The extended check initially failed an assertion that the second and third positive-temperature requests must have identical output for the same seed. The live adaptive cache was still changing placements and uploads; this was not a fixed-arithmetic experiment. The three replies had 33, 40 and 34 completion tokens. All were well-formed coherent Chinese replies, and the configured seed was forwarded by the existing server.

The test now records `warm_seeded_choices_equal` instead of asserting byte equality under changing placements. It retains hard assertions for health, model listing, greedy math, stream termination, Anthropic text, 1/2/3/4/5/8-token output boundaries, long prefill carryover, non-empty sampled replies and real MTP offer counters. This change does not relax either expert parity tolerances or the independent sequential-prefix rollback checks.

Fixed placement was separately checked with both GPU and CPU experts: all four accepted-prefix cases have maximum logit error and KL equal to zero. This does not promise deterministic full replies while the live cache keeps adapting.

The rerun's complete responses and the observed seed-repeat equality flag are in `api-extended.json`.
