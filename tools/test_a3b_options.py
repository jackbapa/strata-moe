"""Offline public config/benchmark boundary checks; no inference or network."""
from __future__ import annotations
import contextlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import a3b
from bench_a3b import adaptive_counters, cache_decay_counters, mtp_fused_counters, prefill_share_counters


def string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def tiny_tokenizer_gguf(path):
    # A real, minimal GGUF header exercises the production metadata reader;
    # the config generator needs tokenizer metadata, not tensor payloads.
    fields = [("general.architecture", 8, string("qwen35moe")),
              ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + string("a") + string("b")),
              ("tokenizer.ggml.merges", 9, struct.pack("<IQ", 8, 0)),
              ("tokenizer.ggml.token_type", 9, struct.pack("<IQii", 5, 2, 1, 1)),
              ("tokenizer.chat_template", 8, string("{{ messages }}"))]
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, len(fields)) +
                     b"".join(string(key) + struct.pack("<I", kind) + payload for key, kind, payload in fields))


class ConfigOptionsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="a3b-options-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "model with spaces.gguf"
        tiny_tokenizer_gguf(self.model)
        self.profile = self.root / "local profile.txt"
        self.profile.write_text("format/model hashes are checked by the engine at startup", encoding="utf-8")
        self.mtp = self.root / "mtp.gguf"
        self.mtp.write_bytes(b"configuration checks existence; engine validates actual MTP")

    def configure(self, *extra, name="config.json"):
        output = self.root / name
        arguments = ["a3b.py", "config", "--model", str(self.model), "--out", str(output), *map(str, extra)]
        with patch.object(a3b, "ROOT", self.root), patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            a3b.main()
        return json.loads(output.read_text(encoding="utf-8"))

    def test_defaults_do_not_enable_new_optimizations(self):
        config = self.configure()
        self.assertEqual(config["args"], ["--model", str(self.model.resolve()), "--ctx", "4096", "--batch", "128",
                                          "--device", "0", "--vram-mib", "0"])
        self.assertTrue({"--expert-profile", "--async-expert-events", "--gpu-static-experts", "--gpu-dynamic-experts", "--mtp-adaptive", "--mtp-fused-rounds", "--cache-decay-interval", "--hybrid-experts", "--hybrid-doorbell", "--cpu-expert-pool", "--prefill-cpu-share", "--pcie-frac", "--native-dense-exact", "--native-attention-exact", "--strata-prefill-mmq", "--kv-q8", "--kv-q4", "--kv-tiered", "--kv-tiered-f16-short"}.isdisjoint(config["args"]))
        self.assertEqual(config["host"], "127.0.0.1")
        self.assertEqual(config["idle_unload_s"], 0)

    def test_q8_kv_is_explicit_and_accepts_native_256k_window(self):
        config = self.configure("--ctx", 262144, "--kv-q8", name="q8-kv.json")
        self.assertIn("--kv-q8", config["args"])
        self.assertEqual(config["args"][config["args"].index("--ctx") + 1], "262144")

    def test_q4_kv_is_explicit_and_excludes_q8(self):
        config = self.configure("--ctx", 262144, "--kv-q4", name="q4-kv.json")
        self.assertIn("--kv-q4", config["args"])
        self.assertNotIn("--kv-q8", config["args"])
        with self.assertRaises(ValueError):
            self.configure("--kv-q4", "--kv-q8", name="invalid-kv.json")

    def test_tiered_kv_requires_long_context_and_allows_low_vram_tiers(self):
        for options in ((), ("--native-gpu", "--ctx", 65536)):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "--kv-tiered requires"):
                self.configure(*options, "--kv-tiered", name="invalid-tier.json")
        config = self.configure("--native-gpu", "--ctx", 262144, "--kv-tiered", name="tiered-kv.json")
        self.assertIn("--kv-tiered", config["args"])
        self.assertEqual(config["args"][config["args"].index("--ctx") + 1], "262144")
        low_vram = self.configure("--ctx", 262144, "--kv-tiered", "--kv-tier-low-vram", name="low-vram-tier.json")
        self.assertIn("--kv-tier-low-vram", low_vram["args"])
        self.assertIn("--kv-tiered", low_vram["args"])
        with self.assertRaisesRegex(ValueError, "--kv-tier-low-vram requires"):
            self.configure("--ctx", 262144, "--kv-tier-low-vram", name="invalid-low-vram-tier.json")

    def test_f16_short_kv_tier_is_explicit_and_keeps_mtp_attention_batched(self):
        config = self.configure("--native-gpu", "--ctx", 262144, "--mtp-model", self.mtp,
                                "--kv-tiered", "--kv-tiered-f16-short", name="f16-short.json")
        for flag in ("--kv-tiered", "--kv-tiered-f16-short", "--native-gpu", "--mtp-model"):
            self.assertEqual(config["args"].count(flag), 1, flag)
        for flag in ("--kv-q8", "--kv-q4", "--native-attention-exact"):
            self.assertNotIn(flag, config["args"])
        explicit = self.configure("--native-gpu", "--ctx", 262144, "--mtp-model", self.mtp,
                                  "--kv-tiered", "--kv-tiered-f16-short", "--native-attention-exact",
                                  name="f16-short-exact.json")
        self.assertEqual(explicit["args"].count("--native-attention-exact"), 1)

    def test_f16_short_kv_tier_rejects_missing_tier_or_explicit_kv_formats_before_export(self):
        cases = [(), ("--kv-q8",), ("--kv-q4",)]
        for index, extra in enumerate(cases):
            options = ("--native-gpu", "--ctx", 262144) + (("--kv-tiered",) if extra else ())
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "--kv-tiered-f16-short requires"):
                self.configure(*options, "--kv-tiered-f16-short", *extra, name=f"invalid-f16-tier-{index}.json")
            self.assertFalse((self.root / f"invalid-f16-tier-{index}.json").exists())

    def test_grouped_q6_requires_exact_mtp_and_is_opt_in(self):
        with self.assertRaises(ValueError):
            self.configure("--q6-grouped-exact", name="invalid-q6.json")
        config = self.configure("--native-gpu", "--native-dense-exact",
                                "--mtp-model", self.mtp, "--q6-grouped-exact",
                                name="grouped-q6.json")
        self.assertEqual(config["env"], {"STRATA_A3B_Q6_GROUPED_EXACT": "1"})

    def test_strata_iq_expert_kernel_is_explicit_and_separate_from_q6(self):
        for flags in (("--strata-native-experts",),
                      ("--native-gpu", "--device", "1", "--strata-native-experts"),
                      ("--strata-fused-gu",)):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                self.configure(*flags, name="invalid-strata.json")
        config = self.configure("--native-gpu", "--device", "0", "--strata-native-experts",
                                "--strata-fused-gu", name="strata-iq.json")
        self.assertIn("--strata-native-experts", config["args"])
        self.assertEqual(config["env"], {"STRATA_A3B_FUSED_GU": "1"})
        self.assertNotIn("STRATA_A3B_Q6_GROUPED_EXACT", config["env"])

    def test_original_strata_prefill_mmq_is_independent_opt_in(self):
        for flags in (("--strata-prefill-mmq",),
                      ("--native-gpu", "--device", "1", "--strata-prefill-mmq")):
            with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, "--strata-prefill-mmq requires"):
                self.configure(*flags, name="invalid-prefill-mmq.json")
        config = self.configure("--native-gpu", "--strata-prefill-mmq", name="prefill-mmq.json")
        self.assertEqual(config["args"].count("--strata-prefill-mmq"), 1)
        self.assertNotIn("--strata-native-experts", config["args"])

    def test_iq3_model_does_not_share_different_q8_mtp_embedding(self):
        self.model = self.root / a3b.QUANTS["IQ3_S"][0]
        tiny_tokenizer_gguf(self.model)
        config = self.configure("--native-gpu", "--mtp-model", self.mtp,
                                "--ctx", 262144, "--kv-q8", name="iq3-mtp.json")
        self.assertNotIn("--share-mtp-weights", config["args"])
        self.assertIn("--defer-mtp-catchup", config["args"])
        self.assertIn("--kv-q8", config["args"])

    def test_profile_requires_gpu_only_and_a_local_file_before_export(self):
        cases = [("--expert-profile", self.profile),
                 ("--gpu-only-experts", "--expert-profile", self.root / "missing.txt"),
                 ("--gpu-only-experts", "--expert-profile", self.root)]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises((ValueError, FileNotFoundError)):
                self.configure(*arguments)
            self.assertFalse((self.root / "models").exists())
            self.assertFalse((self.root / "config.json").exists())

    def test_profile_and_async_are_independent_opt_ins(self):
        for index, (use_profile, use_async) in enumerate(((True, False), (False, True), (True, True))):
            arguments = ["--gpu-only-experts"]
            if use_profile:
                arguments += ["--expert-profile", self.profile]
            if use_async:
                arguments += ["--async-expert-events"]
            config = self.configure(*arguments, name=f"config{index}.json")
            self.assertEqual("--expert-profile" in config["args"], use_profile)
            self.assertEqual("--async-expert-events" in config["args"], use_async)
            if use_profile:
                self.assertEqual(config["args"][config["args"].index("--expert-profile") + 1], str(self.profile.resolve()))
            self.assertNotIn("--mtp-adaptive", config["args"])
            self.assertNotIn("--gpu-static-experts", config["args"])
            self.assertNotIn("--gpu-dynamic-experts", config["args"])

    def test_prefill_ring_and_host_stager_are_explicit(self):
        baseline = self.configure("--gpu-only-experts", name="ring-default.json")
        self.assertNotIn("--expert-ring-banks", baseline["args"])
        self.assertNotIn("--prefill-copy-ahead", baseline["args"])
        tuned = self.configure("--gpu-only-experts", "--expert-ring-banks", 4,
                               "--prefill-copy-ahead", name="ring-opt-in.json")
        self.assertEqual(tuned["args"][tuned["args"].index("--expert-ring-banks") + 1], "4")
        self.assertIn("--prefill-copy-ahead", tuned["args"])
        for invalid in (1, 9):
            with self.assertRaises(ValueError):
                self.configure("--gpu-only-experts", "--expert-ring-banks", invalid,
                               name=f"ring-invalid-{invalid}.json")

    def test_hybrid_pool_profile_and_gpu_trunk_options_are_explicit(self):
        config = self.configure("--hybrid-experts", "--cpu-expert-pool", "--pageable-experts",
                                "--expert-profile", self.profile, "--async-expert-events",
                                "--cache-decay-interval", 4, "--native-dense-exact", "--native-attention-exact",
                                "--mtp-model", self.mtp, "--mtp-adaptive")
        for flag in ("--hybrid-experts", "--cpu-expert-pool", "--fused-moe", "--gpu-embedding", "--gpu-sampling",
                     "--pageable-experts", "--expert-profile", "--async-expert-events", "--cache-decay-interval",
                     "--native-dense-exact", "--native-attention-exact", "--mtp-adaptive", "--exact-verify",
                     "--share-mtp-weights", "--defer-mtp-catchup"):
            self.assertEqual(config["args"].count(flag), 1, flag)
        self.assertEqual(config["args"][config["args"].index("--cache-decay-interval") + 1], "4")
        for flag in ("--native-gpu", "--gpu-only-experts", "--gpu-static-experts", "--gpu-dynamic-experts"):
            self.assertNotIn(flag, config["args"])

    def test_hybrid_does_not_force_pool_profile_or_exact_trunk(self):
        config = self.configure("--hybrid-experts")
        for flag in ("--hybrid-experts", "--fused-moe", "--gpu-embedding"):
            self.assertIn(flag, config["args"])
        for flag in ("--hybrid-doorbell", "--cpu-expert-pool", "--prefill-cpu-share", "--mtp-fused-rounds", "--expert-profile", "--async-expert-events", "--cache-decay-interval",
                     "--native-dense-exact", "--native-attention-exact", "--mtp-model", "--pcie-frac"):
            self.assertNotIn(flag, config["args"])

    def test_doorbell_implies_hybrid_pool_and_explicit_fixed_cache_constraints(self):
        config = self.configure("--hybrid-doorbell", "--expert-profile", self.profile)
        for flag in ("--hybrid-doorbell", "--hybrid-experts", "--cpu-expert-pool", "--fused-moe",
                     "--gpu-embedding", "--gpu-sampling", "--expert-profile", "--pcie-frac", "--cache-decay-interval"):
            self.assertEqual(config["args"].count(flag), 1, flag)
        for flag, value in (("--device", "0"), ("--expert-profile", str(self.profile.resolve())),
                            ("--pcie-frac", "0.0"), ("--cache-decay-interval", "0")):
            self.assertEqual(config["args"][config["args"].index(flag) + 1], value)
        for flag in ("--native-gpu", "--gpu-only-experts", "--gpu-static-experts", "--gpu-dynamic-experts",
                     "--mtp-model", "--native-dense-exact", "--native-attention-exact"):
            self.assertNotIn(flag, config["args"])
        explicit = self.configure("--hybrid-doorbell", "--hybrid-experts", "--cpu-expert-pool",
                                  "--expert-profile", self.profile, "--pcie-frac", 0,
                                  "--cache-decay-interval", 0, name="explicit-doorbell.json")
        self.assertEqual(explicit["args"], config["args"])

    def test_doorbell_allows_existing_hybrid_tuning_and_mtp_draft_boundaries(self):
        for drafts in (1, 7):
            with self.subTest(drafts=drafts):
                config = self.configure("--hybrid-doorbell", "--expert-profile", self.profile,
                                        "--pageable-experts", "--async-expert-events", "--native-dense-exact",
                                        "--native-attention-exact", "--mtp-model", self.mtp,
                                        "--mtp-draft", drafts, "--mtp-adaptive", name=f"doorbell{drafts}.json")
                for flag in ("--hybrid-doorbell", "--hybrid-experts", "--cpu-expert-pool", "--fused-moe",
                             "--pageable-experts", "--async-expert-events", "--native-dense-exact",
                             "--native-attention-exact", "--mtp-adaptive", "--exact-verify",
                             "--share-mtp-weights", "--defer-mtp-catchup"):
                    self.assertEqual(config["args"].count(flag), 1, flag)
                self.assertEqual(config["args"][config["args"].index("--mtp-draft") + 1], str(drafts))

    def test_doorbell_rejects_conflicting_modes_and_parameters_before_export(self):
        cases = [("--native-gpu",), ("--gpu-only-experts",), ("--gpu-static-experts",),
                 ("--gpu-dynamic-experts",), ("--device", 1), ("--pcie-frac", .25),
                 ("--pcie-frac", 1), ("--cache-decay-interval", 4), ("--mtp-draft", 8),
                 ("--mtp-model", self.mtp, "--mtp-draft", 8)]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.configure("--hybrid-doorbell", "--expert-profile", self.profile, *options)
            self.assertFalse((self.root / "models").exists())
            self.assertFalse((self.root / "config.json").exists())

    def test_doorbell_requires_existing_local_profile_before_export(self):
        cases = [(), ("--expert-profile", self.root / "missing.profile"), ("--expert-profile", self.root)]
        for options in cases:
            with self.subTest(options=options), self.assertRaises((ValueError, FileNotFoundError)):
                self.configure("--hybrid-doorbell", *options)
            self.assertFalse((self.root / "models").exists())
            self.assertFalse((self.root / "config.json").exists())

    def test_normal_hybrid_retains_nonzero_policy_and_eight_draft_options(self):
        config = self.configure("--hybrid-experts", "--expert-profile", self.profile,
                                "--pcie-frac", .25, "--cache-decay-interval", 4,
                                "--mtp-model", self.mtp, "--mtp-draft", 8)
        for flag, value in (("--pcie-frac", "0.25"), ("--cache-decay-interval", "4"), ("--mtp-draft", "8")):
            self.assertEqual(config["args"][config["args"].index(flag) + 1], value)
        self.assertNotIn("--hybrid-doorbell", config["args"])
        self.assertNotIn("--cpu-expert-pool", config["args"])

    def test_prefill_share_requires_hybrid_pool_and_doorbell_can_supply_both(self):
        for options in ((), ("--hybrid-experts",), ("--native-gpu",), ("--gpu-only-experts",)):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "--prefill-cpu-share requires"):
                self.configure("--prefill-cpu-share", *options)
            self.assertFalse((self.root / "models").exists())
            self.assertFalse((self.root / "config.json").exists())
        hybrid = self.configure("--hybrid-experts", "--cpu-expert-pool", "--prefill-cpu-share",
                                "--pcie-frac", 1, name="hybrid-share.json")
        doorbell = self.configure("--hybrid-doorbell", "--expert-profile", self.profile,
                                  "--prefill-cpu-share", name="doorbell-share.json")
        for config in (hybrid, doorbell):
            for flag in ("--prefill-cpu-share", "--hybrid-experts", "--cpu-expert-pool"):
                self.assertEqual(config["args"].count(flag), 1, flag)

    def test_fused_mtp_requires_model_gpu_zero_exact_dense_and_at_most_seven_drafts(self):
        cases = [("--hybrid-experts", "--native-dense-exact"),
                 ("--mtp-model", self.mtp),
                 ("--mtp-model", self.mtp, "--hybrid-experts"),
                 ("--mtp-model", self.mtp, "--native-gpu", "--native-dense-exact", "--device", 1),
                 ("--mtp-model", self.mtp, "--gpu-only-experts", "--native-dense-exact", "--mtp-draft", 8)]
        for options in cases:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "--mtp-fused-rounds requires"):
                self.configure("--mtp-fused-rounds", *options)
            self.assertFalse((self.root / "models").exists())
            self.assertFalse((self.root / "config.json").exists())
        with self.assertRaises(FileNotFoundError):
            self.configure("--mtp-fused-rounds", "--hybrid-experts", "--native-dense-exact",
                           "--mtp-model", self.root / "missing-mtp.gguf")
        self.assertFalse((self.root / "models").exists())

    def test_fused_mtp_preserves_existing_automatic_gpu_sampling_and_hidden_options(self):
        for index, mode in enumerate(("--native-gpu", "--gpu-only-experts", "--hybrid-experts", "--hybrid-doorbell")):
            options = [mode, "--mtp-model", self.mtp, "--native-dense-exact", "--mtp-draft", 7]
            if mode == "--hybrid-doorbell":
                options += ["--expert-profile", self.profile]
            fixed = self.configure(*options, name=f"unfused{index}.json")
            fused = self.configure(*options, "--mtp-fused-rounds", name=f"fused{index}.json")
            self.assertEqual([argument for argument in fused["args"] if argument != "--mtp-fused-rounds"], fixed["args"])
            for flag in ("--mtp-fused-rounds", "--gpu-sampling", "--defer-mtp-catchup", "--share-mtp-weights",
                         "--native-dense-exact", "--native-attention-exact", "--exact-verify"):
                self.assertEqual(fused["args"].count(flag), 1, flag)
        static = self.configure("--gpu-only-experts", "--gpu-static-experts", "--expert-profile", self.profile,
                                "--mtp-model", self.mtp, "--mtp-fused-rounds", name="static-fused.json")
        self.assertEqual(static["args"].count("--native-dense-exact"), 1)

    def test_existing_mtp_does_not_force_fused_rounds_or_prefill_share(self):
        config = self.configure("--hybrid-experts", "--cpu-expert-pool", "--mtp-model", self.mtp)
        self.assertNotIn("--mtp-fused-rounds", config["args"])
        self.assertNotIn("--prefill-cpu-share", config["args"])

    def test_explicit_pcie_fraction_requires_hybrid_and_finite_unit_interval(self):
        cases = [("--pcie-frac", 0), ("--gpu-only-experts", "--pcie-frac", .5),
                 ("--native-gpu", "--pcie-frac", 1), ("--hybrid-experts", "--pcie-frac", -.01),
                 ("--hybrid-experts", "--pcie-frac", 1.01),
                 ("--hybrid-experts", "--pcie-frac=nan"), ("--hybrid-experts", "--pcie-frac=inf"),
                 ("--hybrid-experts", "--pcie-frac=-inf")]
        for options in cases:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "--pcie-frac requires"):
                self.configure(*options)
            self.assertFalse((self.root / "models").exists())
            self.assertFalse((self.root / "config.json").exists())

    def test_pcie_fraction_boundaries_and_interior_forward_only_when_explicit(self):
        for index, fraction in enumerate((0, .25, 1)):
            config = self.configure("--hybrid-experts", "--pcie-frac", fraction, name=f"pcie{index}.json")
            self.assertEqual(config["args"].count("--pcie-frac"), 1)
            self.assertEqual(float(config["args"][config["args"].index("--pcie-frac") + 1]), fraction)

    def test_expert_modes_are_mutually_exclusive_before_export(self):
        for options in (("--hybrid-experts", "--native-gpu"), ("--hybrid-experts", "--gpu-only-experts"),
                        ("--native-gpu", "--gpu-only-experts"),
                        ("--hybrid-experts", "--native-gpu", "--gpu-only-experts")):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "select one GPU expert mode"):
                self.configure(*options)
            self.assertFalse((self.root / "models").exists())
            self.assertFalse((self.root / "config.json").exists())

    def test_pool_and_exact_native_dependencies_fail_before_export(self):
        for options in (("--cpu-expert-pool",), ("--cpu-expert-pool", "--gpu-only-experts"),
                        ("--cpu-expert-pool", "--native-gpu"), ("--native-dense-exact",),
                        ("--native-attention-exact",), ("--async-expert-events",)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.configure(*options)
            self.assertFalse((self.root / "models").exists())
            self.assertFalse((self.root / "config.json").exists())

    def test_explicit_exact_flags_work_in_all_three_modes_without_duplicates(self):
        for index, mode in enumerate(("--native-gpu", "--gpu-only-experts", "--hybrid-experts")):
            config = self.configure(mode, "--native-dense-exact", "--native-attention-exact",
                                    "--mtp-model", self.mtp, name=f"exact{index}.json")
            self.assertEqual(config["args"].count("--native-dense-exact"), 1)
            self.assertEqual(config["args"].count("--native-attention-exact"), 1)

    def test_hybrid_profile_file_validation_precedes_export(self):
        for path in (self.root / "missing.profile", self.root):
            with self.subTest(path=path), self.assertRaises(FileNotFoundError):
                self.configure("--hybrid-experts", "--expert-profile", path)
            self.assertFalse((self.root / "models").exists())
        for static_flag in ("--gpu-static-experts", "--gpu-dynamic-experts"):
            with self.subTest(static_flag=static_flag), self.assertRaises(ValueError):
                self.configure("--hybrid-experts", "--expert-profile", self.profile, static_flag)
            self.assertFalse((self.root / "models").exists())

    def test_static_requires_fixed_profile_gpu_zero_and_no_decay_before_export(self):
        cases = [("--gpu-static-experts",),
                 ("--gpu-static-experts", "--gpu-only-experts"),
                 ("--gpu-static-experts", "--gpu-only-experts", "--expert-profile", self.profile, "--device", 1),
                 ("--gpu-static-experts", "--gpu-only-experts", "--expert-profile", self.profile, "--cache-decay-interval", 4)]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "--gpu-static-experts requires"):
                self.configure(*arguments)
            self.assertFalse((self.root / "models").exists())
            self.assertFalse((self.root / "config.json").exists())
            dynamic = ("--gpu-dynamic-experts", *arguments[1:])
            with self.subTest(arguments=dynamic), self.assertRaisesRegex(ValueError, "--gpu-dynamic-experts requires"):
                self.configure(*dynamic)
            self.assertFalse((self.root / "models").exists())
            self.assertFalse((self.root / "config.json").exists())

    def test_static_profile_and_mtp_forward_only_the_selected_mode(self):
        fixed = self.configure("--gpu-only-experts", "--expert-profile", self.profile, name="profile.json")
        static = self.configure("--gpu-only-experts", "--expert-profile", self.profile,
                                "--gpu-static-experts", name="static.json")
        self.assertEqual(static["args"], [*fixed["args"], "--gpu-static-experts"])
        self.assertNotIn("--native-dense-exact", static["args"])
        self.assertNotIn("--gpu-dynamic-experts", static["args"])
        dynamic = self.configure("--gpu-only-experts", "--expert-profile", self.profile,
                                "--gpu-dynamic-experts", name="dynamic.json")
        self.assertEqual(dynamic["args"], [*fixed["args"], "--gpu-static-experts", "--gpu-dynamic-experts"])
        both = self.configure("--gpu-only-experts", "--expert-profile", self.profile,
                             "--gpu-static-experts", "--gpu-dynamic-experts", name="both.json")
        self.assertEqual(both["args"], dynamic["args"])
        mtp = self.configure("--gpu-only-experts", "--expert-profile", self.profile,
                             "--gpu-static-experts", "--mtp-model", self.mtp,
                             "--mtp-draft", 6, "--mtp-adaptive", name="static-mtp.json")
        for flag in ("--gpu-static-experts", "--native-dense-exact", "--native-attention-exact",
                     "--share-mtp-weights", "--defer-mtp-catchup", "--exact-verify", "--mtp-adaptive"):
            self.assertIn(flag, mtp["args"])

    def test_adaptive_requires_mtp_before_export(self):
        with self.assertRaisesRegex(ValueError, "requires --mtp-model"):
            self.configure("--mtp-adaptive")
        self.assertFalse((self.root / "models").exists())

    def test_adaptive_only_forwarded_when_selected(self):
        fixed = self.configure("--mtp-model", self.mtp, name="fixed.json")
        adaptive = self.configure("--mtp-model", self.mtp, "--mtp-adaptive", name="adaptive.json")
        self.assertNotIn("--mtp-adaptive", fixed["args"])
        self.assertIn("--mtp-adaptive", adaptive["args"])
        self.assertIn("--exact-verify", adaptive["args"])

    def test_existing_config_remains_unchanged(self):
        output = self.root / "config.json"
        output.write_bytes(b"existing user settings")
        for mode in ("--gpu-only-experts", "--hybrid-doorbell"):
            with self.subTest(mode=mode), self.assertRaises(FileExistsError):
                self.configure(mode, "--expert-profile", self.profile)
            self.assertEqual(output.read_bytes(), b"existing user settings")

    def test_positive_decay_requires_gpu_only_and_valid_range_before_export(self):
        for arguments in (("--cache-decay-interval", 4), ("--gpu-only-experts", "--cache-decay-interval", -1),
                          ("--gpu-only-experts", "--cache-decay-interval", 1025)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.configure(*arguments)
            self.assertFalse((self.root / "models").exists())

    def test_decay_zero_keeps_legacy_and_positive_boundaries_forward(self):
        config = self.configure("--cache-decay-interval", 0)
        self.assertNotIn("--cache-decay-interval", config["args"])
        for interval in (1, 1024):
            config = self.configure("--gpu-only-experts", "--cache-decay-interval", interval, name=f"decay{interval}.json")
            self.assertEqual(config["args"][config["args"].index("--cache-decay-interval") + 1], str(interval))


class AdaptiveLogTests(unittest.TestCase):
    def test_optional_legacy_log_has_no_adaptive_fields(self):
        self.assertEqual(adaptive_counters("A3B MTP: windows 8 offered 16 accepted 10; draft 5.0 ms catchup 1.0 ms\n"), {})

    def test_zero_draft_and_unused_arms_keep_counts_and_total_time(self):
        result = adaptive_counters("other line\nA3B MTP adaptive: K0=5/10.0ms K1=2/9.0ms K2=0/0.0ms\nother line\n")
        self.assertEqual(result["mtp_adaptive_windows_by_k"], {"0": 5, "1": 2, "2": 0})
        self.assertEqual(result["mtp_adaptive_round_ms_by_k"], {"0": 10.0, "1": 9.0, "2": 0.0})
        self.assertEqual(result["mtp_adaptive_mean_round_ms_by_k"], {"0": 2.0, "1": 4.5, "2": None})

    def test_corrupt_or_partial_adaptive_counters_fail(self):
        for body in ("", "K0=1/2.0ms K0=1/2.0ms", "K1=1/2.0ms", "K0=-1/2.0ms", "K0=1/nanms", "K9=1/2.0ms"):
            with self.subTest(body=body), self.assertRaises(ValueError):
                adaptive_counters("A3B MTP adaptive: " + body + "\n")

    def test_cache_decay_sweep_counters_optional_and_per_phase(self):
        self.assertEqual(cache_decay_counters("older engine log"), {})
        self.assertEqual(cache_decay_counters("A3B cache decay: prefill 0; decode 1337\n"),
                         {"prefill_cache_decay_sweeps": 0, "decode_cache_decay_sweeps": 1337})
        self.assertEqual(cache_decay_counters("A3B cache decay: prefill 20 decode 0\n"),
                         {"prefill_cache_decay_sweeps": 20, "decode_cache_decay_sweeps": 0})

    def test_fused_mtp_counters_optional_and_separate_from_legacy_phase_times(self):
        self.assertEqual(mtp_fused_counters("older engine log"), {})
        self.assertEqual(mtp_fused_counters("A3B MTP fused rounds: calls 11; accepted catch-up rows 24; fused 125.500 ms (excluded from draft/catchup)\n"),
                         {"mtp_fused_rounds": 11, "mtp_fused_catchup_rows": 24, "mtp_fused_round_s": .1255})
        self.assertEqual(mtp_fused_counters("A3B MTP fused rounds: calls 0; accepted catch-up rows 0; fused 0.000 ms (excluded from draft/catchup)\n")["mtp_fused_round_s"], 0)
        with self.assertRaises(ValueError):
            mtp_fused_counters("A3B MTP fused rounds: calls 1; accepted catch-up rows 2; fused nan ms (excluded from draft/catchup)\n")

    def test_prefill_share_counters_are_optional_and_preserve_zeroes(self):
        self.assertEqual(prefill_share_counters("older engine log"), {})
        self.assertEqual(prefill_share_counters("A3B prefill CPU share: calls 40; CPU jobs 10 experts 7\n"),
                         {"prefill_share_calls": 40, "prefill_share_cpu_jobs": 10, "prefill_share_cpu_experts": 7})
        self.assertEqual(prefill_share_counters("A3B prefill CPU share: calls 0; CPU jobs 0 experts 0\n"),
                         {"prefill_share_calls": 0, "prefill_share_cpu_jobs": 0, "prefill_share_cpu_experts": 0})
        with self.assertRaises(ValueError):
            prefill_share_counters("A3B prefill CPU share: calls -1; CPU jobs 0 experts 0\n")


class PatchStateTests(unittest.TestCase):
    def test_verified_patch_suffix_applies_once_and_rejects_source_edits(self):
        with tempfile.TemporaryDirectory(prefix="a3b-patch-suffix-") as directory:
            root = Path(directory)
            vendor = root / "third_party" / "llama.cpp"
            vendor.mkdir(parents=True)
            source = vendor / "example.txt"
            source.write_text("one\n", encoding="utf-8")
            patches = root / "a3b" / "patches"
            patches.mkdir(parents=True)
            (patches / "a.patch").write_text(
                "--- a/example.txt\n+++ b/example.txt\n@@ -1 +1 @@\n-one\n+two\n", encoding="utf-8")
            with patch.object(a3b, "ROOT", root):
                a3b.apply_engine_patches()
                self.assertEqual(source.read_text(encoding="utf-8"), "two\n")
                (patches / "b.patch").write_text(
                    "--- a/example.txt\n+++ b/example.txt\n@@ -1 +1 @@\n-two\n+three\n", encoding="utf-8")
                a3b.apply_engine_patches()
                a3b.apply_engine_patches()
                self.assertEqual(source.read_text(encoding="utf-8"), "three\n")
                source.write_text("unexpected edit\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "patches changed after application"):
                    a3b.apply_engine_patches()


if __name__ == "__main__":
    unittest.main(verbosity=2)
