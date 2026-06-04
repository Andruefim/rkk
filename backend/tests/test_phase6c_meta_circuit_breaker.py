"""Phase 6c gate smoke: MetaCircuitBreaker state machine + scorecard schema."""
from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

from engine.meta_causal import WMetaEnsemble
from engine.meta_circuit_breaker import MetaCircuitBreaker, meta_cb_enabled


class MetaCircuitBreakerTests(unittest.TestCase):
    def test_closed_open_half_open_closed(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "RKK_META_CB_ENABLED": "1",
                "RKK_META_CB_PE_OPEN": "0.25",
                "RKK_META_CB_PE_CLOSE": "0.12",
                "RKK_META_CB_RESET_AFTER": "10",
            },
        ):
            self.assertTrue(meta_cb_enabled())
            cb = MetaCircuitBreaker()
            self.assertEqual(cb.state, cb.CLOSED)
            self.assertTrue(cb.wmeta_active)

            for t in range(8):
                cb.observe(0.5, meta_age=0, tick=t)
            self.assertEqual(cb.state, cb.OPEN)
            self.assertFalse(cb.wmeta_active)

            for t in range(12):
                cb.observe(0.15, meta_age=5000, tick=100 + t)
            self.assertEqual(cb.state, cb.HALF_OPEN)

            for _ in range(30):
                cb.observe(0.05, meta_age=0, tick=200)
            self.assertEqual(cb.state, cb.CLOSED)
            self.assertTrue(cb.wmeta_active)

    def test_pe_open_wmeta_inactive(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"RKK_META_CB_ENABLED": "1", "RKK_META_CB_PE_OPEN": "0.20"},
        ):
            cb = MetaCircuitBreaker()
            for t in range(15):
                cb.observe(0.9, meta_age=0, tick=t)
            self.assertEqual(cb.state, cb.OPEN)
            self.assertFalse(cb.wmeta_active)

    def test_recovery_ticks_logged(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"RKK_META_CB_ENABLED": "1", "RKK_META_CB_RESET_AFTER": "5"},
        ):
            cb = MetaCircuitBreaker()
            cb.force_open(tick=100)
            rt = cb.recovery_ticks(150)
            self.assertIsNotNone(rt)
            self.assertGreaterEqual(int(rt), 50)

    def test_half_open_resets_w_meta(self) -> None:
        with mock.patch.dict(os.environ, {"RKK_META_CB_ENABLED": "1"}):
            cb = MetaCircuitBreaker()
            w = WMetaEnsemble(torch.device("cpu"))
            cb.force_half_open(tick=10)
            self.assertTrue(cb.reset_w_meta_if_needed(w))

    def test_meta_age_open(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"RKK_META_CB_ENABLED": "1", "RKK_META_CB_AGE_OPEN": "50"},
        ):
            cb = MetaCircuitBreaker()
            cb.observe(0.01, meta_age=100, tick=1)
            self.assertEqual(cb.state, cb.OPEN)
