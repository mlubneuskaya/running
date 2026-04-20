#!/usr/bin/env python3
"""Smoke tests for the TCN gait phase detection model.

No real data required — all tests use synthetic VideoRecords.

Run locally
-----------
    python tests/smoke_tcn.py          # standalone, prints PASS/FAIL per test
    pytest tests/smoke_tcn.py -v       # with pytest

Tests
-----
1.  Output shape equals input shape for any (B, T, F)
2.  log-softmax probabilities sum to 1
3.  Even kernel_size raises AssertionError
4.  Receptive field matches formula (default config and parametrised)
5.  All parameters receive non-NaN gradients after one backward pass
6.  3-epoch training loop completes with finite loss and saves checkpoint
7.  Checkpoint round-trip: reloaded model produces identical output
8.  Variable-length sequences are padded correctly by collate_fn
9.  Feature subset (n_features < 22) works end-to-end through the dataset
10. Early stopping fires before max_epochs when val loss stops improving
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.gait.detection.model import TCN
from src.gait.detection.train import Trainer, TrainerConfig
from src.gait.gait_data.dataset import (
    VideoRecord,
    GaitWindowDataset,
    GaitSequenceDataset,
    compute_class_weights,
)


# ── synthetic data ────────────────────────────────────────────────────────────

def _make_record(T: int = 300, n_features: int = 22, seed: int = 0) -> VideoRecord:
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((T, n_features)).astype(np.float32)
    phase = np.linspace(0, 6 * np.pi, T)
    labels = np.where(np.sin(phase) > 0.4, 0,
             np.where(np.sin(phase) < -0.4, 1, 2)).astype(np.int64)
    return VideoRecord(video_path="synthetic", athlete=f"athlete_{seed}", features=features, labels=labels)


def _records(n: int = 6, T: int = 250) -> list[VideoRecord]:
    return [_make_record(T=T, seed=i) for i in range(n)]


# ── helpers ───────────────────────────────────────────────────────────────────

def _small_model(**kwargs) -> TCN:
    """TCN with minimal capacity for fast tests."""
    return TCN(n_features=kwargs.pop("n_features", 22), n_blocks=2,
               n_filters=16, kernel_size=3, **kwargs)


def _loaders(records, window_size=50, feature_idx=None):
    train_ds = GaitWindowDataset(records[:-1], window_size=window_size, feature_idx=feature_idx)
    val_ds   = GaitSequenceDataset(records[-1:],                          feature_idx=feature_idx)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              collate_fn=GaitSequenceDataset.collate, num_workers=0)
    return train_loader, val_loader


# ── tests ─────────────────────────────────────────────────────────────────────

def test_output_shape():
    model = TCN()
    for B, T in [(1, 1), (2, 50), (4, 300)]:
        out = model(torch.randn(B, T, 22))
        assert out.shape == (B, T, 3), f"shape mismatch for B={B} T={T}: {out.shape}"


def test_log_probs_sum_to_one():
    model = TCN()
    out = model(torch.randn(3, 120, 22))
    sums = out.exp().sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        f"probabilities do not sum to 1 (max dev {(sums - 1).abs().max():.2e})"


def test_even_kernel_raises():
    raised = False
    try:
        TCN(kernel_size=4)
    except AssertionError:
        raised = True
    assert raised, "TCN(kernel_size=4) should raise AssertionError"


def test_receptive_field_default():
    # 4 blocks, kernel=3, dilations 1+2+4+8=15
    # RF = 1 + 2*(3-1)*15 = 61
    assert TCN().receptive_field == 61


def test_receptive_field_formula():
    for n_blocks, k in [(3, 3), (4, 5), (5, 7)]:
        model = TCN(n_blocks=n_blocks, kernel_size=k)
        expected = 1 + sum(2 * (k - 1) * (2 ** i) for i in range(n_blocks))
        assert model.receptive_field == expected, \
            f"n_blocks={n_blocks} k={k}: RF={model.receptive_field}, expected={expected}"


def test_gradients_reach_all_params():
    model = TCN()
    x = torch.randn(2, 80, 22)
    labels = torch.randint(0, 3, (2, 80))
    loss = torch.nn.NLLLoss()(model(x).reshape(-1, 3), labels.reshape(-1))
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"


def test_training_loop_three_epochs():
    recs = _records(n=5)
    model = _small_model()
    weights = compute_class_weights(recs)
    train_loader, val_loader = _loaders(recs, window_size=50)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt = f.name

    cfg = TrainerConfig(
        max_epochs=3, batch_size=2, lr=1e-3,
        early_stopping_patience=10, window_size=50, checkpoint_path=ckpt,
    )
    result = Trainer(model, weights, cfg).fit(train_loader, val_loader)

    assert np.isfinite(result["best_val_loss"]), "val loss is not finite"
    assert result["epochs_trained"] == 3, \
        f"expected 3 epochs, got {result['epochs_trained']}"
    assert os.path.exists(ckpt), "checkpoint file not created"
    os.unlink(ckpt)


def test_checkpoint_roundtrip():
    model = _small_model()
    x = torch.randn(1, 100, 22)
    model.eval()
    with torch.no_grad():
        out_before = model(x).clone()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt = f.name
    torch.save(model.state_dict(), ckpt)

    model2 = _small_model()
    model2.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model2.eval()
    with torch.no_grad():
        out_after = model2(x)

    assert torch.allclose(out_before, out_after, atol=1e-6), \
        "reloaded model output differs from saved model"
    os.unlink(ckpt)


def test_variable_length_collation():
    lengths = [80, 150, 220]
    records = [_make_record(T=t, seed=i) for i, t in enumerate(lengths)]
    ds = GaitSequenceDataset(records)
    loader = DataLoader(ds, batch_size=3, collate_fn=GaitSequenceDataset.collate)
    feats, labels, masks = next(iter(loader))

    assert feats.shape == (3, 220, 22), f"unexpected padded shape: {feats.shape}"
    assert masks[0, :80].all()  and not masks[0, 80:].any(),  "mask wrong for seq 0"
    assert masks[1, :150].all() and not masks[1, 150:].any(), "mask wrong for seq 1"
    assert masks[2].all(),                                      "mask wrong for seq 2"

    out = TCN()(feats)
    assert out.shape == (3, 220, 3)


def test_feature_subset_end_to_end():
    feature_idx = list(range(6))   # positions only
    recs = _records(n=5)
    model = _small_model(n_features=6)
    weights = compute_class_weights(recs)
    train_loader, val_loader = _loaders(recs, window_size=50, feature_idx=feature_idx)

    cfg = TrainerConfig(
        max_epochs=2, batch_size=2, lr=1e-3,
        early_stopping_patience=10, window_size=50,
    )
    result = Trainer(model, weights, cfg).fit(train_loader, val_loader)
    assert np.isfinite(result["best_val_loss"])


def test_early_stopping_fires():
    """Early stopping should trigger well before max_epochs when val loss plateaus."""
    recs = _records(n=4, T=200)
    model = _small_model()
    weights = compute_class_weights(recs)
    train_loader, val_loader = _loaders(recs, window_size=50)

    cfg = TrainerConfig(
        max_epochs=100, batch_size=2, lr=1e-3,
        early_stopping_patience=3, window_size=50,
    )
    result = Trainer(model, weights, cfg).fit(train_loader, val_loader)
    assert result["epochs_trained"] < 100, \
        f"early stopping did not fire (ran all 100 epochs)"


# ── standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    suite = [
        test_output_shape,
        test_log_probs_sum_to_one,
        test_even_kernel_raises,
        test_receptive_field_default,
        test_receptive_field_formula,
        test_gradients_reach_all_params,
        test_training_loop_three_epochs,
        test_checkpoint_roundtrip,
        test_variable_length_collation,
        test_feature_subset_end_to_end,
        test_early_stopping_fires,
    ]

    passed, failed = 0, 0
    for fn in suite:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(0 if failed == 0 else 1)
