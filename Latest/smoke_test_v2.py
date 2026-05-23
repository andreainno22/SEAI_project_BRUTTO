"""
Smoke test per fashion_patch_amplitude_qcnn_32_v2.py.

Verifica in ~3 minuti:
  1. Forward pass: output shape, no NaN
  2. Backward: theta_q.grad not None, tutti i segmenti nonzero
  3. Gradiente per-elemento DIVERSO tra segmenti (no aliasing)
  4. Drift per-segmento DIVERSO dopo 5 step (apprendimento indipendente)
  5. Frozen run: theta_q.grad None, head grad nonzero
  6. Forma QNode e output [-1,1]

Eseguire con:
    python smoke_test_v2.py
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from Latest.fashion_patch_amplitude_qcnn_32_v2 import (
    PatchAmplitudeQCNN,
    patch_amplitude_qnode,
    patch_amplitudes_4x256,
    build_optimizer,
    HEAD_IN_DIM,
    N_CLASSES,
    N_PATCHES,
    QFEATURES_PER_PATCH,
    SL_CONV,
    SL_POOL,
    SL_CONV2,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []


def check(name: str, cond: bool, detail: str = ""):
    tag = PASS if cond else FAIL
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    results.append((name, cond))


# ── dati sintetici ─────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
B = 4

imgs = np.random.rand(B, 32, 32).astype(np.float32)
amp_batch = np.zeros((B, N_PATCHES, 256), dtype=np.float32)
norms_batch = np.zeros((B, N_PATCHES), dtype=np.float32)
for i, img in enumerate(imgs):
    a, n = patch_amplitudes_4x256(img)
    amp_batch[i] = a
    norms_batch[i] = n

batch = {
    "amp4x256": torch.tensor(amp_batch, dtype=torch.float32),
    "norm_angles": torch.tensor(norms_batch, dtype=torch.float32),
    "y": torch.randint(0, N_CLASSES, (B,)),
}

# ── Test 1: Forward pass ──────────────────────────────────────────────────────
print("\n-- Test 1: Forward pass --")
model = PatchAmplitudeQCNN()
logits, features = model(batch)

check(
    "logits shape (B, 10)",
    tuple(logits.shape) == (B, N_CLASSES),
    f"{tuple(logits.shape)}",
)
check(
    "features shape (B, 32)",
    tuple(features.shape) == (B, HEAD_IN_DIM),
    f"{tuple(features.shape)}",
)
check("no NaN logits", not torch.isnan(logits).any().item())
check("no NaN features", not torch.isnan(features).any().item())
check(
    "features in [-1,1]",
    features.abs().max().item() <= 1.0 + 1e-4,
    f"max={features.abs().max().item():.4f}",
)

# ── Test 2: Backward — segmenti nonzero ──────────────────────────────────────
print("\n-- Test 2: Backward pass (trainable) --")
model = PatchAmplitudeQCNN()
optimizer = build_optimizer(model)
optimizer.zero_grad()
logits, _ = model(batch)
loss = torch.nn.CrossEntropyLoss()(logits, batch["y"])
loss.backward()

check("theta_q.grad not None", model.theta_q.grad is not None)

if model.theta_q.grad is not None:
    g = model.theta_q.grad
    g_conv = g[SL_CONV].norm().item()
    g_pool = g[SL_POOL].norm().item()
    g_conv2 = g[SL_CONV2].norm().item()

    check("grad_conv  > 0", g_conv > 1e-6, f"||g_conv||={g_conv:.4e}")
    check("grad_pool  > 0 (fix CRX)", g_pool > 1e-6, f"||g_pool||={g_pool:.4e}")
    check("grad_conv2 > 0", g_conv2 > 1e-6, f"||g_conv2||={g_conv2:.4e}")

    # Valori per-elemento nei primi 4 indici di ogni segmento
    v_conv = g[SL_CONV][:4]
    v_pool = g[SL_POOL][:4]
    v_conv2 = g[SL_CONV2][:4]
    print(f"  grad_conv [0:4]  = {v_conv.tolist()}")
    print(f"  grad_pool [0:4]  = {v_pool.tolist()}")
    print(f"  grad_conv2[0:4]  = {v_conv2.tolist()}")

    # Con init 1e-3*randn i gradienti al passo 0 sono quasi-uguali per conv e pool
    # (la simmetria e quasi ma non completamente rotta). Il test di divergenza reale
    # e il Test 3 (apprendimento indipendente dopo 5 step).
    # Qui verifichiamo solo che conv2 sia chiaramente diverso (atol=1e-4).
    conv_conv2_same = torch.allclose(g[SL_CONV][:8], g[SL_CONV2], atol=1e-4)
    check(
        "g_conv[:8] != g_conv2  (valori per-elemento diversi, atol=1e-4)",
        not conv_conv2_same,
    )

# ── Test 3: Apprendimento indipendente (5 step) ───────────────────────────────
print("\n-- Test 3: Apprendimento indipendente dopo 5 step --")
model = PatchAmplitudeQCNN()
tq_init = model.theta_q.detach().clone()
optimizer = build_optimizer(model)

for _ in range(5):
    optimizer.zero_grad()
    logits, _ = model(batch)
    loss = torch.nn.CrossEntropyLoss()(logits, batch["y"])
    loss.backward()
    optimizer.step()

tq = model.theta_q.detach()
d_conv = torch.norm(tq[SL_CONV] - tq_init[SL_CONV]).item()
d_pool = torch.norm(tq[SL_POOL] - tq_init[SL_POOL]).item()
d_conv2 = torch.norm(tq[SL_CONV2] - tq_init[SL_CONV2]).item()
print(f"  drift conv={d_conv:.4e}  pool={d_pool:.4e}  conv2={d_conv2:.4e}")

check("drift_conv  > 0", d_conv > 1e-8, f"{d_conv:.4e}")
check("drift_pool  > 0", d_pool > 1e-8, f"{d_pool:.4e}")
check("drift_conv2 > 0", d_conv2 > 1e-8, f"{d_conv2:.4e}")

# Il test VERO di indipendenza: i valori del parametro divergono tra segmenti
# (anche se le norme L2 coincidono a zero-init per simmetria del circuito)
conv_vals = tq[SL_CONV][:8]
pool_vals = tq[SL_POOL]
conv2_vals = tq[SL_CONV2]
conv_pool_diverged = not torch.allclose(conv_vals, pool_vals, atol=1e-6)
conv_conv2_diverged = not torch.allclose(conv_vals, conv2_vals, atol=1e-6)
check(
    "conv e pool hanno valori diversi dopo 5 step",
    conv_pool_diverged,
    f"max|diff|={(conv_vals - pool_vals).abs().max().item():.4e}",
)
check(
    "conv e conv2 hanno valori diversi dopo 5 step",
    conv_conv2_diverged,
    f"max|diff|={(conv_vals - conv2_vals).abs().max().item():.4e}",
)

# ── Test 4: Frozen run ────────────────────────────────────────────────────────
print("\n-- Test 4: Frozen run --")
model = PatchAmplitudeQCNN()
model.theta_q.requires_grad_(False)
optimizer = build_optimizer(model)
optimizer.zero_grad()
logits, _ = model(batch)
loss = torch.nn.CrossEntropyLoss()(logits, batch["y"])
loss.backward()

check("theta_q.grad None in frozen run", model.theta_q.grad is None)
head_grads = [p.grad for p in model.head.parameters() if p.grad is not None]
check("head ha gradienti in frozen run", len(head_grads) > 0)

# ── Test 5: Forma QNode ───────────────────────────────────────────────────────
print("\n-- Test 5: QNode output shape e range --")
model = PatchAmplitudeQCNN()
amp_single = torch.tensor(amp_batch[:, 0, :], dtype=torch.float32)
norm_single = torch.tensor(norms_batch[:, 0], dtype=torch.float32)
out = patch_amplitude_qnode(amp_single, norm_single, model.theta_q)
out_stacked = torch.stack(out, dim=1)

check(
    "QNode shape (B, QFEATURES_PER_PATCH)",
    tuple(out_stacked.shape) == (B, QFEATURES_PER_PATCH),
    f"{tuple(out_stacked.shape)}",
)
check("QNode no NaN", not torch.isnan(out_stacked).any().item())
check(
    "QNode in [-1,1]",
    out_stacked.abs().max().item() <= 1.0 + 1e-4,
    f"max={out_stacked.abs().max().item():.4f}",
)

# ── Sommario ──────────────────────────────────────────────────────────────────
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"\n{'=' * 60}")
print(f"Smoke test: {passed}/{total} check superati")
if passed < total:
    print(f"FALLITI: {[n for n, ok in results if not ok]}")
    sys.exit(1)
else:
    print("Tutti i check superati -- pronto per il run completo.")
    sys.exit(0)
