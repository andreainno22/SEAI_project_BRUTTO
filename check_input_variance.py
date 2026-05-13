"""
Verifica se feat_std_mean_test bassa è dovuta al barren plateau
o alla bassa varianza dei dati di input al circuito quantistico.

Analizza la distribuzione di quad_means (input a E1/E2/E4) e quads64 (input a E3)
su un campione del test set di Fashion-MNIST 16x16.
"""
import numpy as np
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])
test_ds = torchvision.datasets.FashionMNIST("./data", train=False, download=True, transform=transform)

N = 500
rng = np.random.default_rng(0)
idxs = rng.choice(len(test_ds), N, replace=False)

QUAD_META_PATCH_IDXS = [
    [[0,1],[2,3],[8,9],[10,11],[16,17],[18,19],[24,25],[26,27]],
    [[4,5],[6,7],[12,13],[14,15],[20,21],[22,23],[28,29],[30,31]],
    [[32,33],[34,35],[40,41],[42,43],[48,49],[50,51],[56,57],[58,59]],
    [[36,37],[38,39],[44,45],[46,47],[52,53],[54,55],[60,61],[62,63]],
]

def extract(idx):
    x, _ = test_ds[idx]
    X = x.squeeze(0).numpy()  # (16,16)
    patches = []
    for r in range(8):
        for c in range(8):
            patches.append(np.array([X[2*r,2*c], X[2*r,2*c+1], X[2*r+1,2*c], X[2*r+1,2*c+1]]))
    patches = np.stack(patches)           # (64, 4)
    means64 = patches.mean(axis=1)        # (64,)
    quad_means = np.array([
        [(means64[p[0]] + means64[p[1]]) * 0.5 for p in QUAD_META_PATCH_IDXS[q]]
        for q in range(4)
    ], dtype=np.float32)                  # (4, 8)
    quads64 = np.stack([
        X[0:8, 0:8].reshape(-1),
        X[0:8, 8:16].reshape(-1),
        X[8:16, 0:8].reshape(-1),
        X[8:16, 8:16].reshape(-1),
    ])                                    # (4, 64)
    return quad_means, quads64

all_qm  = []  # quad_means: input RY(pi*x) per E1/E2/E4
all_q64 = []  # quads64:    input AmplitudeEmbedding per E3

for i in idxs:
    qm, q64 = extract(int(i))
    all_qm.append(qm)
    all_q64.append(q64)

all_qm  = np.stack(all_qm)   # (500, 4, 8)
all_q64 = np.stack(all_q64)  # (500, 4, 64)

# --- Statistiche quad_means (input E1/E2/E4) ---
flat_qm = all_qm.reshape(N, -1)   # (500, 32) — stesso layout di feat_vec
print("=" * 60)
print("INPUT AL CIRCUITO: quad_means  (forma 500x32)")
print("=" * 60)
print(f"  Media globale:        {flat_qm.mean():.4f}")
print(f"  Std globale:          {flat_qm.std():.4f}")
print(f"  Min / Max:            {flat_qm.min():.4f} / {flat_qm.max():.4f}")
print()

# std per feature (asse campioni) — stesso calcolo di feat_std nel codice
per_feature_std_input = flat_qm.std(axis=0)  # (32,)
print(f"  std per-feature (media):  {per_feature_std_input.mean():.4f}")
print(f"  std per-feature (min):    {per_feature_std_input.min():.4f}")
print(f"  std per-feature (max):    {per_feature_std_input.max():.4f}")
print()

# Angolo dopo encoding RY(pi * x): input effettivo al gate
angles = np.pi * flat_qm  # (500, 32)
print("ANGOLI RY(pi*x) effettivi:")
print(f"  Media:  {angles.mean():.4f} rad  ({np.degrees(angles.mean()):.1f}°)")
print(f"  Std:    {angles.std():.4f} rad  ({np.degrees(angles.std()):.1f}°)")
print(f"  Range:  [{angles.min():.4f}, {angles.max():.4f}] rad")
print()

# --- Statistiche quads64 (input E3 AmplitudeEmbedding) ---
flat_q64 = all_q64.reshape(N, -1)  # (500, 256)
print("=" * 60)
print("INPUT AL CIRCUITO: quads64  (forma 500x256, per E3)")
print("=" * 60)
print(f"  std per-feature (media):  {flat_q64.std(axis=0).mean():.4f}")
print(f"  std per-feature (min):    {flat_q64.std(axis=0).min():.4f}")
print(f"  std per-feature (max):    {flat_q64.std(axis=0).max():.4f}")
print()

# --- Confronto con feat_std osservata ---
print("=" * 60)
print("CONFRONTO")
print("=" * 60)
print(f"  std input quad_means (pre-circuito):   {per_feature_std_input.mean():.4f}")
print(f"  feat_std_mean_test   (post-circuito):  ~0.0620")
print()
ratio = per_feature_std_input.mean() / 0.0620
if per_feature_std_input.mean() < 0.10:
    print("  >> INPUT a bassa varianza: la poca diversita' nei dati")
    print("     contribuisce alla feat_std bassa indipendentemente dal BP.")
elif ratio > 3:
    print(f"  >> Input std ({per_feature_std_input.mean():.3f}) >> feat_std (0.062): il circuito")
    print("     COMPRIME la varianza. Segnale forte di barren plateau.")
else:
    print(f"  >> Input std ({per_feature_std_input.mean():.3f}) ~ feat_std (0.062):")
    print("     la bassa varianza e' ereditata dai dati, non solo dal BP.")

# --- Distribuzione per decile ---
print()
print("Distribuzione quad_means (decili):")
percentiles = np.percentile(flat_qm, [0,10,25,50,75,90,100])
labels      = ["min","10%","25%","50%","75%","90%","max"]
for l, v in zip(labels, percentiles):
    print(f"  {l:>4s}: {v:.4f}")
