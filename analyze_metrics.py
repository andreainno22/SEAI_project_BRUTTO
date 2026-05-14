import pandas as pd
import numpy as np

df = pd.read_csv('fashion_8qubit_16x16/ALL_metrics.csv')
print('SHAPE:', df.shape)
print('VARIANTI:', df.variant.unique(), '  SEEDS:', df.seed.unique(), '  EPOCHS:', sorted(df.epoch.unique()))

# -----------------------------------------------------------
print('\n\n=== feat_std_mean_test: andamento per variante/seed ===')
for variant in ['E1','E2','E3','E4']:
    print(f'\n{variant}:')
    for seed in [0,1,2]:
        sub = df[(df.variant==variant)&(df.seed==seed)].sort_values('epoch')
        v = sub['feat_std_mean_test'].values
        print(f'  seed{seed}: ep1={v[0]:.4f}  ep8={v[7]:.4f}  ep15={v[-1]:.4f}  min={v.min():.4f}  max={v.max():.4f}  delta={v[-1]-v[0]:+.4f}')
        print(f'           valori: {[round(x,4) for x in v]}')

# -----------------------------------------------------------
print('\n\n=== val_acc e val_loss per epoch: picco e final (seed0) ===')
for variant in ['E1','E2','E3','E4']:
    sub = df[(df.variant==variant)&(df.seed==0)].sort_values('epoch')
    best_val_ep  = sub.loc[sub['val_acc'].idxmax(),  'epoch']
    best_val_acc = sub['val_acc'].max()
    best_loss_ep = sub.loc[sub['val_loss'].idxmin(), 'epoch']
    best_loss    = sub['val_loss'].min()
    final        = sub.iloc[-1]
    print(f'{variant} s0: best_val_acc={best_val_acc:.3f}@ep{best_val_ep}  best_val_loss={best_loss:.3f}@ep{best_loss_ep}  '
          f'final_val_acc={final.val_acc:.3f}  final_test_acc={final.test_acc:.3f}  final_val_loss={final.val_loss:.3f}')

# -----------------------------------------------------------
print('\n\n=== overfitting: train_acc vs val_acc vs test_acc (seed0, tutti epoch) ===')
for variant in ['E1','E2','E3','E4']:
    sub = df[(df.variant==variant)&(df.seed==0)].sort_values('epoch')
    print(f'\n{variant} seed0:')
    print(f'  {"ep":>3}  {"tr_acc":>7}  {"va_acc":>7}  {"te_acc":>7}  {"tr_loss":>8}  {"va_loss":>8}  {"acc_gap":>8}  {"loss_gap":>9}')
    for _, r in sub.iterrows():
        print(f'  {int(r.epoch):3d}  {r.train_acc_diag:7.3f}  {r.val_acc:7.3f}  {r.test_acc:7.3f}  '
              f'{r.train_loss_diag:8.4f}  {r.val_loss:8.4f}  {r.train_acc_diag-r.val_acc:+8.3f}  {r.val_loss-r.train_loss_diag:+9.4f}')

# -----------------------------------------------------------
print('\n\n=== gradienti (seed0) ===')
for variant in ['E1','E2','E3','E4']:
    sub = df[(df.variant==variant)&(df.seed==0)].sort_values('epoch')
    print(f'\n{variant} seed0:')
    for col in ['grad_e1_embed','grad_e4_embed','grad_qkernel','grad_head']:
        v = sub[col].values
        nonzero = v[v>0]
        min_nz = f'{nonzero.min():.6f}' if len(nonzero) > 0 else 'ALL_ZERO'
        print(f'  {col:18s}: mean={v.mean():.4f}  std={v.std():.4f}  zeros={int((v==0).sum())}/15  '
              f'min_nz={min_nz}  max={v.max():.4f}')

# -----------------------------------------------------------
print('\n\n=== pred_hist_test (distribuzione predizioni, seed0, ep1 vs ep15) ===')
import ast
for variant in ['E1','E2','E3','E4']:
    sub = df[(df.variant==variant)&(df.seed==0)].sort_values('epoch')
    h1  = ast.literal_eval(sub.iloc[0]['pred_hist_test'])
    h15 = ast.literal_eval(sub.iloc[-1]['pred_hist_test'])
    print(f'{variant} ep1 : {h1}  sum={sum(h1)}')
    print(f'{variant} ep15: {h15}  sum={sum(h15)}')
    print()
