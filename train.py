"""Train/evaluate CBMA-TN-VQC and run the toy electrolyzer dispatch. Example: python train.py --epochs 2"""
import argparse, numpy as np, torch
from src.train_common import get_data, fit, evaluate
from src.cbma_tn_vqc import CBMA_TN_VQC, electrolyzer_h2_kgph
from src.data import power_curve
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/synthetic/synthetic_scada.csv", help="SCADA-style CSV; pass \"\" to generate synthetic data in memory"); ap.add_argument("--n_synth", type=int, default=1500); ap.add_argument("--lookback", type=int, default=24)
ap.add_argument("--horizon", type=int, default=1); ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--qubits", type=int, default=4); ap.add_argument("--layers", type=int, default=2); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--quick", action="store_true", help="fast smoke run (<1 min): last 500 rows, few epochs")
a = ap.parse_args()
if a.quick: a.epochs = 1
a.max_rows = 500 if a.quick else None
torch.manual_seed(a.seed)
tr, va, te, sc = get_data(a.csv, a.n_synth, a.lookback, a.horizon, a.max_rows)
m = CBMA_TN_VQC(tr[0].shape[2], a.lookback, n_qubits=a.qubits, n_layers=a.layers)
fit(m, tr, va, a.epochs, lr=2e-3); evaluate(m, te, sc, name="cbma_tn_vqc")
with torch.no_grad(): ws = m(torch.tensor(te[0])).numpy() * sc[1] + sc[0]
h2 = electrolyzer_h2_kgph(power_curve(ws, rated=10000.0))
print(f"toy electrolyzer (forecast-informed setpoint): mean H2 = {h2.mean():.1f} kg/h over {len(h2)} steps")
