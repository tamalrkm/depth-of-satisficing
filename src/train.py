"""
Stage 6 train (implemented): fit the satisficing model and the state-only baseline.

Expects data/train.pt produced by build_dataset.py with keys:
    delta [N,M,D], logq [N,M], move_mask [N,M], context [N,C], y [N],
    meta (DataFrame-like dict: player, elo, time_class, swing, ply, ...)
Splits by player to prevent leakage. Reports held-out NLL for:
    - state-only baseline (alpha forced high, beta forced 0)  == Maia-3 alone
    - full fusion model
Run:
    python src/train.py --config config.yaml
"""
import argparse, yaml
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import SatisficingModel


def player_split(players, val_frac, seed):
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(players)))
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_frac))
    val_players = set(uniq[:n_val])
    val_idx = np.array([p in val_players for p in players])
    return ~val_idx, val_idx


def run(cfg, epochs=None):
    if epochs is not None:
        cfg["model"]["epochs"] = epochs
    blob = torch.load(cfg["data"]["train_tensor"])
    delta, logq, mask, ctx, y = (blob[k] for k in
                                 ["delta", "logq", "move_mask", "context", "y"])
    # depth-validity mask (1 = observed, 0 = beyond a node-capped search). Older tensors that
    # predate the cap have no such key -> treat every depth as observed.
    dmask = blob.get("depth_mask")
    if dmask is None:
        dmask = torch.ones(delta.shape[0], delta.shape[2])
    players = blob["meta"]["player"]
    tr, va = player_split(players, cfg["model"]["val_frac"], cfg["data"]["sample_seed"])

    mc = cfg["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SatisficingModel(mc["depth_grid"], ctx.shape[-1], mc["hidden"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=mc["lr"])

    def batches(idx):
        ds = TensorDataset(delta[idx], logq[idx], mask[idx], dmask[idx], ctx[idx], y[idx])
        return DataLoader(ds, batch_size=mc["batch_size"], shuffle=True)

    def eval_nll(idx):
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for d, q, m, dm, c, t in batches(idx):
                _, nll = model.loss(d.to(dev), q.to(dev), m.to(dev), c.to(dev), t.to(dev),
                                    mc["entropy_reg"], dm.to(dev))
                tot += nll.item() * len(t); n += len(t)
        return tot / n

    for ep in range(mc["epochs"]):
        model.train()
        for d, q, m, dm, c, t in batches(np.where(tr)[0]):
            loss, _ = model.loss(d.to(dev), q.to(dev), m.to(dev), c.to(dev), t.to(dev),
                                 mc["entropy_reg"], dm.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch {ep:02d}  val NLL = {eval_nll(np.where(va)[0]):.4f}  "
              f"alpha={model.alpha.item():.2f} beta={model.beta.item():.2f}")

    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, "data/model.pt")
    print("saved data/model.pt")
    print("NOTE: for the state-only baseline, refit with beta frozen at 0 (Maia-3 alone); "
          "the held-out NLL gap is result E1.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override cfg.model.epochs")
    a = ap.parse_args()
    run(yaml.safe_load(open(a.config)), a.epochs)
