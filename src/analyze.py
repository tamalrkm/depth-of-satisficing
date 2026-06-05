"""
Stage 7 (scaffold): the results (E1-E6).

E1  held-out move prediction: fusion vs state-only (Maia-3) NLL + top-1/3,
    stratified by time_class x swing_class -> the pre-registered interaction (Fig 4).
E2  depth of satisficing vs rating, with bootstrap/Bayesian CIs (Fig 2a);
    within-player collapse under time pressure using clock bins (Fig 2b).
E3  convergent validity: corr(model dhat, observed time_spent), held out, by phase (Fig 3).
E4  profiles: per-player [mean dhat, trap-susceptibility, deep-discovery, time-elasticity];
    ridge regression elo recovery with nested CV; champion decomposition (Fig 5).
E5  synthetic recovery: estimated vs planted depth (Fig 6).
E6  (optional) Go transfer.

TODO: load data/model.pt, compute model.depth_of_satisficing over the
held-out set, then produce each figure + the stats table.
"""
import argparse, yaml

def main(cfg):
    raise NotImplementedError("implement E1-E6")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml")
    main(yaml.safe_load(open(ap.parse_args().config)))
