"""
Stage 6 model (implemented): the satisficing predictor.

Player = latent distribution over effective search depths (pi). At a given depth the move
is chosen by a soft-max over depth-d regret fused with the Maia policy as a product of
experts. The move likelihood marginalises depth. Depth of satisficing = posterior over
depth given the played move.

Tensors (padded per batch, M = max #moves, D = #depths on the grid):
    delta     [B, M, D]  regret in win-prob units (best_winprob_d - winprob_id), >=0
    logq      [B, M]     log Maia-3 policy
    move_mask [B, M]     1 for real moves, 0 for padding
    context   [B, C]     elo, time-class one-hot, clock, ply, total-swing, ...
    y         [B]        index of the played move
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e9


class SatisficingModel(nn.Module):
    def __init__(self, depth_grid, context_dim, hidden=128):
        super().__init__()
        self.register_buffer("depth_vals", torch.tensor(depth_grid, dtype=torch.float))
        self.D = len(depth_grid)
        self.pi_net = nn.Sequential(
            nn.Linear(context_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.D),
        )
        self._alpha = nn.Parameter(torch.tensor(0.0))  # pattern weight (softplus)
        self._beta = nn.Parameter(torch.tensor(1.0))   # regret precision (softplus)

    @property
    def alpha(self): return F.softplus(self._alpha)

    @property
    def beta(self): return F.softplus(self._beta)

    def forward(self, delta, logq, move_mask, context, depth_mask=None):
        B, M, D = delta.shape
        pi = F.softmax(self.pi_net(context), dim=-1)              # [B, D]
        if depth_mask is not None:
            # Node-capped positions have no engine regret past their reached depth. Drop those
            # depths from the marginalisation and renormalise pi over the observed ones, rather
            # than forward-filling (which would assert "regret stops changing" -- it wasn't).
            pi = pi * depth_mask
            pi = pi / pi.sum(-1, keepdim=True).clamp_min(1e-9)

        # per-depth choice logits: -beta*delta + alpha*logq, masked over moves
        logits = -self.beta * delta + self.alpha * logq.unsqueeze(-1)   # [B, M, D]
        mask = move_mask.unsqueeze(-1).bool()
        logits = logits.masked_fill(~mask, NEG)
        p_i_given_d = F.softmax(logits, dim=1)                   # over moves -> [B, M, D]

        p_i = (pi.unsqueeze(1) * p_i_given_d).sum(-1)            # [B, M]
        return p_i, pi, p_i_given_d

    def loss(self, delta, logq, move_mask, context, y, entropy_reg=0.01, depth_mask=None):
        p_i, pi, _ = self.forward(delta, logq, move_mask, context, depth_mask)
        p_y = p_i.gather(1, y.unsqueeze(1)).squeeze(1).clamp_min(1e-9)
        nll = -torch.log(p_y).mean()
        ent_pen = (pi * (pi.clamp_min(1e-9)).log()).sum(-1).mean()  # = -entropy; minimise -> spread pi
        return nll + entropy_reg * ent_pen, nll.detach()

    @torch.no_grad()
    def depth_of_satisficing(self, delta, logq, move_mask, context, y, depth_mask=None):
        """Posterior over effective depth given the played move, and its expectation."""
        _, pi, p_i_given_d = self.forward(delta, logq, move_mask, context, depth_mask)
        p_y_given_d = p_i_given_d.gather(
            1, y.view(-1, 1, 1).expand(-1, 1, self.D)).squeeze(1)   # [B, D]
        r = pi * p_y_given_d
        r = r / r.sum(-1, keepdim=True).clamp_min(1e-9)             # posterior r_d
        dhat = (r * self.depth_vals).sum(-1)                       # [B]
        return dhat, r
