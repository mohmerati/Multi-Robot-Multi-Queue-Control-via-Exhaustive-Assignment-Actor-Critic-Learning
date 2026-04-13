import torch
import torch.nn as nn
import torch.nn.functional as F

class RobotQueueAttentionActor(nn.Module):
    """
    Assignment-only actor.

    Busy robots are FORCED to stay.
    Idle robots receive learned logits over queues.
    Distinct-destination masking across robots is still handled by the
    existing SB3 autoregressive sampler/evaluator in QueuePPOPolicy.

    No queue-id embedding. Shared queue scorer restores permutation symmetry.
    """
    def __init__(
        self,
        N,
        M,
        queue_cost_weights,
        d_model=128,
        d_robot_emb=16,
        max_q_cap=100.0,
        rate_scale=1.0,
        weight_scale=1.0,
    ):
        super().__init__()
        self.N = N
        self.M = M
        self.max_q_cap = float(max_q_cap)
        self.rate_scale = float(rate_scale)
        self.weight_scale = float(weight_scale)
        self.score_temp = float(d_model) ** 0.5

        qcw = torch.as_tensor(queue_cost_weights, dtype=torch.float32)
        self.register_buffer("queue_cost_weights", qcw)

        self.robot_emb = nn.Embedding(num_embeddings=N, embedding_dim=d_robot_emb)

        # queue features:
        # q, lambda, w, q*w, occupied_now, free_now
        self.queue_enc = nn.Sequential(
            nn.Linear(6, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        # robot features:
        # robot-location embedding + local q, lambda, w, busy
        self.robot_enc = nn.Sequential(
            nn.Linear(d_robot_emb + 4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        self.queue_bias = nn.Linear(d_model, 1, bias=False)
        self.idle_self_bias = nn.Parameter(torch.zeros(1))

    def forward(self, obs, rates_b):
        queues = obs["queues"].float()     # [B,N]
        robots = obs["robots"].long()      # [B,M]
        B, N, M = queues.shape[0], self.N, self.M
        dev = queues.device

        w = self.queue_cost_weights.view(1, N).expand(B, N).to(dev)

        q_norm = queues / max(self.max_q_cap, 1.0)
        lam_norm = rates_b.float() / max(self.rate_scale, 1e-8)
        w_norm = w / max(self.weight_scale, 1e-8)
        qw_norm = q_norm * w_norm

        occ = torch.zeros(B, N, device=dev)
        occ.scatter_add_(1, robots, torch.ones(B, M, device=dev))
        occ_bool = occ > 0
        occ_float = occ_bool.float()

        q_feats = torch.stack(
            [q_norm, lam_norm, w_norm, qw_norm, occ_float, 1.0 - occ_float],
            dim=-1
        )                                          # [B,N,6]
        q_tokens = self.queue_enc(q_feats)         # [B,N,d]
        q_bias = self.queue_bias(q_tokens).squeeze(-1)  # [B,N]

        q_here = q_norm.gather(1, robots)          # [B,M]
        lam_here = lam_norm.gather(1, robots)      # [B,M]
        w_here = w_norm.gather(1, robots)          # [B,M]
        busy = (queues.gather(1, robots) > 0)      # [B,M] bool

        r_emb = self.robot_emb(robots)             # [B,M,d_robot_emb]
        r_feats = torch.cat(
            [r_emb, q_here.unsqueeze(-1), lam_here.unsqueeze(-1), w_here.unsqueeze(-1), busy.float().unsqueeze(-1)],
            dim=-1
        )                                          # [B,M,d_robot_emb+4]
        r_tokens = self.robot_enc(r_feats)         # [B,M,d]

        # Pairwise robot-queue scores
        scores = torch.einsum("bmd,bnd->bmn", r_tokens, q_tokens) / self.score_temp
        scores = scores + q_bias[:, None, :]

        self_loc = F.one_hot(robots, num_classes=N).bool()                 # [B,M,N]

        # Current-occupancy feasibility:
        # queue must be currently free, unless robot stays at its own location
        allow_now = (~occ_bool).unsqueeze(1).expand(B, M, N) | self_loc
        scores = scores.masked_fill(~allow_now, -1e9)

        # Small stay bias only for idle robots
        scores = scores + self.idle_self_bias * self_loc.float()

        # FORCE busy robots to stay (exhaustive structure)
        forced_logits = torch.full_like(scores, -1e9)
        forced_logits.scatter_(2, robots.unsqueeze(-1), 0.0)
        scores = torch.where(busy.unsqueeze(-1), forced_logits, scores)

        return scores