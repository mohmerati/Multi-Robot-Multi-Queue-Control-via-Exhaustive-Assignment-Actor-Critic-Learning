import torch
import torch.nn as nn

class CTDECritic(nn.Module):
    """
    Simple centralized pooled critic.
    Permutation-friendly over queues and robots.
    No queue-id embeddings, no urgency subnet.
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

        qcw = torch.as_tensor(queue_cost_weights, dtype=torch.float32)
        self.register_buffer("queue_cost_weights", qcw)

        self.robot_emb = nn.Embedding(num_embeddings=N, embedding_dim=d_robot_emb)

        # queue features:
        # q, lambda, w, q*w, occupied_now
        self.queue_enc = nn.Sequential(
            nn.Linear(5, d_model),
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

        # pooled critic head
        self.value_head = nn.Sequential(
            nn.Linear(2 * d_model + 4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, obs, rates):
        queues = obs["queues"].float()   # [B,N]
        robots = obs["robots"].long()    # [B,M]
        B, N, M = queues.shape[0], self.N, self.M
        dev = queues.device

        if rates.dim() == 1:
            rates_b = rates.view(1, N).expand(B, N)
        else:
            rates_b = rates

        w = self.queue_cost_weights.view(1, N).expand(B, N).to(dev)

        q_norm = queues / max(self.max_q_cap, 1.0)
        lam_norm = rates_b / max(self.rate_scale, 1e-8)
        w_norm = w / max(self.weight_scale, 1e-8)
        qw_norm = q_norm * w_norm

        occ = torch.zeros(B, N, device=dev)
        occ.scatter_add_(1, robots, torch.ones(B, M, device=dev))
        occ = (occ > 0).float()

        q_feats = torch.stack([q_norm, lam_norm, w_norm, qw_norm, occ], dim=-1)  # [B,N,5]
        q_tokens = self.queue_enc(q_feats)                                         # [B,N,d]

        q_here = q_norm.gather(1, robots)      # [B,M]
        lam_here = lam_norm.gather(1, robots)  # [B,M]
        w_here = w_norm.gather(1, robots)      # [B,M]
        busy = (queues.gather(1, robots) > 0).float()

        r_emb = self.robot_emb(robots)         # [B,M,d_robot_emb]
        r_feats = torch.cat(
            [r_emb, q_here.unsqueeze(-1), lam_here.unsqueeze(-1), w_here.unsqueeze(-1), busy.unsqueeze(-1)],
            dim=-1
        )                                      # [B,M,d_robot_emb+4]
        r_tokens = self.robot_enc(r_feats)     # [B,M,d]

        q_pool = q_tokens.mean(dim=1)          # [B,d]
        r_pool = r_tokens.mean(dim=1)          # [B,d]

        globals_ = torch.stack(
            [
                qw_norm.sum(dim=1),               # total weighted normalized backlog
                qw_norm.max(dim=1).values,        # largest weighted normalized queue
                q_norm.mean(dim=1),               # mean normalized queue
                (1.0 - busy).mean(dim=1),         # idle fraction
            ],
            dim=-1
        )                                       # [B,4]

        value = self.value_head(torch.cat([q_pool, r_pool, globals_], dim=-1)).squeeze(-1)
        return value