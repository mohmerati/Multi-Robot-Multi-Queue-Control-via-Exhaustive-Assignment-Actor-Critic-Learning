import torch
import torch.nn as nn
from torch.distributions import Categorical
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from src.models.actor import RobotQueueAttentionActor
from src.models.critic import CTDECritic

class QueuePolicyNet_Attn(nn.Module):
    """
    New policy net:
      - assignment-only actor
      - pooled centralized critic
      - no urgency subnetwork
      - no queue-id embedding
    """
    def __init__(
        self,
        M,
        N,
        d_r=16,
        arrival_rates=None,
        queue_cost_weights=None,
        max_queue_length=100,
        actor_d_model=128,
        critic_d_model=128,
    ):
        super().__init__()
        self.M, self.N, self.d_r = M, N, d_r

        rates = torch.as_tensor(arrival_rates, dtype=torch.float32)
        qcw = torch.as_tensor(queue_cost_weights, dtype=torch.float32)

        self.register_buffer("rates", rates)
        self.register_buffer("queue_cost_weights", qcw)

        rate_scale = float(rates.max().item()) if rates.numel() > 0 else 1.0
        weight_scale = float(qcw.max().item()) if qcw.numel() > 0 else 1.0
        max_q_cap = float(max_queue_length)

        self.actor_attn = RobotQueueAttentionActor(
            N=N,
            M=M,
            queue_cost_weights=qcw,
            d_model=actor_d_model,
            d_robot_emb=d_r,
            max_q_cap=max_q_cap,
            rate_scale=rate_scale,
            weight_scale=weight_scale,
        )

        self.critic = CTDECritic(
            N=N,
            M=M,
            queue_cost_weights=qcw,
            d_model=critic_d_model,
            d_robot_emb=d_r,
            max_q_cap=max_q_cap,
            rate_scale=rate_scale,
            weight_scale=weight_scale,
        )

    def forward(self, obs):
        queues = obs["queues"].float()
        B = queues.shape[0]
        N = self.N

        rates_b = self.rates.view(1, N).expand(B, N)

        logits = self.actor_attn(obs, rates_b)   # [B,M,N]
        value = self.critic(obs, self.rates)     # [B]
        return logits, value
    
class QueuePPOPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        arrival_rates,
        queue_cost_weights,
        max_queue_length,
        **kwargs
    ):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        self.M = int(len(action_space.nvec))
        self.N = int(action_space.nvec[0])

        arrival_rates = torch.as_tensor(arrival_rates, dtype=torch.float32, device=self.device)
        queue_cost_weights = torch.as_tensor(queue_cost_weights, dtype=torch.float32, device=self.device)

        self.net = QueuePolicyNet_Attn(
            M=self.M,
            N=self.N,
            d_r=16,
            arrival_rates=arrival_rates,
            queue_cost_weights=queue_cost_weights,
            max_queue_length=max_queue_length,
            actor_d_model=128,
            critic_d_model=128,
        )

        self._action_dims = action_space.nvec.tolist()
        self._dist = MultiCategoricalDistribution(self._action_dims)
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )

    # === IMPORTANT: provide a dummy mlp_extractor so SB3 is happy ===
    def _build_mlp_extractor(self) -> None:
        class _Bypass(nn.Module):
            def __init__(self):
                super().__init__()
                # SB3 looks for these two attributes to size default heads;
                # we give harmless nonzero placeholders.
                self.latent_dim_pi = 1
                self.latent_dim_vf = 1
            def forward(self, features: torch.Tensor):
                # Not used by our policy; return dummy tensors with correct batch dim
                b = features.shape[0] if features.ndim > 0 else 1
                z = torch.zeros((b, 1), device=features.device, dtype=features.dtype)
                return z, z
        self.mlp_extractor = _Bypass()

    def sample_autoreg_and_logp(self, logits, robots, deterministic: bool):
        """
        logits: [B, M, N]  (raw per-robot logits from actor)
        robots: [B, M]     (current robot locations)
        Returns:
        actions: [B, M]
        logp:    [B]
        """
        B, M, N = logits.shape
        dev = logits.device

        occ = torch.zeros(B, N, dtype=torch.bool, device=dev)  # occupied by earlier robots
        actions = torch.empty(B, M, dtype=torch.long, device=dev)
        logp = torch.zeros(B, device=dev)

        for r in range(M):
            # allow: unoccupied OR own location (staying is always allowed)
            self_loc = F.one_hot(robots[:, r], num_classes=N).bool()      # [B, N]
            allow = (~occ) | self_loc                                     # [B, N]

            logits_r = logits[:, r, :].masked_fill(~allow, -1e9)          # [B, N]
            dist_r = Categorical(logits=logits_r)

            if deterministic:
                a_r = torch.argmax(logits_r, dim=-1)
            else:
                a_r = dist_r.sample()

            actions[:, r] = a_r
            logp = logp + dist_r.log_prob(a_r)

            # mark chosen destination as occupied for later robots
            occ.scatter_(1, a_r[:, None], True)

        return actions, logp
    
    def autoreg_logp_of_actions(self, logits, robots, actions):
        """
        Compute log pi(actions|obs) under the same autoregressive masking rule.
        logits:  [B, M, N]
        robots:  [B, M]
        actions: [B, M]
        Returns:
        logp: [B]
        """
        B, M, N = logits.shape
        dev = logits.device
        occ = torch.zeros(B, N, dtype=torch.bool, device=dev)
        logp = torch.zeros(B, device=dev, dtype=logits.dtype)

        actions = actions.long()
        robots = robots.long()

        for r in range(M):
            a_r = actions[:, r].long()
            self_loc = F.one_hot(robots[:, r], num_classes=N).bool()
            logits_r = logits[:, r, :].clone()
            allow = (~occ) | self_loc
            logits_r = logits_r.masked_fill(~allow, -1e9)
            dist_r = Categorical(logits=logits_r)
            logp = logp + dist_r.log_prob(a_r)
            occ.scatter_(1, a_r[:, None], True)

        return logp

    def forward(self, obs, deterministic: bool = False):
        """
        Return: action [B, M], value [B], log_prob [B]
        Uses SB3's MultiCategoricalDistribution so that .predict() and training paths agree.
        """
        logits, value = self.net(obs)                     # [B, M, N], [B]
        robots = obs['robots'].long()                     # [B, M]
        # B = logits.shape[0]
        ## Flatten per-robot logits for MultiCategoricalDistribution: [B, sum(nvec)] = [B, M*N]
        # logits_flat = logits.view(B, -1)
        # dist = self._dist.proba_distribution(logits_flat)
        # action = dist.get_actions(deterministic=deterministic)   # [B, M]
        # log_prob = dist.log_prob(action)                         # [B]
        action, log_prob = self.sample_autoreg_and_logp(logits, robots, deterministic)   # [B, M], [B]
        
        return action, value, log_prob
    
    def _predict(self, observation, deterministic: bool = False):
        """
        SB3 calls this inside model.predict().
        observation: torch.Tensor or dict of torch.Tensor (SB3 already tensorizes obs)
        Return: actions as torch.LongTensor [B, M]
        """
        logits, _ = self.net(observation)          # [B, M, N]
        robots = observation["robots"].long()      # [B, M]
        actions, _ = self.sample_autoreg_and_logp(logits, robots, deterministic)
        return actions

    # SB3 uses this during updates to rebuild the distribution
    def get_distribution(self, obs):
        """
        Build an SB3 MultiCategoricalDistribution from the network logits.
        Returns an object with .get_actions(), .log_prob(), .entropy().
        """
        logits, _ = self.net(obs)               # [B, M, N]
        B = logits.shape[0]
        logits_flat = logits.view(B, -1)        # [B, sum(nvec)]
        return self._dist.proba_distribution(logits_flat)
        # action, log_prob = self.sample_autoreg_and_logp(logits, robots, deterministic)   # [B, M], [B]
        # return lo

    def predict_values(self, obs):
        """
        Get the estimated values according to the current policy.
        :param obs:
        :return: the estimated values.
        """
        # Pass the dictionary observation directly to the network's forward_critic
        return self.forward_critic(obs)


    # Used in PPO training loop to compute log-probs/entropy on minibatches
    def evaluate_actions(self, obs, actions):
        """
        obs: dict batch (from rollout buffer)
        actions: LongTensor of shape [batch, M]
        Return: values [batch], log_prob [batch], entropy [batch]
        """
        logits, value = self.net(obs)                   # [B, M, N], [B]
        # B = logits.shape[0]
        # logits_flat = logits.view(B, -1)              # [B, sum(nvec)]
        # dist = self._dist.proba_distribution(logits_flat)
        # log_prob = dist.log_prob(actions)             # [B]
        # entropy = dist.entropy()                      # [B]
        # return value, log_prob, entropy
        robots = obs['robots'].long()                     # [B, M]
        log_prob = self.autoreg_logp_of_actions(logits, robots, actions)  # [B]
        # Entropy of an AR policy is sum of entropies of each conditional.
        # We compute it along the same realized occupancy trajectory induced by 'actions'.
        B, M, N = logits.shape
        dev = logits.device
        occ = torch.zeros(B, N, dtype=torch.bool, device=dev)
        self_loc = F.one_hot(robots, num_classes=N).bool()

        entropy = torch.zeros(B, dtype=torch.float32, device=dev)
        for r in range(M):
            allow_r = (~occ) | self_loc[:, r, :]
            logits_r = logits[:, r, :].clone()
            logits_r = logits_r.masked_fill(~allow_r, -1e9)
            dist_r = Categorical(logits=logits_r)
            entropy += dist_r.entropy()
            a_r = actions[:, r].long()
            occ.scatter_(1, a_r[:, None], True)

        return value, log_prob, entropy

    # Optional, used by SB3 in some code paths
    def forward_actor(self, obs):
        logits, _ = self.net(obs)
        return logits

    def forward_critic(self, obs):
        _, value = self.net(obs)
        return value