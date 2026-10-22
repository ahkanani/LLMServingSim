import random
from dataclasses import dataclass

from .expert_offload import ExpertDistribution
from .logger import get_logger


_BLOCK_COPY_WARNED = set()


@dataclass
class RoutingResult:
    """Routing information for a single MoE layer.

    ``local_tokens``, ``activated_experts``, and ``source_tokens`` keep the
    legacy per-EP-rank contract used by trace generation.  ``per_expert_tokens``
    is the detailed global expert-token histogram needed by expert offload
    stages.  ``tokens_by_rank_expert`` is optional validation/debug detail with
    shape ``[ep_rank][expert_id]``.
    """
    local_tokens: list          # [rank] -> distinct token count routed to this rank
    activated_experts: list     # [rank] -> number of distinct experts activated on this rank
    source_tokens: list         # [rank] -> token count originating from this rank before dispatch
    per_expert_tokens: list = None     # [expert_id] -> token-expert pair count
    tokens_by_rank_expert: list = None # [rank][expert_id] -> token-expert pair count
    token_expert_ids: object = None    # immutable [real token][top-k] selections for TRACE


class GateRouter:
    """Simulator-side model of the MoE gate + expert dispatch.

    Policies:
        BALANCED  (default) — closed-form pigeonhole approximation of
                              a trained learned gate with load-balancing
                              auxiliary loss. Deterministic.
        RR                  — deterministic round-robin per token.
        RAND                — uniform random per token (seedable).
        CUSTOM              — user-supplied ``_custom_gate_function``.

    ``block_copy``: simulator-side optimization that emits one
    transformer block's trace and replays it ``num_hidden_layers``
    times instead of re-computing the routing every layer. Cuts
    trace-generation time by roughly ``num_hidden_layers`` × on MoE
    models. Safe whenever every layer's routing produces the same
    (local_tokens, activated_experts) pair. CUSTOM routing automatically
    disables block copy when its CSV contains layer-specific rows.
    """

    _SUPPORTED_POLICIES = ("BALANCED", "RR", "RAND", "CUSTOM")

    def __init__(
        self,
        node_id,
        instance_id,
        num_local_experts,
        num_experts_per_tok=1,
        routing_policy="BALANCED",
        seed=42,
        block_copy=True,
        expert_distribution=None,
        routing_distribution_csv=None,
        num_layers=None,
    ):
        self.instance_id = instance_id
        self.E = int(num_local_experts)
        self.k = max(1, min(int(num_experts_per_tok), self.E))
        self.routing_policy = routing_policy.upper()
        self.seed = seed
        self.rnd = random.Random(seed) if seed is not None else random
        self.block_copy = bool(block_copy)
        self.expert_distribution = expert_distribution
        self.routing_distribution_csv = routing_distribution_csv
        self.num_layers = num_layers
        self._logged_custom_distribution_layers = set()
        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)

        if self.routing_policy == "RR":
            self.routing_fn = self._rr_routing
        elif self.routing_policy == "RAND":
            self.routing_fn = self._rand_routing
        elif self.routing_policy == "BALANCED":
            # ``route_ep`` bypasses ``routing_fn`` for this policy —
            # the per-rank (local_tokens, activated_experts) pair is
            # computed analytically. ``routing_fn`` still points at a
            # valid function for the few non-route_ep call sites.
            self.routing_fn = self._rand_routing
        elif self.routing_policy == "CUSTOM":
            self.expert_distribution = self._resolve_expert_distribution(
                self.expert_distribution,
                self.routing_distribution_csv,
                num_layers=self.num_layers,
            )
            if self.expert_distribution is None:
                raise ValueError(
                    "CUSTOM expert routing requires routing_distribution_csv or "
                    "an ExpertDistribution object"
                )
            self._disable_block_copy_if_layer_variant()
            self.routing_fn = self._custom_gate_function
        else:
            raise ValueError(
                f"Unknown routing_policy {routing_policy!r}. "
                f"Supported: {', '.join(self._SUPPORTED_POLICIES)}"
            )

    @staticmethod
    def expert_owner(expert_id, ep_size, num_experts):
        """Determine which EP rank owns a given expert. Even distribution across ranks."""
        return min(int(expert_id * ep_size // num_experts), ep_size - 1)

    def _resolve_expert_distribution(self, expert_distribution, routing_distribution_csv, *, num_layers=None):
        if expert_distribution is not None:
            return expert_distribution
        if routing_distribution_csv in (None, ""):
            return None
        return ExpertDistribution.load(
            routing_distribution_csv,
            num_experts=self.E,
            num_layers=num_layers,
        )

    def _disable_block_copy_if_layer_variant(self):
        """Force per-layer traces when CUSTOM routing has layer overrides."""
        if not self.block_copy:
            return
        if self.expert_distribution is None or not self.expert_distribution.has_layer_specific_rows():
            return
        self.block_copy = False
        key = ("CUSTOM", str(self.expert_distribution.path))
        if key in _BLOCK_COPY_WARNED:
            return
        _BLOCK_COPY_WARNED.add(key)
        self.logger.warning(
            "Disabling MoE block-copy for CUSTOM expert routing because distribution %s "
            "has layer-specific rows; generating per-layer traces.",
            self.expert_distribution.path,
        )

    def _rr_routing(self, token_idx, E, k):
        base = token_idx % E
        return [(base + o) % E for o in range(k)]

    def _rand_routing(self, token_idx, E, k):
        return self.rnd.sample(range(E), k)

    def _custom_gate_function(self, token_idx, E, k):
        raise NotImplementedError(
            "CUSTOM routing uses aggregate CSV allocation via route_ep(); "
            "per-token custom gate sampling is not implemented"
        )

    def route(self, layer_num, batch_id, total_len):
        """Returns flat token counts per expert (used when EP=1)."""
        result = self.route_ep(layer_num, batch_id, total_len, ep_size=1)
        return result.per_expert_tokens

    def route_ep(self, layer_num, batch_id, total_len, ep_size):
        """EP-aware routing: returns per-rank token counts and activated experts.

        Tokens are distributed evenly across EP ranks before dispatch
        (matching vLLM's EP execution model). Each token selects k
        experts; the owning rank receives the token for local
        execution. Expert-to-rank assignment uses even partitioning:
        ``expert_id * ep // num_experts``.

        BALANCED short-circuits the per-token draw and uses the
        pigeonhole expression — trained MoE gates (Qwen3, Mixtral,
        DeepSeek, …) are load-balance-regularised so per-expert
        traffic is approximately uniform at serving time.
        """
        total_len = int(total_len)
        ep_size = max(1, int(ep_size))

        # Distribute source tokens evenly across EP ranks
        base = total_len // ep_size
        remainder = total_len % ep_size
        source_tokens = [base + (1 if r < remainder else 0) for r in range(ep_size)]

        if self.routing_policy == "BALANCED":
            local_tokens, activated_counts, per_expert_tokens, tokens_by_rank_expert = (
                self._balanced_route_ep(total_len, ep_size, source_tokens)
            )
        elif self.routing_policy == "CUSTOM":
            local_tokens, activated_counts, per_expert_tokens, tokens_by_rank_expert = (
                self._custom_route_ep(layer_num, total_len, ep_size)
            )
        else:
            local_tokens, activated_counts, per_expert_tokens, tokens_by_rank_expert = (
                self._sampled_route_ep(total_len, ep_size, source_tokens)
            )

        assigned_pairs = sum(per_expert_tokens)
        expected_pairs = total_len * self.k
        if assigned_pairs != expected_pairs:
            raise AssertionError(
                f"Routing invariant failed for policy={self.routing_policy}: "
                f"sum(per_expert_tokens)={assigned_pairs}, expected={expected_pairs}"
            )

        self.logger.info(
            "layer=%d policy=%s E=%d k=%d ep=%d batch=%s tokens=%d local=%s activated=%s",
            layer_num, self.routing_policy, self.E, self.k, ep_size,
            batch_id, total_len, local_tokens, activated_counts,
        )

        return RoutingResult(
            local_tokens=local_tokens,
            activated_experts=activated_counts,
            source_tokens=source_tokens,
            per_expert_tokens=per_expert_tokens,
            tokens_by_rank_expert=tokens_by_rank_expert,
        )

    def _sampled_route_ep(self, total_len, ep_size, source_tokens):
        """Per-token routing path used by non-BALANCED policies.

        ``local_tokens`` is deliberately counted once per destination rank per
        token, while ``per_expert_tokens`` counts every selected expert-token
        pair.  This preserves the legacy MoE latency lookup contract and also
        exposes the finer histogram needed to split hot/cold experts later.
        """
        local_tokens = [0] * ep_size
        activated_experts = [set() for _ in range(ep_size)]
        per_expert_tokens = [0] * self.E
        tokens_by_rank_expert = [[0] * self.E for _ in range(ep_size)]
        token_idx = 0

        for src_rank in range(ep_size):
            for _ in range(source_tokens[src_rank]):
                selected = self.routing_fn(token_idx, self.E, self.k)
                if len(selected) != self.k:
                    raise ValueError(
                        f"Routing policy {self.routing_policy} returned {len(selected)} "
                        f"experts for token {token_idx}; expected k={self.k}"
                    )

                dest_ranks = set()
                seen_experts = set()
                for expert_id in selected:
                    expert_id = int(expert_id)
                    if expert_id < 0 or expert_id >= self.E:
                        raise ValueError(
                            f"Routing policy {self.routing_policy} returned expert_id={expert_id}; "
                            f"valid range is [0, {self.E})"
                        )
                    if expert_id in seen_experts:
                        raise ValueError(
                            f"Routing policy {self.routing_policy} returned duplicate expert_id={expert_id} "
                            f"for token {token_idx}"
                        )
                    seen_experts.add(expert_id)

                    owner = self.expert_owner(expert_id, ep_size, self.E)
                    per_expert_tokens[expert_id] += 1
                    tokens_by_rank_expert[owner][expert_id] += 1
                    activated_experts[owner].add(expert_id)
                    dest_ranks.add(owner)

                for owner in dest_ranks:
                    local_tokens[owner] += 1
                token_idx += 1

        activated_counts = [len(s) for s in activated_experts]
        return local_tokens, activated_counts, per_expert_tokens, tokens_by_rank_expert

    def _custom_route_ep(self, layer_num, total_len, ep_size):
        """Route from a CSV-backed probability distribution.

        The first CUSTOM implementation is deterministic aggregate routing:
        allocate ``total_len * k`` expert-token pairs across experts using the
        normalized CSV probabilities and largest-remainder rounding.  Because
        the aggregate distribution does not preserve per-token co-activation,
        per-rank ``local_tokens`` is a conservative upper bound:
        ``min(total_len, pairs_to_rank)``.  This avoids under-estimating NPU MoE
        lookup tokens before later stages add exact hot/cold split semantics.
        """

        if self.expert_distribution is None:
            raise ValueError(
                "CUSTOM expert routing requires routing_distribution_csv or "
                "an ExpertDistribution object"
            )

        per_expert_tokens = self.expert_distribution.allocate_counts_for_layer(
            int(layer_num),
            int(total_len) * self.k,
            num_experts=self.E,
        )
        if layer_num not in self._logged_custom_distribution_layers:
            probs = self.expert_distribution.probabilities_for_layer(
                int(layer_num),
                num_experts=self.E,
            )
            self.logger.info(
                "layer=%d policy=CUSTOM distribution=%s normalized_sum=%.12f",
                int(layer_num), self.expert_distribution.path, sum(probs),
            )
            self._logged_custom_distribution_layers.add(layer_num)

        tokens_by_rank_expert = [[0] * self.E for _ in range(ep_size)]
        pairs_by_rank = [0] * ep_size
        activated_counts = [0] * ep_size
        for expert_id, count in enumerate(per_expert_tokens):
            owner = self.expert_owner(expert_id, ep_size, self.E)
            tokens_by_rank_expert[owner][expert_id] = count
            pairs_by_rank[owner] += count
            if count > 0:
                activated_counts[owner] += 1

        local_tokens = [min(int(total_len), pairs) for pairs in pairs_by_rank]
        return local_tokens, activated_counts, per_expert_tokens, tokens_by_rank_expert

    def _balanced_route_ep(self, total_len, ep_size, source_tokens):
        """Closed-form per-rank load for a perfectly-balanced learned gate.

        Pigeonhole model:
          * total expert-token pairs = ``total_len * k``
          * split evenly across experts using deterministic largest remainder
          * per-rank activated experts come from nonzero owned expert bins

        Per-rank token count keeps the legacy probability approximation:

          P(token hits r) = 1 − ((ep − 1) / ep) ** k

        which collapses to ~1 when ``k ≫ ep`` (so every token reaches
        every rank) and degrades gracefully for small-k / large-ep.
        """
        k = self.k
        per_expert_tokens = self._evenly_split_pairs(total_len * k, self.E)
        tokens_by_rank_expert = [[0] * self.E for _ in range(ep_size)]
        for expert_id, count in enumerate(per_expert_tokens):
            owner = self.expert_owner(expert_id, ep_size, self.E)
            tokens_by_rank_expert[owner][expert_id] = count

        # Preserve the legacy BALANCED aggregate model used by existing MoE
        # latency lookups; per_expert_tokens is a new detailed companion, not
        # a behavior change for NPU expert timing.
        E_rank = max(1, self.E // ep_size)
        pairs_per_rank = (total_len * k) / ep_size
        activated_per_rank = min(int(round(pairs_per_rank)), E_rank)
        activated_counts = [activated_per_rank] * ep_size

        if ep_size <= 1:
            hit_prob = 1.0
        else:
            hit_prob = 1.0 - ((ep_size - 1) / ep_size) ** k
        tokens_to_r = int(round(total_len * hit_prob))
        local_tokens = [tokens_to_r] * ep_size
        return local_tokens, activated_counts, per_expert_tokens, tokens_by_rank_expert

    @staticmethod
    def _evenly_split_pairs(total_pairs, buckets):
        """Deterministically split integer pair count across buckets."""
        buckets = max(1, int(buckets))
        base = int(total_pairs) // buckets
        remainder = int(total_pairs) % buckets
        return [base + (1 if idx < remainder else 0) for idx in range(buckets)]
