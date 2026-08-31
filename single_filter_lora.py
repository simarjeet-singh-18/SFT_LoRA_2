import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _simulate_quantize_dequantize(W: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """
    Dependency-free stand-in for the quantization step LoftQ's init procedure needs.

    Real LoftQ/QLoRA implementations quantize to NF4 (a non-uniform, information-
    theoretically-motivated 4-bit codebook) via bitsandbytes' CUDA kernels. This repo
    intentionally avoids that dependency (bitsandbytes needs a working CUDA build
    matched to the exact torch/CUDA version, and offline compute nodes -- like a
    SLURM cluster with no internet access on the compute nodes -- often can't `pip
    install` it at all). Instead, this does simple per-tensor uniform affine
    quantization: linearly map W's [min, max] range onto 2**bits integer levels,
    then map back to float. This captures the CORE idea LoftQ's init needs
    (quantization loses information; the low-rank adapter should be initialized to
    approximate what was lost, not started from an unrelated random point) without
    matching NF4's bit-exact behavior. Treat --init-method loftq as an approximation
    of the paper's IDEA, not a reproduction of its exact numbers.
    """
    with torch.no_grad():
        w_min, w_max = W.min(), W.max()
        levels = 2 ** bits
        if (w_max - w_min).item() == 0.0:
            return W.clone()
        scale = (w_max - w_min) / (levels - 1)
        q = torch.round((W - w_min) / scale)
        q = torch.clamp(q, 0, levels - 1)
        return q * scale + w_min


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear layer with a low-rank adapter. Supports two adapter
    architectures (adapter_type) and two initialization schemes (init_method),
    independently combinable:

    adapter_type="lora" (default): standard LoRA,
        W' = W_base + scaling * (B @ A)

    adapter_type="dora": Weight-Decomposed Low-Rank Adaptation. Decomposes the
    EFFECTIVE weight into a trainable per-output-neuron magnitude vector `m` and a
    direction that LoRA's low-rank update reshapes:
        V'    = W_base + scaling * (B @ A)                    # same as LoRA's W'
        W'    = m * V' / ||V'||_row                            # m: (out_features,)
    where ||.||_row is the L2 norm of each output neuron's row (dim=1), matching the
    convention used elsewhere (e.g. Hugging Face PEFT's DoRA). `m` is initialized to
    W_base's own row norms, so W' = W_base exactly at init (same "no-op at init"
    property LoRA gets from zero-initializing B). Adds `out_features` extra
    trainable scalars per layer -- negligible parameter cost -- and empirically
    tends to track full fine-tuning's update direction more closely than plain LoRA.

    init_method="default" (unchanged): lora_A ~ Kaiming-uniform, lora_B = 0.

    init_method="loftq": see _loftq_init below -- alternating quantize/SVD-residual
    fit, replacing base_layer's weight with a quantized version and initializing
    lora_A/lora_B to approximate what quantization lost, instead of starting from a
    near-zero effective update.
    """
    def __init__(self, base_layer: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.0,
                 apply_ortho: bool = True, adapter_type: str = "lora", init_method: str = "default",
                 loftq_bits: int = 4, loftq_iters: int = 5):
        super().__init__()
        assert adapter_type in ("lora", "dora"), f"adapter_type must be 'lora' or 'dora', got {adapter_type!r}"
        assert init_method in ("default", "loftq"), f"init_method must be 'default' or 'loftq', got {init_method!r}"
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.adapter_type = adapter_type
        self.init_method = init_method
        # Whether compute_lora_orthogonality_loss should include this layer.
        # Lets a subset of LoRA-wrapped layers use the orthogonality regularizer
        # ("SFT+LoRA-ortho") while the rest behave as plain LoRA -- e.g. restricting
        # the (more expensive/restrictive) orthogonality penalty to only the blocks
        # most sensitive to the downstream task (see select_top_sensitive_blocks
        # below), rather than applying it uniformly across every LoRA layer in the
        # network. Default True preserves the old all-layers behavior for any
        # existing code that doesn't pass this explicitly.
        self.apply_ortho = apply_ortho

        if rank > 0:
            # IMPORTANT: match base_layer.weight's device/dtype here, not just
            # leave these as default CPU float32 tensors. Plain LoRA gets away
            # with skipping this (nothing touches lora_A/lora_B until the whole
            # model's later .to(device) call), but DoRA's magnitude init below
            # computes with base_layer.weight IMMEDIATELY, inside __init__ --
            # if the model was already moved to GPU before inject_lora() runs
            # (the normal case: train_sfp_lora.py does model.to(device) early,
            # then injects LoRA/DoRA afterward), base_layer.weight is already on
            # cuda while a device-less torch.zeros(...) defaults to cpu, causing
            # "Expected all tensors to be on the same device" at this exact line.
            base_device = base_layer.weight.device
            base_dtype = base_layer.weight.dtype
            self.lora_A = nn.Parameter(torch.zeros(rank, base_layer.in_features,
                                                     device=base_device, dtype=base_dtype))
            self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank,
                                                     device=base_device, dtype=base_dtype))
            self.reset_parameters()

            if init_method == "loftq":
                self._loftq_init(bits=loftq_bits, n_iters=loftq_iters)

            if adapter_type == "dora":
                # Row norm of the EFFECTIVE weight at this point (== base_layer's
                # weight if init_method="default" since B=0 there; == the
                # quantized+residual-fit weight if init_method="loftq"). Either
                # way, initializing m to this row norm makes W' == the current
                # effective weight exactly at init.
                with torch.no_grad():
                    V0 = self.base_layer.weight + self.scaling * (self.lora_B @ self.lora_A)
                    row_norm = V0.norm(p=2, dim=1)  # (out_features,)
                self.magnitude = nn.Parameter(row_norm)

    def reset_parameters(self):
        if self.rank > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    @torch.no_grad()
    def _loftq_init(self, bits: int = 4, n_iters: int = 5):
        """
        Alternating quantize / SVD-residual-fit initialization (approximates the
        LoftQ paper's Algorithm 1; see _simulate_quantize_dequantize for the
        quantization-backend caveat).

        For n_iters rounds:
          1. quantize the current residual R = W_base - B@A  ->  Q
          2. take the low-rank SVD of what Q couldn't capture, W_base - Q, and
             refit B@A to approximate it (top-`rank` singular directions)
        After the loop, base_layer.weight is REPLACED by the final Q (frozen, as
        always), and lora_A/lora_B hold the low-rank fit to (W_base - Q) -- so
        Q + B@A approximates the original full-precision W_base as closely as this
        rank/bit-width combination allows, rather than starting from B=0 (i.e. an
        effective update of exactly zero, as in default init) on top of a
        quantized base_layer.weight (which is how naive QLoRA initializes: better
        than nothing, but LoftQ's whole point is that alternating with the SVD fit
        gets a measurably tighter approximation).
        """
        W = self.base_layer.weight.data.clone()
        BA = torch.zeros_like(W)
        # Fallback defaults in case SVD fails on the very first iteration (rare, but
        # possible on a pathological all-zero or near-singular residual) -- keeps
        # the layer at its safe default-init state (zero effective update, original
        # full-precision weight) instead of crashing the whole run.
        B_new, A_new = self.lora_B.data.clone(), self.lora_A.data.clone()
        Q_final = W.clone()

        for _ in range(n_iters):
            # Quantize what the low-rank term (from the previous round) doesn't
            # already capture, then re-fit the low-rank term via SVD of (W - Q) --
            # i.e. what's left of the ORIGINAL weight after subtracting this
            # round's quantized approximation, NOT (residual - Q). Fitting against
            # (residual - Q) instead of (W - Q) was tried first and verified (via a
            # numpy prototype) to make the reconstruction error oscillate instead
            # of converge across iterations; (W - Q) converges monotonically-ish
            # and consistently beats naive single-shot quantization.
            residual = W - BA
            Q = _simulate_quantize_dequantize(residual, bits=bits)
            target_for_svd = W - Q
            try:
                U, S, Vh = torch.linalg.svd(target_for_svd, full_matrices=False)
            except RuntimeError:
                break
            r = self.rank
            U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
            sqrt_S_r = torch.sqrt(S_r.clamp(min=0.0))
            B_new = U_r * sqrt_S_r.unsqueeze(0)          # (out_features, rank)
            A_new = sqrt_S_r.unsqueeze(1) * Vh_r          # (rank, in_features)
            BA = B_new @ A_new
            Q_final = Q

        # Absorb the fixed scaling factor now so that forward()'s
        # self.scaling * (B @ A) reproduces BA exactly (B_new @ A_new above is the
        # raw residual fit, not yet divided by this layer's LoRA scaling factor).
        # self.scaling is always > 0 here (rank > 0 is guaranteed by the caller).
        sqrt_scaling = math.sqrt(self.scaling)
        self.lora_B.data.copy_(B_new / sqrt_scaling)
        self.lora_A.data.copy_(A_new / sqrt_scaling)
        self.base_layer.weight.data.copy_(Q_final)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank <= 0:
            return self.base_layer(x)

        if self.adapter_type == "lora":
            result = self.base_layer(x)
            lora_out = F.linear(self.dropout(x), self.lora_A)
            lora_out = F.linear(lora_out, self.lora_B)
            return result + self.scaling * lora_out

        # adapter_type == "dora": build the full effective weight explicitly (can't
        # decompose into base_layer(x) + delta the way plain LoRA does, since the
        # row-norm renormalization couples every output neuron's scale to the FULL
        # V = W_base + scaling*(B@A), not just the low-rank part). Dropout is
        # applied once, to the input, matching standard LoRA/DoRA dropout placement.
        V = self.base_layer.weight + self.scaling * (self.lora_B @ self.lora_A)
        row_norm = V.norm(p=2, dim=1, keepdim=True)  # (out_features, 1)
        W_eff = self.magnitude.unsqueeze(1) * V / row_norm.clamp(min=1e-8)
        return F.linear(self.dropout(x), W_eff, self.base_layer.bias)

    def orthogonality_penalty(self):
        """
        Returns (A_term, B_term), the (rank x rank) residual matrices whose squared
        Frobenius norm, NORMALIZED BY rank^2, penalizes lora_A / lora_B for being
        far from row/column-orthonormal. None if rank <= 0 (no LoRA params to
        regularize on this layer).

        Shapes: lora_A is (rank, in_features), lora_B is (out_features, rank), with
        rank << in_features/out_features. Because of that, lora_A can only ever have
        AT MOST `rank` linearly independent directions among its in_features-dim rows
        -- so the meaningful orthogonality constraint is on those `rank` ROWS being
        mutually orthonormal, i.e. A @ A.T ~ I_rank (a (rank x rank) identity is
        achievable; forcing A.T @ A, which is (in_features x in_features) and has
        rank <= rank < in_features, could never equal an identity matrix).
        Symmetrically for lora_B, whose `rank` COLUMNS are the quantity that can be
        made orthonormal: B.T @ B ~ I_rank.

        Driving both toward the identity pushes each adapter's `rank` update
        directions to be linearly independent of one another, i.e. discourages the
        adapter from wasting capacity by learning redundant (near-parallel) columns.

        NORMALIZATION: A_term/B_term are (rank x rank) matrices, so their raw
        squared Frobenius norm scales with O(rank^2) even at a FIXED relative
        deviation from orthonormality (e.g. lora_B starts at exactly zero, so
        B_term = -I_rank and ||B_term||_F^2 = rank at initialization alone, before
        any training). Dividing by rank^2 here turns this into a mean squared
        per-entry deviation, so a given --lora-ortho-lambda1/2 value means roughly
        the same regularization STRENGTH regardless of --lora-rank -- previously,
        doubling --lora-rank would roughly double the raw penalty at the same
        actual orthonormality, silently requiring the lambda to be re-tuned every
        time --lora-rank changed.
        """
        if self.rank <= 0:
            return None
        eye_r = torch.eye(self.rank, device=self.lora_A.device, dtype=self.lora_A.dtype)
        A_term = (self.lora_A @ self.lora_A.t() - eye_r) / self.rank   # (rank, rank), pre-normalized
        B_term = (self.lora_B.t() @ self.lora_B - eye_r) / self.rank   # (rank, rank), pre-normalized
        return A_term, B_term


def compute_lora_orthogonality_loss(model: nn.Module, lambda1: float = 0.0, lambda2: float = 0.0) -> torch.Tensor:
    """
    Averages (not sums) the orthogonality regularizer
    lambda1 * ||A@A.T - I||_F^2 + lambda2 * ||B.T@B - I||_F^2 (each pre-normalized
    by rank -- see LoRALinear.orthogonality_penalty) over every LoRALinear
    submodule in `model` that has rank > 0 AND apply_ortho=True (module.apply_ortho
    defaults to True, so with no selective marking this covers every LoRA layer,
    exactly as before this flag existed; see inject_lora's ortho_block_indices
    param and select_top_sensitive_blocks for how to mark only a subset).

    NORMALIZATION: this now AVERAGES across LoRA-wrapped layers instead of
    SUMMING. Summing meant the total penalty magnitude scaled with however many
    linear layers happened to have LoRA injected into them (e.g. it changes with
    --num-filter-blocks, or with target_keywords covering more/fewer sublayers),
    so the same lambda value applied a very different effective regularization
    strength on a shallow vs. a deep injection pattern. Averaging makes
    --lora-ortho-lambda1/2 behave consistently across those configuration
    choices, so it's tunable as "how strongly do I want each adapter regularized"
    rather than "how strongly do I want the WHOLE MODEL regularized, which
    happens to depend on how many layers got LoRA".

    Returns a 0-dim tensor on the same device as the model's parameters, so it can
    always be added directly to the task loss (returns exactly 0.0, with no graph
    attached to lora_A/lora_B, when both lambdas are 0 -- the default -- so runs
    that don't pass either flag are numerically unaffected).

    NOTE ON EXISTING --lora-ortho-lambda1/2 VALUES: because this normalizes both
    by rank (per-term) and by layer count (via the average), the same lambda
    value now produces a MUCH SMALLER raw loss contribution than before this
    change (previously, the unnormalized sum could already be comparable in
    magnitude to the task loss itself right at initialization -- see module
    docstring). If you were already using e.g. --lora-ortho-lambda1 0.01
    --lora-ortho-lambda2 0.01, you will likely want to increase both by roughly
    one to two orders of magnitude (e.g. try 0.1-1.0) to get a comparable
    regularization STRENGTH to what the old unnormalized version applied --
    the old values weren't wrong, they just meant something different (and less
    reliably reproducible) than they will now.
    """
    device = next(model.parameters()).device
    total = torch.zeros((), device=device)
    if lambda1 == 0.0 and lambda2 == 0.0:
        return total

    n_layers = 0
    for module in model.modules():
        if isinstance(module, LoRALinear) and module.rank > 0 and module.apply_ortho:
            terms = module.orthogonality_penalty()
            if terms is None:
                continue
            A_term, B_term = terms
            layer_loss = torch.zeros((), device=device)
            if lambda1 != 0.0:
                layer_loss = layer_loss + lambda1 * torch.sum(A_term * A_term)
            if lambda2 != 0.0:
                layer_loss = layer_loss + lambda2 * torch.sum(B_term * B_term)
            total = total + layer_loss
            n_layers += 1

    if n_layers > 0:
        total = total / n_layers
    return total


class FilterResidualMLP(nn.Module):
    """
    Optional nonlinear residual branch for a filter block: fc2(GELU(fc1(x))).

    fc2 is ZERO-INITIALIZED (weight and bias), so this branch contributes EXACTLY
    ZERO at initialization -- the same safe-start trick LoRALinear already uses for
    its own lora_B matrix. This means attaching this branch to a filter block does
    NOT disturb that block's pseudoinverse-inherited behavior at step 1; the branch
    can only start contributing once gradients move fc2 away from zero during
    training.

    fc1 keeps its default (random) init. This is fine precisely because fc2 starts
    at zero: whatever fc1 outputs gets multiplied by zero at fc2 regardless, so a
    random fc1 can't destabilize anything at initialization.

    Unlike the purely-linear --filter-block-layers stack, this branch genuinely
    adds expressivity (the GELU nonlinearity means fc2(GELU(fc1(x))) is NOT
    reducible to a single linear map) -- this is the mechanism for real added
    capacity in the filter block, implemented as a safe zero-init residual rather
    than by making the block's main path nonlinear (which would break the
    closed-form pseudoinverse init and risk destabilizing the inherited behavior).

    scaling = alpha / hidden_dim, mirroring LoRALinear's alpha/rank normalization:
    keeps the branch's effective update magnitude comparable across different
    hidden_dim choices once it does start contributing.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.scaling = alpha / hidden_dim if hidden_dim > 0 else 1.0

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc2(self.act(self.fc1(x)))
        return self.scaling * self.dropout(out)


class SingleFilterBlock(nn.Module):
    """
    Single linear filter block replacing a pruned Transformer block.
    Initialized via Ridge Pseudo-Inverse (paper Eq. 3-4).

    Optionally attaches a nonlinear zero-init residual branch (FilterResidualMLP)
    when residual_hidden_dim > 0 -- see that class's docstring for why this is the
    safe way to add genuine nonlinear capacity without disturbing the pseudoinverse
    init. Default (residual_hidden_dim=0) is UNCHANGED from all previous versions:
    no residual submodule is constructed at all, so state_dict keys for existing
    checkpoints trained without this feature still match exactly.
    """
    def __init__(self, embed_dim: int, dropout: float = 0.0,
                 residual_hidden_dim: int = 0, residual_alpha: float = 1.0, residual_dropout: float = 0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        self.dropout = nn.Dropout(p=dropout)

        self.residual = None
        if residual_hidden_dim > 0:
            self.residual = FilterResidualMLP(
                embed_dim=embed_dim, hidden_dim=residual_hidden_dim,
                alpha=residual_alpha, dropout=residual_dropout,
            )

    def init_from_pinv(self, X_in: torch.Tensor, X_out: torch.Tensor):
        """
        Fits matrix W such that X_in @ W ≈ X_out. Only touches the main linear
        path (self.weight/self.bias) -- the residual branch (if present) stays at
        its zero-init starting point regardless, exactly as intended.
        """
        with torch.no_grad():
            X_in_flat = X_in.reshape(-1, X_in.size(-1)).float()
            X_out_flat = X_out.reshape(-1, X_out.size(-1)).float()

            eye = torch.eye(X_in_flat.size(1), device=X_in.device) * 1e-4
            pinv = torch.linalg.solve(X_in_flat.T @ X_in_flat + eye, X_in_flat.T @ X_out_flat)

            self.weight.copy_(pinv.T.to(self.weight.dtype))
            self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.residual is not None:
            out = out + self.residual(x)
        return self.dropout(out)


class MultiLayerFilterBlock(nn.Module):
    """
    N-layer (N >= 2) generalization of SingleFilterBlock's linear filter block.

    IMPORTANT: the stacked layers themselves are purely linear (no activation
    between them), so stacking them does NOT add expressivity beyond a single
    layer -- the composed transformation is mathematically still just one linear
    map (matrix product collapses). This class exists to let you experiment with
    depth/parameterization while preserving the EXACT SAME inheritance property as
    the single-layer case. For genuine added capacity, attach a nonlinear
    zero-init residual branch instead (residual_hidden_dim > 0 -- see
    FilterResidualMLP's docstring); that's a separate, safer mechanism than making
    this stack itself nonlinear, since it doesn't disturb the closed-form
    pseudoinverse init.

    Weight initialization generalizes the paper's pseudoinverse trick (Eq. 3-4) to
    N layers as follows:
      - Solve the SAME problem as the single-layer case: min_W ||X_in@W - X_out||_F^2
        -> W = X_in^+ @ X_out
      - Initialize exactly ONE layer (the last one, closest to the block's output)
        with W and zero bias -- identical to SingleFilterBlock's own init
      - Initialize every OTHER layer to the identity transform (identity weight
        matrix, zero bias)
      - Composing identity maps changes nothing, so the WHOLE STACK's product is
        IDENTICAL to a single layer initialized with W, regardless of N. This
        exactly preserves the "inherits the original block's behavior" property
        at any depth.
    """

    def __init__(self, embed_dim: int, num_layers: int, dropout: float = 0.0,
                 residual_hidden_dim: int = 0, residual_alpha: float = 1.0, residual_dropout: float = 0.0):
        super().__init__()
        assert num_layers >= 2, "MultiLayerFilterBlock requires num_layers >= 2 (use SingleFilterBlock for 1)"
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        # Identity-init every layer up front, so the block is at least a
        # well-behaved no-op even before init_from_pinv() is called (rather than
        # torch's default random nn.Linear init).
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.copy_(torch.eye(embed_dim))
                layer.bias.zero_()

        self.dropout = nn.Dropout(p=dropout)

        self.residual = None
        if residual_hidden_dim > 0:
            self.residual = FilterResidualMLP(
                embed_dim=embed_dim, hidden_dim=residual_hidden_dim,
                alpha=residual_alpha, dropout=residual_dropout,
            )

    def init_from_pinv(self, X_in: torch.Tensor, X_out: torch.Tensor):
        """
        Fits W such that X_in @ W ~= X_out (same as SingleFilterBlock.init_from_pinv),
        assigns W to the LAST layer, and resets every other layer to identity. See
        class docstring for why this exactly generalizes the single-layer inheritance
        property to any depth.
        """
        with torch.no_grad():
            X_in_flat = X_in.reshape(-1, X_in.size(-1)).float()
            X_out_flat = X_out.reshape(-1, X_out.size(-1)).float()

            eye = torch.eye(X_in_flat.size(1), device=X_in.device) * 1e-4
            pinv = torch.linalg.solve(X_in_flat.T @ X_in_flat + eye, X_in_flat.T @ X_out_flat)
            W = pinv.T.to(self.layers[0].weight.dtype)

            for layer in self.layers[:-1]:
                layer.weight.copy_(torch.eye(self.embed_dim, device=W.device, dtype=W.dtype))
                layer.bias.zero_()
            self.layers[-1].weight.copy_(W)
            self.layers[-1].bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_input = x  # residual branch sees the block's ORIGINAL input, not the stacked-layer intermediate
        for layer in self.layers:
            x = layer(x)
        if self.residual is not None:
            x = x + self.residual(residual_input)
        return self.dropout(x)


def make_filter_block(embed_dim: int, num_layers: int = 1, dropout: float = 0.0,
                       residual_hidden_dim: int = 0, residual_alpha: float = 1.0, residual_dropout: float = 0.0):
    """
    Factory: returns SingleFilterBlock for num_layers==1 (unchanged, backward
    compatible with all existing checkpoints), or MultiLayerFilterBlock for
    num_layers > 1. residual_hidden_dim > 0 attaches a nonlinear zero-init
    residual branch to either (see FilterResidualMLP); default 0 = no residual
    branch, fully backward compatible.
    """
    if num_layers <= 1:
        return SingleFilterBlock(embed_dim=embed_dim, dropout=dropout,
                                  residual_hidden_dim=residual_hidden_dim,
                                  residual_alpha=residual_alpha, residual_dropout=residual_dropout)
    return MultiLayerFilterBlock(embed_dim=embed_dim, num_layers=num_layers, dropout=dropout,
                                  residual_hidden_dim=residual_hidden_dim,
                                  residual_alpha=residual_alpha, residual_dropout=residual_dropout)


def _normalize_indices(block_idx_or_indices) -> set:
    """Accepts a single int or any iterable of ints; always returns a set of ints."""
    if isinstance(block_idx_or_indices, int):
        return {block_idx_or_indices}
    return set(block_idx_or_indices)


def count_parameter_breakdown(model: nn.Module, pruned_block_idx=None) -> dict:
    """
    Categorizes every parameter in the model for reporting/plotting purposes.
    Categories (checked in this order, first match wins):
      - filter_block     : any SingleFilterBlock/MultiLayerFilterBlock replacing
                            model.blocks[idx] for idx in pruned_block_idx
      - lora              : LoRA A/B matrices injected into remaining blocks
      - layernorm         : unfrozen LN params (norm1/norm2/final norm)
      - head              : classifier head
      - trainable_backbone: anything else that IS trainable (e.g. full-finetune mode,
                             where the whole backbone is unfrozen and there's no filter
                             block / LoRA to separate out)
      - frozen_backbone   : anything else that is NOT trainable
    Also returns total_params, trainable_params, and trainable_pct.

    pruned_block_idx: None (full-finetune runs, no block substitution happened),
    a single int (legacy single-block runs), or a list/set of ints (multi-block runs).
    """
    pruned_indices = _normalize_indices(pruned_block_idx) if pruned_block_idx is not None else set()

    breakdown = {
        "filter_block": 0,
        "lora": 0,
        "layernorm": 0,
        "head": 0,
        "trainable_backbone": 0,
        "frozen_backbone": 0,
    }
    total = 0
    trainable = 0

    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n

        if any(f"blocks.{idx}" in name for idx in pruned_indices):
            breakdown["filter_block"] += n
        elif "lora_" in name or name.endswith(".magnitude"):
            breakdown["lora"] += n
        elif "norm" in name.lower():
            breakdown["layernorm"] += n
        elif "head" in name:
            breakdown["head"] += n
        elif param.requires_grad:
            breakdown["trainable_backbone"] += n
        else:
            breakdown["frozen_backbone"] += n

    breakdown["total_params"] = total
    breakdown["trainable_params"] = trainable
    breakdown["trainable_pct"] = (100.0 * trainable / total) if total > 0 else 0.0
    return breakdown


def substitute_filter_block(model: nn.Module, block_idx: int, num_layers: int = 1, dropout: float = 0.0,
                             residual_hidden_dim: int = 0, residual_alpha: float = 1.0, residual_dropout: float = 0.0):
    """
    Replaces model.blocks[block_idx] with a fresh filter block (SingleFilterBlock
    if num_layers==1, MultiLayerFilterBlock otherwise). Does NOT perform the
    pseudoinverse init -- call .init_from_pinv(X_in, X_out) on the returned module
    afterward, using block inputs/outputs extracted from the model's CURRENT state.

    For multi-block runs, substitute blocks in INCREASING index order and extract
    each block's I/O data AFTER earlier substitutions have already happened, so
    later filter blocks correctly learn to map from the already-modified preceding
    representations (mirrors the paper's own sequential dual-layer construction,
    Fig. 3).

    residual_hidden_dim > 0 attaches a nonlinear zero-init residual branch to the
    filter block (see FilterResidualMLP) -- default 0 = no residual branch.
    """
    embed_dim = getattr(model, "embed_dim", 768)
    filter_block = make_filter_block(embed_dim=embed_dim, num_layers=num_layers, dropout=dropout,
                                      residual_hidden_dim=residual_hidden_dim,
                                      residual_alpha=residual_alpha, residual_dropout=residual_dropout)
    model.blocks[block_idx] = filter_block
    return filter_block


def compute_block_target_param_count(block: nn.Module, target_keywords: list = ["qkv", "proj", "fc1", "fc2"]) -> int:
    """
    Sums weight+bias parameter counts over every nn.Linear submodule of `block`
    whose name matches target_keywords -- i.e. the same set of layers inject_lora
    would wrap with adapters. Used to answer "how many parameters would full
    fine-tuning have trained in this block's target layers", as the reference
    point for --compensate-params: how many parameters were LOST by replacing this
    block with a (typically much smaller) filter block, that we then try to
    compensate for by adding extra LoRA/DoRA rank to the OTHER blocks.

    Call this BEFORE substitute_filter_block replaces the block -- afterward, the
    original Linear layers no longer exist to count.
    """
    total = 0
    for name, module in block.named_modules():
        if any(kw in name for kw in target_keywords) and isinstance(module, nn.Linear):
            total += module.weight.numel()
            if module.bias is not None:
                total += module.bias.numel()
    return total


def compute_compensated_rank(
    model: nn.Module,
    excluded_block_indices,
    base_rank: int,
    param_deficit: int,
    target_keywords: list = ["qkv", "proj", "fc1", "fc2"],
) -> dict:
    """
    Solves for a LoRA/DoRA rank >= base_rank such that the EXTRA adapter
    parameters added (relative to base_rank) across every LoRA-eligible layer in
    every non-excluded block approximately covers param_deficit -- the parameter
    "budget" lost by replacing a block with a smaller filter block (see
    compute_block_target_param_count).

    Each target Linear layer of shape (out_features, in_features) costs
    (in_features + out_features) extra parameters per +1 rank (that's exactly
    lora_A's and lora_B's per-rank-unit size: lora_A row is in_features long,
    lora_B column is out_features long). Summing that over every target layer in
    every non-excluded block gives a total "cost per rank unit"; dividing the
    deficit by that gives how many EXTRA ranks are needed.

    NOTE: this only accounts for the rank-DEPENDENT cost. DoRA's per-layer
    magnitude vector (out_features per layer) is a separate, rank-INDEPENDENT
    fixed cost that doesn't change with rank, so it's intentionally excluded from
    this "cost per rank unit" calculation (including it would bias the solved
    rank without actually helping compensate proportionally to the deficit).

    Returns {"compensated_rank": int, "cost_per_rank_unit": int, "extra_rank": int}.
    If param_deficit <= 0 (filter block wasn't actually smaller, or --compensate-params
    is being used somewhere it doesn't make sense), returns base_rank unchanged.
    """
    excluded = _normalize_indices(excluded_block_indices)
    cost_per_rank_unit = 0
    for idx, block in enumerate(model.blocks):
        if idx in excluded:
            continue
        for name, module in block.named_modules():
            if any(kw in name for kw in target_keywords) and isinstance(module, nn.Linear):
                cost_per_rank_unit += module.in_features + module.out_features

    if param_deficit <= 0 or cost_per_rank_unit <= 0:
        return {"compensated_rank": base_rank, "cost_per_rank_unit": cost_per_rank_unit, "extra_rank": 0}

    extra_rank = math.ceil(param_deficit / cost_per_rank_unit)
    return {
        "compensated_rank": base_rank + extra_rank,
        "cost_per_rank_unit": cost_per_rank_unit,
        "extra_rank": extra_rank,
    }


def select_top_sensitive_blocks(saliencies: dict, excluded_indices, num_blocks: int) -> list:
    """
    Picks the `num_blocks` block indices with the HIGHEST SNIP saliency among
    candidates NOT already in excluded_indices (the filter-substituted blocks).

    Context: SNIP saliency (see snip_selection.py) measures |grad * weight| summed
    per block -- roughly, "how much would the task loss change if this block's
    weights were perturbed". Filter-block selection already uses the LOWEST-
    saliency blocks (keep='low': the least task-sensitive / most redundant blocks
    are the ones considered safe to replace). This function does the opposite at
    the OTHER end of that same ranking: among the blocks that keep their original
    weights (i.e. get LoRA rather than being replaced), it identifies the
    num_blocks blocks the task is MOST sensitive to.

    Returned indices are meant to be passed as inject_lora's ortho_block_indices,
    so those specific blocks' LoRA adapters get the orthogonality regularizer
    (encouraging their limited rank-r capacity to be used non-redundantly, since
    perturbing these blocks matters most to the task) while every other LoRA
    block is left as plain (unregularized) LoRA.

    Returns indices sorted in INCREASING order (cosmetic only -- inject_lora
    doesn't care about order, this just makes printed/logged output deterministic
    and readable).
    """
    excluded = _normalize_indices(excluded_indices)
    candidates = [(idx, score) for idx, score in saliencies.items() if idx not in excluded]
    if num_blocks > len(candidates):
        print(f"[SFP] Warning: requested num_blocks={num_blocks} for orthogonality selection exceeds "
              f"the number of LoRA-eligible blocks ({len(candidates)}); clamping to {len(candidates)}.")
        num_blocks = len(candidates)
    if num_blocks <= 0:
        return []
    top = sorted(candidates, key=lambda kv: -kv[1])[:num_blocks]
    return sorted(idx for idx, _ in top)


def inject_lora(
    model: nn.Module,
    excluded_block_indices,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.0,
    target_keywords: list = ["qkv", "proj", "fc1", "fc2"],
    ortho_block_indices=None,
    adapter_type: str = "lora",
    init_method: str = "default",
    loftq_bits: int = 4,
    loftq_iters: int = 5,
) -> int:
    """
    Wraps target linear layers with LoRALinear in every block EXCEPT those in
    excluded_block_indices (the filter-substituted blocks). Returns total number
    of adapter parameters injected (LoRA/DoRA low-rank params, plus DoRA's
    per-layer magnitude vector if adapter_type="dora").

    ortho_block_indices: None (default) -> every injected LoRALinear gets
    apply_ortho=True (the original, non-selective behavior: the orthogonality
    regularizer, if enabled via nonzero lambda1/lambda2, applies uniformly to
    every LoRA layer in the model).

    ortho_block_indices: an iterable of block indices (e.g. from
    select_top_sensitive_blocks) -> ONLY LoRALinear layers inside those specific
    blocks get apply_ortho=True; every other LoRA-wrapped block gets
    apply_ortho=False, i.e. becomes plain (unregularized) LoRA regardless of the
    lambda values passed to compute_lora_orthogonality_loss. Pass an empty list to
    make every LoRA layer plain (equivalent to lambda1=lambda2=0, but keeps LoRA
    itself active).

    adapter_type / init_method / loftq_bits / loftq_iters: see LoRALinear's
    docstring. Applied identically to every injected layer.
    """
    excluded = _normalize_indices(excluded_block_indices)
    ortho_blocks = None if ortho_block_indices is None else _normalize_indices(ortho_block_indices)
    lora_params = 0
    for idx, block in enumerate(model.blocks):
        if idx in excluded:
            continue
        apply_ortho_here = True if ortho_blocks is None else (idx in ortho_blocks)
        for name, module in list(block.named_modules()):
            if any(kw in name for kw in target_keywords) and isinstance(module, nn.Linear):
                parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = block if parent_name == "" else block.get_submodule(parent_name)

                lora_layer = LoRALinear(module, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout,
                                         apply_ortho=apply_ortho_here, adapter_type=adapter_type,
                                         init_method=init_method, loftq_bits=loftq_bits, loftq_iters=loftq_iters)
                setattr(parent, attr_name, lora_layer)
                lora_params += lora_rank * (module.in_features + module.out_features)
                if adapter_type == "dora":
                    lora_params += module.out_features  # the magnitude vector
    return lora_params


def freeze_non_trainable(model: nn.Module, filter_block_indices) -> int:
    """
    Sets requires_grad=True for LoRA params, filter-block params (any block index
    in filter_block_indices), LayerNorm params, and the classifier head;
    requires_grad=False for everything else. Returns total LayerNorm param count
    (for logging). LN unfreezing follows the paper's ablation (Table 4): substitution
    induces activation-distribution shifts, and retraining LN params is needed to
    realign feature geometry.
    """
    indices = _normalize_indices(filter_block_indices)

    ln_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            for p in module.parameters():
                ln_params += p.numel()

    for name, param in model.named_parameters():
        if (
            "lora_" in name
            or any(f"blocks.{idx}" in name for idx in indices)
            or "head" in name
            or "norm" in name.lower()
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False

    return ln_params


def apply_single_filter_and_lora(
    model: nn.Module,
    pruned_block_idx: int,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.0,
    filter_num_layers: int = 1,
    filter_dropout: float = 0.0,
    filter_residual_hidden_dim: int = 0,
    filter_residual_alpha: float = 1.0,
    filter_residual_dropout: float = 0.0,
    target_keywords: list = ["qkv", "proj", "fc1", "fc2"],
    ortho_block_indices=None,
    adapter_type: str = "lora",
    init_method: str = "default",
    loftq_bits: int = 4,
    loftq_iters: int = 5,
):
    """
    Backward-compatible convenience wrapper for the SINGLE-block case, built on top
    of substitute_filter_block / inject_lora / freeze_non_trainable. Behavior is
    unchanged from all previous versions of this function when filter_num_layers=1
    and filter_residual_hidden_dim=0 (both defaults) -- both are new, letting the
    filter block use more than one (purely linear) layer and/or a nonlinear
    zero-init residual branch; see MultiLayerFilterBlock's and FilterResidualMLP's
    docstrings for details.

    ortho_block_indices: see inject_lora's docstring -- None (default) applies the
    orthogonality regularizer to every LoRA layer if lambda1/lambda2 are nonzero
    (unchanged old behavior); pass a specific set of block indices (e.g. from
    select_top_sensitive_blocks) to restrict it to only those blocks' LoRA layers.

    adapter_type / init_method / loftq_bits / loftq_iters: see LoRALinear's
    docstring -- adapter_type="lora"/init_method="default" (both defaults)
    reproduce all prior behavior exactly.

    Does NOT perform the pseudoinverse init itself -- caller still does that
    afterward via filter_block.init_from_pinv(X_in, X_out), exactly as before.

    For multi-block runs (more than one filter-substituted block), don't use this
    function -- call substitute_filter_block / inject_lora / freeze_non_trainable
    directly in a loop instead (see train_sfp_lora.py's main() for the pattern).
    """
    filter_block = substitute_filter_block(
        model, pruned_block_idx, num_layers=filter_num_layers, dropout=filter_dropout,
        residual_hidden_dim=filter_residual_hidden_dim,
        residual_alpha=filter_residual_alpha, residual_dropout=filter_residual_dropout,
    )
    lora_params = inject_lora(model, pruned_block_idx, lora_rank, lora_alpha, lora_dropout, target_keywords,
                               ortho_block_indices=ortho_block_indices, adapter_type=adapter_type,
                               init_method=init_method, loftq_bits=loftq_bits, loftq_iters=loftq_iters)
    ln_params = freeze_non_trainable(model, pruned_block_idx)

    layer_desc = "Single" if filter_num_layers <= 1 else f"{filter_num_layers}-Layer"
    residual_desc = f" + residual(hidden={filter_residual_hidden_dim})" if filter_residual_hidden_dim > 0 else ""
    print(f"[SFP-SingleFilter] Substituted block {pruned_block_idx} with {layer_desc} Filter Block{residual_desc}.")
    print(f"[SFP-SingleFilter] Injected {lora_params:,} {adapter_type.upper()} parameters "
          f"(rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}, init={init_method}).")
    print(f"[SFP-SingleFilter] Unfroze {ln_params:,} LayerNorm parameters across all blocks.")
    return filter_block