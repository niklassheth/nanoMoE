"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn import functional as F

from manager import MANAGER

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Router(nn.Module):
    def __init__(self, config):
        super().__init__()

        # router settings
        self.top_k = config.top_k
        self.n_exp = config.n_exp
        assert self.top_k >= 1 and self.top_k <= config.n_exp
        self.use_noisy_top_k = config.use_noisy_top_k
        self.train_capacity = config.train_capacity
        self.eval_capacity = config.eval_capacity
        self.min_capacity = config.min_capacity
        self.router_use_full_prec = config.router_use_full_prec

        # auxiliary / load balancing loss settings
        self.use_aux_loss = config.use_aux_loss
        self.use_router_z_loss = config.use_router_z_loss

        # linear projection for (noisy) softmax gating
        # no bias is used, see page 4 eq (4) in (https://arxiv.org/abs/1701.06538)
        self.w_g = nn.Linear(config.n_embd, config.n_exp, bias=False)
        self.w_noise = nn.Linear(config.n_embd, config.n_exp, bias=False) if self.use_noisy_top_k else None
    def forward(self, x):
        """
        Computes routing information for tokens, including which experts to use,
        the weights for their outputs, and their position within the expert's batch.
        This implementation is memory-efficient and avoids quadratic scaling with batch size.
        """
        # The router can be sensitive to precision issues, so we can run it in full float32.
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        ctx = nullcontext() if not self.router_use_full_prec else torch.amp.autocast(device_type=device_type, enabled=False)

        with ctx:
            B, T, C = x.size()
            num_tokens = B * T
            x_flat = x.view(num_tokens, C)

            # 1. GET ROUTING LOGITS
            # ---------------------
            logits = self.w_g(x_flat)  # [B*T, n_exp]
            if self.training and self.use_noisy_top_k:
                noise = F.softplus(self.w_noise(x_flat))
                noise *= torch.randn_like(noise)
                logits += noise

            # 2. COMPUTE LOSSES (if training)
            # -------------------------------
            if self.training:
                # Router Z-loss prevents logits from growing too large
                if self.use_router_z_loss:
                    z_loss = self.compute_router_z_loss(logits.view(B, T, -1))
                    MANAGER.add_router_z_loss(z_loss)
                
                # Find top-k choices for each token
                top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1) # [B*T, k]
                
                # The auxiliary loss encourages load balancing across experts
                if self.use_aux_loss:
                    # To compute aux loss, we need the full probability distribution,
                    # not just for the top k. We can create this sparsely.
                    all_probs = torch.zeros_like(logits)
                    all_probs.scatter_(-1, top_k_indices, F.softmax(top_k_logits, dim=-1))
                    aux_loss = self.compute_aux_loss(all_probs.view(B, T, -1), top_k_indices.view(B, T, -1))
                    MANAGER.add_aux_loss(aux_loss)
            else:
                 # At inference, we just need the top-k
                 top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)

            # 3. COMPUTE ROUTER PROBABILITIES
            # --------------------------------
            # We normalize the probabilities over the top-k experts
            router_probs = F.softmax(top_k_logits, dim=-1) # [B*T, k]

            # 4. DETERMINE TOKEN RANKS WITH CAPACITY LIMITING
            # -----------------------------------------------
            exp_capacity = self.get_capacity(num_tokens)
            
            # Create a one-hot mask of the chosen experts for each token. Shape: [B*T, k, n_exp]
            expert_mask_one_hot = F.one_hot(top_k_indices, num_classes=self.n_exp)

            # This is the critical step to ensure load balancing prioritizes top-1 experts.
            # We flatten the k dimension first, so cumsum processes all top-1 choices, then all top-2, etc.
            # This is the memory-efficient equivalent of the original logic.
            reshaped_mask = expert_mask_one_hot.permute(1, 0, 2).reshape(self.top_k * num_tokens, self.n_exp)
            cumulative_sum = torch.cumsum(reshaped_mask, dim=0)
            
            # Reshape back to the original layout
            position_in_expert = cumulative_sum.reshape(self.top_k, num_tokens, self.n_exp).permute(1, 0, 2)
            
            # The rank is the position, but we only care about the rank for the selected expert.
            # We multiply by the one-hot mask to zero out positions for non-selected experts.
            rank = (position_in_expert - 1) * expert_mask_one_hot
            
            # 5. GENERATE FINAL MASKS AND RANKS FOR THE MOE LAYER
            # ----------------------------------------------------
            # Create a mask to drop tokens that exceed the expert's capacity
            capacity_mask = rank < exp_capacity

            # The final expert mask includes both the expert choice and the capacity check.
            final_expert_mask = expert_mask_one_hot * capacity_mask # [B*T, k, n_exp]
            
            # Router probabilities are also masked. If a token is dropped, its probability is zero.
            # We check if the token was assigned to any expert in its k-th slot.
            probs_mask = (final_expert_mask.sum(dim=-1) > 0) # [B*T, k]
            router_probs_masked = router_probs * probs_mask

            # The final rank is collapsed to a single value per top-k choice
            final_rank = torch.sum(rank, dim=-1) # [B*T, k]

            # The MOELayer will use these tensors to efficiently dispatch and combine tokens.
            # Their memory usage all scale linearly with (B * T).
            return final_expert_mask, router_probs_masked, top_k_indices, final_rank 
    
    def compute_aux_loss(self, expert_probs: torch.Tensor, indices: torch.Tensor):
        """
        Computes Switch Transformer auxiliary loss (https://arxiv.org/abs/2101.03961)
        See equations (4)-(6) on page 7
        """

        # equation (5): compute ratio of tokens allocated to each expert
        # total number of tokens is defined as total tokens in batch * k
        # (k = 1) for the Switch Transformer
        with torch.no_grad():
            one_hot_indices = F.one_hot(indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
            one_hot_indices = torch.sum(one_hot_indices.float(), dim=2)  # [B, T, n_exp] (sum over k dimension)
            tokens_per_expert = torch.mean(one_hot_indices.float(), dim=(0, 1))

        # equation (6): compute ratio of router probability allocated to each expert
        prob_per_expert = torch.mean(expert_probs.float(), dim=(0, 1))

        # equation (4): take a scaled dot product between prob/token allocation vectors
        # multiply the result by the number of experts
        return self.n_exp * torch.sum(prob_per_expert * tokens_per_expert)
    
    def compute_router_z_loss(self, logits: torch.Tensor):
        """
        Computes ST-MoE router z loss (https://arxiv.org/abs/2202.08906)
        See equation (5) on page 7
        """
    
        # exponentiate logits, sum logits of each expert, take log, and square
        # code below is the same as:
        # > z_loss = torch.exp(logits)
        # > z_loss = torch.sum(z_loss, dim=-1)
        # > z_loss = torch.log(z_loss) ** 2.0
        z_loss = torch.logsumexp(logits, dim=-1) ** 2.0  # [B, T, n_exp]

        # sum over all tokens and divide by total number of tokens
        return torch.mean(z_loss)

    def get_capacity(self, tokens_per_batch):
        # expert capacity is given by (tokens_per_batch / num_experts) * capacity_factor
        # see eq (3) in Switch Transformer (https://arxiv.org/abs/2101.03961)
        capacity_factor = self.train_capacity if self.training else self.eval_capacity
        capacity = math.floor(self.top_k * capacity_factor * tokens_per_batch / self.n_exp)
        capacity += capacity % 2 # make sure capacity is an even number
        capacity = max(capacity, self.min_capacity) # use min capacity
        assert capacity > 0
        return int(capacity)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class MLPExperts(nn.Module):
    """
    implementation of multiple MLP-based experts that can process input
    in batch -- based upon ColossalAI OpenMoE but simple, has optional bias, and
    uses a bmm instead of a loop over a mm for each expert to improve efficiency
    link: https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/moe/experts.py
    """
    def __init__(self, config):
        # TODO: add param init
        super().__init__()
        self.bias = config.bias

        self.c_fc = nn.Parameter(torch.empty(config.n_exp, config.n_embd, 4 * config.n_embd))
        self.c_proj = nn.Parameter(torch.empty(config.n_exp, 4 * config.n_embd, config.n_embd))
        self.fc_bias = nn.Parameter(torch.empty(config.n_exp, 1, 4 * config.n_embd)) if self.bias else None
        self.proj_bias = nn.Parameter(torch.empty(config.n_exp, 1, config.n_embd)) if self.bias else None
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
    

    def forward(self, x):
        x = torch.bmm(x, self.c_fc)
        if self.bias:
            x += self.fc_bias
        x = self.gelu(x)
        x = torch.bmm(x, self.c_proj)
        if self.bias:
            x += self.proj_bias
        x = self.dropout(x)
        return x

class MOELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = Router(config)
        self.experts = MLPExperts(config)
        self.n_exp = config.n_exp
        self.top_k = config.top_k

    def forward(self, x: torch.Tensor):
        B, T, C = x.size() # Keep track of original shape

        # --- Get routing information ---
        # Call the router with the ORIGINAL 3D tensor. The router will handle flattening internally
        # and return routing info shaped for a flattened list of tokens.
        expert_mask, router_probs, top_k_indices, rank = self.router(x)
        # expert_mask: [B*T, k, n_exp], router_probs: [B*T, k], etc.

        # Now, flatten the input tensor for the dispatch operation
        x_flat = x.view(B * T, C)

        # --- Dispatch tokens to experts (the "scatter" part) ---
        exp_capacity = self.router.get_capacity(B * T)
        expert_inputs = torch.zeros(self.n_exp, exp_capacity, C, dtype=x.dtype, device=x.device)

        # Get the indices for the valid assignments that are within capacity
        flat_top_k_indices = top_k_indices.view(-1)
        flat_rank = rank.view(-1)
        flat_token_indices = torch.arange(B * T, device=x.device).repeat_interleave(self.top_k)

        valid_mask = flat_rank < exp_capacity
        valid_token_indices = flat_token_indices[valid_mask]
        valid_expert_indices = flat_top_k_indices[valid_mask]
        valid_ranks = flat_rank[valid_mask]

        # Use advanced indexing to place tokens from the flattened input into the expert buffer
        expert_inputs[valid_expert_indices, valid_ranks] = x_flat[valid_token_indices]

        # --- Run experts ---
        expert_outputs = self.experts(expert_inputs) # [n_exp, exp_capacity, C]

        # --- Combine expert outputs (the "gather" part) ---
        output_flat = torch.zeros_like(x_flat)

        # Retrieve the expert outputs using the same valid indices
        gated_expert_outputs = expert_outputs[valid_expert_indices, valid_ranks]

        # Filter router probabilities for valid assignments
        valid_router_probs = router_probs.view(-1)[valid_mask].unsqueeze(1)

        # Weight the expert outputs by the router probabilities
        weighted_outputs = gated_expert_outputs * valid_router_probs

        # Use scatter_add_ to combine outputs for tokens sent to multiple experts (k > 1)
        # This adds the weighted outputs back to their original token positions in the flattened output tensor.
        output_flat.scatter_add_(0, valid_token_indices.unsqueeze(1).expand_as(weighted_outputs), weighted_outputs)

        # Reshape output back to the original input shape
        return output_flat.view(B, T, C)

class Block(nn.Module):

    def __init__(self, config, use_moe=False):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        if use_moe:
            self.mlp = MOELayer(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    # MoE-related configs 
    n_exp: int = 1 # if n_exp = 1 we just use regular MLP layers
    top_k: int = 2
    use_aux_loss: bool = False # apply auxiliary loss (from Switch Transformer) in router
    use_router_z_loss: bool = False # apply router z loss (from ST-MoE)
    use_noisy_top_k: bool = False
    aux_loss_weight: float = 0.01 # default setting from Switch Transformer (see top of page 8)
    router_z_loss_weight: float = 0.001 # default setting from ST-MoE (see page 8 eq. 6)
    train_capacity: float = 1.25  # default setting from ST-MoE (see top of page 6)
    eval_capacity: float = 2.0
    min_capacity: int = 4  # minimum batch size to send to any single expert
    stride: int = 2 # one in every stride layers are converted to an MoE
    use_switch_tfm_init: bool = False  # use weight init scheme from Switch Transformer
    switch_tfm_init_scale: float = 1.0
    router_use_full_prec: bool = False  # use float32 precision in the router


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        if config.n_exp == 1:
            # create normal transformer blocks
            blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        else:
            # create transformer blocks, placing an MoE block every <stride> layers
            blocks = []
            for i in range(config.n_layer):
                # TODO: how to implement this?
                # should we change below to i + 1 ?
                use_moe = ((i + 1) % config.stride) == 0
                blocks.append(Block(config, use_moe=use_moe))
            blocks = nn.ModuleList(blocks)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = blocks,
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        # optionall use switch transformer special init scheme for experts
        # See pg. 10 here: https://arxiv.org/abs/2101.03961
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('experts.c_proj'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    @torch.no_grad()
    def _init_weights(self, module):
        # optionally use switch transformer-style initialization
        # see page 10 for switch init explanation: https://arxiv.org/abs/2101.03961
        if isinstance(module, nn.Linear):
            if self.config.use_switch_tfm_init:
                scale = self.config.switch_tfm_init_scale

                # linear layers have flipped dimensions in torch
                # size of weights is [out_dim, in_dim] 
                w_fan_in = module.weight.shape[-1]
                w_std = (scale / w_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=w_std,
                    a=-2*w_std,
                    b=2*w_std,
                )
            else:
                # perform standard (normal) initialization of weights
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            # always initialize bias to zero
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, MLPExperts):
            # we have to init expert weights manually because
            # nn.Parameter is not a type of module in torch
            if self.config.use_switch_tfm_init:
                scale = self.config.switch_tfm_init_scale

                c_fc_fan_in = module.c_fc.shape[-2]
                c_fc_std = (scale / c_fc_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.c_fc,
                    mean=0.0,
                    std=c_fc_std,
                    a=-2*c_fc_std,
                    b=2*c_fc_std,
                )

                c_proj_fan_in = module.c_proj.shape[-2]
                c_proj_std = (scale / c_proj_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.c_proj,
                    mean=0.0,
                    std=c_proj_std,
                    a=-2*c_proj_std,
                    b=2*c_proj_std,
                )
            else:
                # perform standard (normal) initialization of weights
                torch.nn.init.normal_(module.c_fc, mean=0.0, std=0.02)
                torch.nn.init.normal_(module.c_proj, mean=0.0, std=0.02)

            # bias is always initialized to zero
            if module.fc_bias is not None:
                torch.nn.init.zeros_(module.fc_bias)
                torch.nn.init.zeros_(module.proj_bias)
        elif isinstance(module, nn.Embedding):
            # just use standard initialization scheme for embedding always
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            # add the auxiliary load balancing loss and router z loss to the main loss
            if self.config.n_exp > 1 and self.config.use_aux_loss:
                loss += self.config.aux_loss_weight * MANAGER.aggregate_aux_loss()
                MANAGER.reset_aux_loss()
            if self.config.n_exp > 1 and self.config.use_router_z_loss:
                loss += self.config.router_z_loss_weight * MANAGER.aggregate_router_z_loss()
                MANAGER.reset_router_z_loss()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert not 'moe' in model_type, "Pretrained checkpoints not available for MoE"
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # TODO: add expert config
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        # add an extra check for "bias" string to account for bias terms in MoE layers
        decay_params = [p for n, p in param_dict.items() if (p.dim() >= 2 and not n.endswith('bias'))]
        nodecay_params = [p for n, p in param_dict.items() if (p.dim() < 2 or n.endswith('bias'))]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of GPU bfloat16 -> fp32 accum peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        # Determine the theoretical peak FLOPs of the current device using a simple lookup.
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0).lower()

            # Very small lookup table of common GPUs and their BF16/FP16 peak throughput (in FLOPs).
            # TODO: add more GPUs
            flops_table = {
                "3090": 71e12,   # RTX 3090
                "4090": 165e12,  # RTX 4090
                "l40s": 362e12,  # L40S
                "a100": 312e12,  # A100 80GB
                "h100": 990e12,  # H100
                "5070 ti": 176e12,  # RTX 5070 Ti
                "5080": 225e12,  # RTX 5080
                "b200": 2250e12,  # B200
            }

            # Pick the first entry whose key is a substring of the device name; fall back to 0.
            flops_promised = next((v for k, v in flops_table.items() if k in device_name), 0)
        else:
            # If running on CPU or an unknown accelerator, return -1 
            flops_promised = -1
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx   
