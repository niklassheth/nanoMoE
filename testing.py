import torch
from model import Router, SparseMOE, GPTConfig, MOEConfig

inp = torch.randn(16, 1024, 768)

gpt_cfg = GPTConfig()
moe_cfg = MOEConfig(n_exp=4, top_k=2, use_aux_loss=True, use_noisy_top_k=False)

route = Router(gpt_config=gpt_cfg, moe_config=moe_cfg)
route(inp)