import time

# config for training GPT-2 (124M) baseline model (one expert) on two RTX 3090 GPUs
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=2 train.py config/train_gpt2_moe_baseline.py

wandb_log = True
init_from = 'resume' #'scratch'
wandb_project = 'nano-moe'
#wandb_run_name='gpt2-124M-owt ' + time.strftime('%Y-%m-%d %H:%M:%S')
wandb_run_name = 'gpt2-124M-owt-restart-0 2024-04-02 14:39:04'

# model/moe settings
n_exp = 1

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 25B
max_iters = 50000
lr_decay_iters = 50000

# eval stuff
eval_interval = 500
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
