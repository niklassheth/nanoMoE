import time

# config for training GPT-2 (124M) baseline model (one expert) on two RTX 3090 GPUs
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=2 train.py config/train_nano_moe.py

wandb_log = True
init_from = 'scratch'
wandb_project = 'nano-moe'
wandb_run_name ='tinystories-moe' + time.strftime('%Y-%m-%d %H:%M:%S')

# model/moe settings
n_exp = 4
top_k = 2
use_aux_loss = True
aux_loss_weight = 0.01
use_router_z_loss = True
router_z_loss_weight = 0.001
use_noisy_top_k = False
train_capacity = 1.25
eval_capacity = 2.0
stride = 2
use_switch_tfm_init = True
router_use_full_prec = True

# use smaller GPT model
n_layer = 8
n_head = 8
n_embd = 512

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 32
block_size = 512
gradient_accumulation_steps = 2

# epoch-based training
max_epochs = 1
evals_per_epoch = 10  # evaluate 10 times per epoch

# eval stuff
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1