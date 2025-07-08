import time

# config for training GPT-2 (124M) baseline model (one expert) on two RTX 3090 GPUs
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=2 train.py config/train_nano_moe.py

wandb_log = True
init_from = 'scratch'
wandb_project = 'nano-moe'
wandb_run_name ='tinystories-moe' + time.strftime('%Y-%m-%d %H:%M:%S')

# model/moe settings
stride = 100 #disable moe

# use smaller GPT model
n_layer = 8
n_head = 4 # 128 head size works better with qknorm (?)
n_embd = 512

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 128
block_size = 512
gradient_accumulation_steps = 1

# epoch-based training
num_epochs = 10.0
evals_per_epoch = 20
warmup_frac = 0.01
decay_frac = 0.1

# eval stuff
eval_iters = 100
log_interval = 50 # slow as balls

# weight decay
weight_decay = 1e-1
learning_rate = 1.1e-3
#grad_clip = 1.0
