import time

# config for training GPT-2 (124M) baseline model with no RMSNorm parameters
# 2 epoch ablation study to test RMSNorm parameter effectiveness

wandb_log = True
init_from = 'scratch'
wandb_project = 'nano-moe'
wandb_run_name = 'two-epoch-ablation-' + time.strftime('%Y-%m-%d-%H:%M:%S')

# model/moe settings
stride = 100 # disable moe

# use smaller GPT model
n_layer = 8
n_head = 4 
n_embd = 512

# batch settings
batch_size = 128
block_size = 512
gradient_accumulation_steps = 1

# epoch-based training - 2 epochs for ablation
num_epochs = 2.0
evals_per_epoch = 20
warmup_frac = 0.03
decay_frac = 0.2751957541078461

# eval stuff
eval_iters = 100
log_interval = 50

# optimizer settings
weight_decay = 0.1

# muon optimizer learning rates
adam_lr = 0.0007152454333036531 # learning rate for Adam params (gains/biases + non-hidden)
muon_lr = 0.0007152454333036531 * 67 # learning rate for Muon params (hidden weights)