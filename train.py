"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['NCCL_IGNORE_DISABLED_P2P'] = '1'
import time
import math
import pickle
from contextlib import nullcontext
from tqdm import tqdm

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.profiler import profile, record_function, ProfilerActivity

from model import GPTConfig, GPT
from data.tinystories.dataloader import get_dataloader

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) with epoch-based training
# I/O
out_dir = 'out'
log_interval = 25
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'gpt2*'

# wandb logging
wandb_log = True # False # disabled by default
wandb_project = 'nano-moe'
wandb_run_name = 'gpt2-124M-owt' + str(time.time())

# data
dataset = 'tinystories'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
bias = False # do we use bias inside LayerNorm and Linear layers?

# moe
n_exp = 1 # if n_exp = 1 we just use regular MLP layers
top_k = 2
use_aux_loss = False
use_router_z_loss = False
use_noisy_top_k = False
aux_loss_weight = 0.001
router_z_loss_weight = 0.01
train_capacity = 1.25
eval_capacity = 2.0
min_capacity = 4
stride = 2
use_switch_tfm_init = False
switch_tfm_init_scale = 1.0  # recommended 0.1 for stability (pg.10, https://arxiv.org/abs/2101.03961)
router_use_full_prec = False

# adamw optimizer
learning_rate = 6e-4 # max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.0 # clip gradients at this value, or disable if == 0.0

# epoch-based training
num_epochs = 1.0  # total number of epochs to train (can be fractional)
evals_per_epoch = 10  # number of evaluations per epoch
warmup_frac = 0.01  # fraction of total steps used for warmup
decay_frac = 0.1    # fraction of total steps used for final decay

# learning rate schedule
decay_lr = True  # whether to use the warmup/stable/decay schedule

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

# profiling
use_profiler = False # enable PyTorch profiler
profiler_schedule_wait = 2 # number of steps to wait before profiling
profiler_schedule_warmup = 2 # number of warmup steps
profiler_schedule_active = 6 # number of active profiling steps
profiler_schedule_repeat = 1 # number of times to repeat the schedule
profiler_output_dir = './profiler_results' # directory to save profiler results
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# Remove non-existent variables that were removed during epoch-based conversion
config_keys = [k for k in config_keys if k not in ['max_iters', 'lr_decay_iters', 'eval_interval']]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
print(config)
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data loading
data_dir = os.path.join('data', dataset)
num_workers = min(4, os.cpu_count() or 1)

# Initialize DataLoaders
train_loader = get_dataloader(
    data_dir=data_dir,
    split='train',
    batch_size=batch_size,
    block_size=block_size,
    num_workers=num_workers,
    shuffle=True
)

val_loader = get_dataloader(
    data_dir=data_dir,
    split='validation',
    batch_size=batch_size,
    block_size=block_size,
    num_workers=0,  # Keep validation single-threaded for deterministic results
    shuffle=False
)

# Calculate epoch parameters
iters_per_epoch = len(train_loader)
total_iters = int(num_epochs * iters_per_epoch)
warmup_iters = int(warmup_frac * total_iters)
decay_iters = int(decay_frac * total_iters)
decay_start = total_iters - decay_iters
eval_every_n_iters = max(1, iters_per_epoch // evals_per_epoch)

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
tokens_per_epoch = tokens_per_iter * iters_per_epoch

print(f"Epoch configuration:")
print(f"  Iterations per epoch: {iters_per_epoch}")
print(f"  Num epochs: {num_epochs}")
print(f"  Total iterations: {total_iters}")
print(f"  Warmup iters: {warmup_iters}")
print(f"  Decay iters: {decay_iters}")
print(f"  Evaluations per epoch: {evals_per_epoch}")
print(f"  Evaluate every {eval_every_n_iters} iterations")
print(f"  Tokens per iteration: {tokens_per_iter:,}")
print(f"  Tokens per epoch: {tokens_per_epoch:,}")



# training state
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, n_exp=n_exp, top_k=top_k,
                  use_aux_loss=use_aux_loss, use_router_z_loss=use_router_z_loss,
                  use_noisy_top_k=use_noisy_top_k, aux_loss_weight=aux_loss_weight,
                  router_z_loss_weight=router_z_loss_weight, train_capacity=train_capacity,
                  eval_capacity=eval_capacity, min_capacity=min_capacity, stride=stride,
                  use_switch_tfm_init=use_switch_tfm_init, switch_tfm_init_scale=switch_tfm_init_scale,
                  router_use_full_prec=router_use_full_prec) # start with model_args from command line
print('\n\n')
print(model_args)
print('\n\n')
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict()
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# estimate validation loss using many batches
@torch.no_grad()
def estimate_loss():
    model.eval()
    
    val_losses = torch.zeros(eval_iters)
    val_iter = iter(val_loader)
    
    for k in range(eval_iters):
        try:
            X, Y = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            X, Y = next(val_iter)
        
        # Move to device
        if device_type == 'cuda':
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
        else:
            X, Y = X.to(device), Y.to(device)
            
        with ctx:
            _, loss = model(X, Y)
        val_losses[k] = loss.item()
    
    model.train()
    return val_losses.mean()

# learning rate scheduler (warmup -> stable -> decay to zero)
def get_lr(it: int) -> float:
    """Compute learning rate at iteration it."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / float(warmup_iters + 1)
    if it < decay_start:
        return learning_rate
    if it >= total_iters:
        return 0.0
    decay_ratio = (it - decay_start) / float(max(1, decay_iters))
    return learning_rate * (1 - math.sqrt(decay_ratio))

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    wandb.define_metric("train/*", step_metric="tokens_seen")
    wandb.define_metric("val/*", step_metric="tokens_seen")

# training loop
t0 = time.time()
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
running_tokens_per_sec = -1.0  # EMA of tokens processed per second

# Initialize profiler if enabled
profiler = None
if use_profiler and master_process:
    os.makedirs(profiler_output_dir, exist_ok=True)
    activities = [ProfilerActivity.CPU]
    if device_type == 'cuda':
        activities.append(ProfilerActivity.CUDA)
    
    profiler = profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=profiler_schedule_wait,
            warmup=profiler_schedule_warmup,
            active=profiler_schedule_active,
            repeat=profiler_schedule_repeat
        ),
        on_trace_ready=lambda trace: (
            torch.profiler.tensorboard_trace_handler(profiler_output_dir)(trace),
            trace.export_chrome_trace(os.path.join(profiler_output_dir, f"trace_{int(time.time())}.json"))
        )[-1],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
    profiler.start()

global_iter = 0
for epoch in range(math.ceil(num_epochs)):
    if master_process:
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
    else:
        pbar = train_loader
    
    for batch_idx, (X, Y) in enumerate(pbar):
        if global_iter >= total_iters:
            break
        grad_norm = None  # Initialize gradient norm for logging
        
        # Move to device
        if device_type == 'cuda':
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
        else:
            X, Y = X.to(device), Y.to(device)

        # determine and set the learning rate for this iteration
        lr = get_lr(global_iter) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if global_iter > 0 and global_iter % eval_every_n_iters == 0 and master_process:
            val_loss = estimate_loss()
            print(f"epoch {epoch + 1}, step {global_iter}: val loss {val_loss:.4f}")
            if wandb_log:
                wandb.log({
                    "val/loss": val_loss,
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                    "tokens_seen": global_iter * batch_size * block_size,
                }, step=global_iter)
            if val_loss < best_val_loss or always_save_checkpoint:
                best_val_loss = val_loss
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'model_args': model_args,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        if eval_only and epoch == 0 and batch_idx == 0:
            # Run one evaluation then exit
            val_loss = estimate_loss()
            print(f"eval_only mode: val loss {val_loss:.4f}")
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        
        # Handle DDP gradient sync - only sync when we're about to step the optimizer
        if ddp:
            model.require_backward_grad_sync = (global_iter % gradient_accumulation_steps == gradient_accumulation_steps - 1)
        
        # Forward and backward pass for this batch
        with record_function("forward_backward"):
            with ctx:
                with record_function("forward"):
                    logits, loss = model(X, Y)
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            
            # backward pass, with gradient scaling if training in fp16
            with record_function("backward"):
                scaler.scale(loss).backward()
        
        # Only step optimizer every gradient_accumulation_steps iterations
        if (global_iter + 1) % gradient_accumulation_steps == 0:
            with record_function("optimizer_step"):
                # disable gradient clipping for now
                #if grad_clip != 0.0:
                #    scaler.unscale_(optimizer)
                #    # Store gradient norm tensor for later logging (avoid .item() sync here)
                #    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                # step the optimizer and scaler if training in fp16
                scaler.step(optimizer)
                scaler.update()
                # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if global_iter % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            grad_normf = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else None
            # compute tokens per second for this iteration (raw)
            tokens_ps = (batch_size * block_size) / dt if dt > 0 else 0.0

            # update running averages once the loop has warmed up a few iterations (parallels MFU logic)
            if global_iter >= 5:  # let the training loop settle a bit before smoothing
                running_tokens_per_sec = tokens_ps if running_tokens_per_sec == -1.0 else 0.9 * running_tokens_per_sec + 0.1 * tokens_ps
                mfu = raw_model.estimate_mfu(batch_size, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            
            # Update tqdm progress bar with loss, tok/s, and MFU
            pbar.set_postfix({
                'loss': f'{lossf:.4f}',
                'tok/s': tqdm.format_sizeof(running_tokens_per_sec, divisor=1000),
                'mfu': f'{running_mfu*100:.2f}%'
            })
            
            if wandb_log:
                wandb.log({
                    "train/loss_step": lossf,
                    "train/grad_norm": grad_normf,
                    "lr": lr,
                    "mfu": running_mfu*100,
                    "tok_per_sec": running_tokens_per_sec,
                    "time_ms": dt*1000,
                    "tokens_seen": global_iter * batch_size * block_size,
                }, step=global_iter)
        
        # Profiler step
        if profiler is not None:
            profiler.step()

        global_iter += 1

    if global_iter >= total_iters:
        break

# Stop profiler if it was started
if profiler is not None:
    profiler.stop()
    print(f"Profiler results saved to {profiler_output_dir}")

if ddp:
    destroy_process_group()
