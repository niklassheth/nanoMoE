"""
Simple and Fast TinyStories DataLoader v2
Clean, minimal approach following nanogpt style
"""

import os
import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
from multiprocessing import cpu_count
import numpy as np
from tqdm import tqdm
import pickle






def prepare_data(data_dir=None, tokenizer_repo="niklassheth/tinystories-tokenizer"):
    """
    Download, tokenize, and save TinyStories dataset
    Run with: python dataloader_v2.py
    
    Args:
        data_dir: Directory to save dataset files
        tokenizer_repo: Hugging Face tokenizer repository name
    """

    if data_dir is None:
        data_dir = os.path.dirname(__file__)
    
    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories")
    
    # Initialize custom tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
    eos_token = tokenizer.eos_token_id  # End of sequence token
    
    def process_split(split_name: str, split_data):
        print(f"Processing {split_name} split...")
        
        # Filter out very short stories
        print("Filtering short stories...")
        cores = min(cpu_count(), 16)
        filtered = split_data.filter(
            lambda x: len(x["text"]) > 32, 
            num_proc=cores
        )
        
        
        # Tokenize each story and track lengths (like openwebtext)
        def tokenize_story(example):
            text = example['text']
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(eos_token)  # Add EOT after each story
            return {'tokens': tokens, 'len': len(tokens)}
        
        print("Tokenizing stories...")
        tokenized = filtered.map(
            tokenize_story,
            remove_columns=['text'],
            num_proc=cores
        )
        
        # Now efficiently concatenate using numpy (like openwebtext approach)
        print("Concatenating token sequences...")
        arr_len = np.sum(tokenized['len'], dtype=np.uint64)
        print(f"{split_name} will have {arr_len:,} tokens")
        
        # Create binary file and write in batches for memory efficiency
        bin_path = os.path.join(data_dir, f'{split_name}.bin')
        dtype = np.uint16
        arr = np.memmap(bin_path, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024
        
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {bin_path}'):
            # Get batch of tokenized data
            batch = tokenized.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['tokens'])
            # Write to memmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        
        print(f"Saved {split_name} to {bin_path}")
        return int(arr_len)
    
    # Process both splits
    train_tokens = process_split("train", dataset["train"])
    val_tokens = process_split("validation", dataset["validation"])
    
    # Save metadata
    meta = {
        'vocab_size': len(tokenizer),
        'train_tokens': train_tokens,
        'val_tokens': val_tokens,
        'eot_token': eos_token
    }
    
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"✅ Dataset preparation complete!")
    print(f"Train: {train_tokens:,} tokens")
    print(f"Val: {val_tokens:,} tokens")
    print(f"Vocab size: {len(tokenizer)}")

class ChunkDataset(torch.utils.data.Dataset):
    """
    Non-overlapping (or strided) chunks of a mem-mapped token array.
    Every __getitem__(i) returns the i-th chunk, so the dataset has a
    stable length and can be shuffled by a Sampler.
    """
    def __init__(self, bin_path: str, block_size: int, stride: int | None = None):
        self.data       = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.stride     = stride or block_size        # stride=block_size ⇒ no overlap
        self.n_chunks   = (len(self.data) - 1 - block_size) // self.stride + 1

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx: int):
        start = idx * self.stride
        x = torch.from_numpy(self.data[start : start + self.block_size]).long()
        y = torch.from_numpy(self.data[start + 1 : start + self.block_size + 1]).long()
        return x, y

def get_dataloader(
    data_dir: str,
    split: str = "train", 
    batch_size: int = 32,
    block_size: int = 1024,
    num_workers: int = 0,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader for the prepared TinyStories dataset
    
    Args:
        data_dir: Directory containing the .bin files
        split: "train" or "validation" 
        batch_size: Batch size
        block_size: Sequence length
        num_workers: Number of DataLoader workers
        shuffle: Whether to shuffle 
    
    Returns:
        DataLoader yielding (x, y) tensors of shape (batch_size, block_size)
    """
    # Check for .bin file
    bin_path = os.path.join(data_dir, f'{split}.bin')
    if not os.path.exists(bin_path):
        raise FileNotFoundError(
            f"Binary file not found: {bin_path}\n"
            f"Run 'python {__file__}' first to prepare the data."
        )
    
    # Load dataset and create dataloader
    dataset = ChunkDataset(bin_path, block_size)
    print(f"Loaded {split} data: {len(dataset.data):,} tokens")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    return dataloader




if __name__ == "__main__":
    data_dir = os.path.dirname(__file__) or '.'
    
    # Check if data already exists
    train_bin = os.path.join(data_dir, 'train.bin')
    val_bin = os.path.join(data_dir, 'validation.bin')
    
    if os.path.exists(train_bin) and os.path.exists(val_bin):
        print("✅ Binary files already exist, skipping data preparation...")
    else:
        print("Preparing data...")
        prepare_data()
    
    # Test the dataloader
    print("\n" + "="*50)
    print("Testing DataLoader...")
    
    train_loader = get_dataloader(
        data_dir=data_dir,
        split="train", 
        batch_size=2,  # 2 sequences
        block_size=512,  # 512 tokens each
        num_workers=0,  # Keep at 0 for testing to avoid overhead
    )
    
    # Test multiple batches
    tokenizer = AutoTokenizer.from_pretrained("niklassheth/tinystories-tokenizer")
    
    for batch_num, (x, y) in enumerate(train_loader):
        print(f"\nBatch {batch_num + 1}:")
        print(f"Batch shape: x={x.shape}, y={y.shape}")
        
        # Decode both sequences in the batch
        for seq_num in range(x.shape[0]):
            sample_tokens = x[seq_num].tolist() 
            sample_text = tokenizer.decode(sample_tokens)
            print(f"Sequence {seq_num + 1}: '{sample_text}'")
        
        # Verify that y = x shifted by 1
        assert torch.equal(y[0][:-1], x[0][1:]), "y should be x shifted by 1"
        
        if batch_num >= 1:  # Test 2 batches total
            break
    
    print("✅ DataLoader test complete!")