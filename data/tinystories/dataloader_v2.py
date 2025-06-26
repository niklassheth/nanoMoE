"""
Simple and Fast TinyStories DataLoader v2
Clean, minimal approach following nanogpt style
"""

import os
import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
import tiktoken
from multiprocessing import cpu_count
import numpy as np
from tqdm import tqdm
import pickle


def clean_text(text: str) -> str:
    """
    Clean UTF-8 encoding corruption artifacts, focusing on the most common issues.
    Replaces corrupted characters with ASCII equivalents where possible.
    """
    # Most common corruptions from analysis (1.2M+ total occurrences)
    corruption_fixes = {
        'â€œ': '"',    # Left double quotation mark → ASCII quote
        'â€': '"',     # Right double quotation mark → ASCII quote  
        'â€™': "'",    # Right single quotation mark/apostrophe → ASCII apostrophe
        'â€˜': "'",    # Left single quotation mark → ASCII apostrophe
        'â€¦': '...',  # Horizontal ellipsis → ASCII dots
        
        # Additional fixable corruptions from remaining patterns
        'Ã‰': 'É',     # É with acute
        'Ã ': 'À',     # À with grave
        'Ã¨': 'È',     # È with grave
        'Ã©': 'é',     # é with acute
        'Ã¡': 'á',     # á with acute
        'Ã­': 'í',     # í with acute
        'Ã³': 'ó',     # ó with acute
        'Ãº': 'ú',     # ú with acute
        'Ã±': 'ñ',     # ñ with tilde
        'Ã¼': 'ü',     # ü with diaeresis
        'Ã¤': 'ä',     # ä with diaeresis
        'Ã¶': 'ö',     # ö with diaeresis
        'ÃŸ': 'ß',     # German eszett
        'Ã§': 'ç',     # ç with cedilla
        'â"€': '—',    # Em dash
        'â"': '–',     # En dash
    }
    
    cleaned_text = text
    for corrupted, clean in corruption_fixes.items():
        cleaned_text = cleaned_text.replace(corrupted, clean)
    
    return cleaned_text


def has_unfixable_corruption(text: str) -> bool:
    """
    Check if text contains unfixable corruption patterns after cleaning.
    Only filters out complex multi-byte corruptions that can't be reasonably fixed.
    """
    import re
    
    # Only filter unfixable corruption patterns
    unfixable_patterns = [
        r'Ã¢â‚¬',      # Complex multi-byte quote corruptions
        r'â∑',         # Mathematical symbols  
        r'â[^\w\s"€™˜¦"—]',  # â followed by non-standard characters (but allow common punctuation)
        r'Ã[^\s][^\w\s]',    # Ã followed by non-letter sequences
    ]
    
    for pattern in unfixable_patterns:
        if re.search(pattern, text):
            return True
            
    return False


class TokenDataset(IterableDataset):
    """Simple iterable dataset that yields random chunks from a binary token file"""
    
    def __init__(self, bin_path: str, block_size: int):
        self.block_size = block_size
        # Use memmap for efficient random access
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        
    def __iter__(self):
        while True:
            # Generate random starting position
            start = torch.randint(0, len(self.data) - self.block_size, (1,)).item()
            x = torch.from_numpy((self.data[start:start + self.block_size]).astype(np.int64))
            y = torch.from_numpy((self.data[start + 1:start + self.block_size + 1]).astype(np.int64))
            yield x, y

class BlockDataset(torch.utils.data.Dataset):
    """All (block_size+1)-token windows of the mem-mapped file."""
    def __init__(self, bin_path: str, block_size: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.n_samples = len(self.data) - block_size      # every legal start pos

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        a = self.data[idx:idx + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(a[:-1])
        y = torch.from_numpy(a[1:])
        return x, y

class LazyRandPermSampler(torch.utils.data.Sampler):
    def __init__(self, length, chunk=10_000_000, generator=None):
        self.length, self.chunk, self.gen = length, chunk, generator

    def __iter__(self):
        g = torch.default_generator if self.gen is None else self.gen
        start = 0
        while start < self.length:
            n = min(self.chunk, self.length - start)
            yield from (torch.randperm(n, generator=g) + start).tolist()
            start += n

    def __len__(self):
        return self.length

def prepare_data(data_dir=None):
    """
    Download, tokenize, and save TinyStories dataset
    Run with: python dataloader_v2.py
    
    Args:
        data_dir: Directory to save dataset files
    """
    filter_corrupted_accents = True
    clean_corruption = True

    if data_dir is None:
        data_dir = os.path.dirname(__file__)
    
    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories")
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    eos_token = tokenizer.eot_token  # End of text token
    
    def process_split(split_name: str, split_data):
        print(f"Processing {split_name} split...")
        
        # Filter out very short stories
        print("Filtering short stories...")
        cores = min(cpu_count(), 16)
        filtered = split_data.filter(
            lambda x: len(x["text"]) > 32, 
            num_proc=cores
        )
        
        # Filter out stories with unfixable corruption (after attempting to clean)
        if filter_corrupted_accents:
            print("Filtering stories with unfixable corruption...")
            def is_cleanable(example):
                # Try cleaning first
                cleaned = clean_text(example["text"]) if clean_corruption else example["text"]
                # Only filter if still has unfixable corruption after cleaning
                return not has_unfixable_corruption(cleaned)
            
            filtered = filtered.filter(is_cleanable, num_proc=cores)
        
        # Tokenize each story and track lengths (like openwebtext)
        def tokenize_story(example):
            text = example['text']
            if clean_corruption:
                text = clean_text(text)
            tokens = tokenizer.encode_ordinary(text)
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
        'vocab_size': tokenizer.n_vocab,
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
    print(f"Vocab size: {tokenizer.n_vocab}")


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
        shuffle: Whether to shuffle (not critical since we sample randomly)
    
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
    dataset = BlockDataset(bin_path, block_size)
    print(f"Loaded {split} data: {len(dataset.data):,} tokens")
    sampler = LazyRandPermSampler(len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
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
    tokenizer = tiktoken.get_encoding("gpt2")
    
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