"""
Train a custom BPE tokenizer on TinyStories dataset and upload to Hugging Face Hub
Vocab size: 8192 (8182 regular tokens + 10 special tokens)
"""

import os
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from huggingface_hub import login
import argparse




def get_training_corpus():
    """Generator that yields text from TinyStories dataset"""
    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories")
    
    print("Processing training data...")
    for example in dataset["train"]:
        if len(example["text"]) > 32:  # Filter short stories like in dataloader.py
            yield example["text"]


def train_tokenizer(vocab_size=8192):
    """Train a BPE tokenizer on TinyStories dataset"""
    
    # Special tokens (10 total)
    special_tokens = [
        "<|eos|>",
        "<|reserved_1|>",
        "<|reserved_2|>", 
        "<|reserved_3|>",
        "<|reserved_4|>",
        "<|reserved_5|>",
        "<|reserved_6|>", 
        "<|reserved_7|>",
        "<|reserved_8|>",
        "<|unk|>"
    ]
    
    print(f"Training BPE tokenizer with vocab_size={vocab_size}")
    print(f"Special tokens: {special_tokens}")
    
    # Initialize tokenizer with BPE model
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    
    # Use ByteLevel pre-tokenizer and decoder (like GPT-2) to preserve whitespace
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    
    # Create trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,  # Minimum frequency for merges
        show_progress=True
    )
    
    # Train from iterator
    print("Starting training...")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    
    # Don't add post-processor - we'll handle BOS/EOS manually in dataloader
    
    # Save tokenizer
    tokenizer_path = "custom_tinystories_tokenizer.json"
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    
    # Print some stats
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Special tokens mapping:")
    for token in special_tokens:
        print(f"  {token}: {tokenizer.token_to_id(token)}")
    
    return tokenizer


def upload_to_hub(tokenizer, repo_name):
    """Upload tokenizer to Hugging Face Hub"""
    
    print(f"Preparing to upload to Hub as '{repo_name}'")
    
    # Wrap with PreTrainedTokenizerFast
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        padding_side="right",
        truncation_side="right",
        unk_token="<|unk|>",
        eos_token="<|eos|>",
    )
    
    # Save locally first
    local_dir = "./tokenizer_for_upload"
    wrapped_tokenizer.save_pretrained(local_dir)
    print(f"Tokenizer saved locally to {local_dir}")
    
    try:
        # Login to Hugging Face Hub
        print("Logging in to Hugging Face Hub...")
        login()  # This will prompt for token if not already authenticated
        
        # Upload to Hub
        print(f"Uploading to Hub...")
        wrapped_tokenizer.push_to_hub(
            repo_id=repo_name,
            commit_message=f"Upload custom TinyStories BPE tokenizer (vocab_size={tokenizer.get_vocab_size()})",
            private=False  # Set to True if you want a private repo
        )
        
        print(f"✅ Successfully uploaded tokenizer to: https://huggingface.co/{repo_name}")
        print(f"You can now load it with: AutoTokenizer.from_pretrained('{repo_name}')")
        
    except Exception as e:
        print(f"❌ Failed to upload to Hub: {e}")
        print(f"Tokenizer files are saved locally in {local_dir}")
        print("You can manually upload them or try again later.")


def test_tokenizer(tokenizer, text):
    """Test the trained tokenizer with a single text string"""
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded.ids)
    
    print(f"\nOriginal: '{text}'")
    print(f"Tokens: {encoded.tokens}")
    print(f"IDs: {encoded.ids}")
    print(f"Decoded: '{decoded}'")


def run_tests(tokenizer):
    """Run tokenizer tests with default test cases"""
    print("\n" + "="*50)
    print("Testing tokenizer...")
    
    test_texts = [
        "Once upon a time, there was a little girl named Emma.",
        "The cat sat on the mat and looked at the moon.",
        "Hello world!<|eos|>",
    ]
    for text in test_texts:
        test_tokenizer(tokenizer, text)


def main():
    parser = argparse.ArgumentParser(description="Train custom BPE tokenizer for TinyStories")
    parser.add_argument("--vocab-size", type=int, default=8192, help="Vocabulary size (default: 8192)")
    parser.add_argument("--repo-name", type=str, default="tinystories-tokenizer", 
                       help="Hugging Face Hub repository name")
    parser.add_argument("--no-upload", action="store_true", help="Skip uploading to Hub")
    parser.add_argument("--test-only", action="store_true", help="Only test existing tokenizer")
    parser.add_argument("--tokenize", type=str, help="Tokenize a custom string and exit")
    
    args = parser.parse_args()
    
    if args.tokenize or args.test_only:
        # Load existing tokenizer and test
        try:
            tokenizer = Tokenizer.from_file("custom_tinystories_tokenizer.json")
            if args.tokenize:
                print("\nTokenizing custom text...")
                test_tokenizer(tokenizer, args.tokenize)
            else:
                run_tests(tokenizer)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
        return
    
    # Train tokenizer
    tokenizer = train_tokenizer(vocab_size=args.vocab_size)
    
    # Test tokenizer with default cases
    run_tests(tokenizer)
    
    # Upload to Hub (unless disabled)
    if not args.no_upload:
        upload_to_hub(tokenizer, repo_name=args.repo_name)
    else:
        print("Skipping upload to Hub (--no-upload flag set)")


if __name__ == "__main__":
    main()