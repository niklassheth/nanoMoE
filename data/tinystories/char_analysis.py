#!/usr/bin/env python3
"""
Character-level analysis of the TinyStories dataset.
Analyzes character frequencies in the original raw text data.
"""

import os
import json
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count, Manager
from datasets import load_dataset
from tqdm import tqdm
import unicodedata
import sys
import time
import psutil


# Global Unicode category cache
_unicode_cache = {}

def get_unicode_category(char):
    """Get Unicode category with caching."""
    if char not in _unicode_cache:
        _unicode_cache[char] = unicodedata.category(char)
    return _unicode_cache[char]


def analyze_text_chunk(text_list):
    """Analyze a list of texts and return character counts."""
    char_counts = Counter()
    char_categories = defaultdict(Counter)
    
    # Concatenate all texts for more efficient processing
    combined_text = ''.join(text_list)
    
    # Count all characters at once
    char_counts.update(combined_text)
    
    # Categorize unique characters only (not every occurrence)
    for char in char_counts.keys():
        category = get_unicode_category(char)
        char_categories[category][char] = char_counts[char]
    
    return char_counts, char_categories


def process_batch(batch_texts):
    """Process a batch of texts and return aggregated character statistics."""
    # Process the entire batch at once for efficiency
    char_counts, char_categories = analyze_text_chunk(batch_texts)
    return char_counts, char_categories


def analyze_dataset(split_name="train", batch_size=20000, max_workers=None, sample_size=None):
    """
    Analyze character frequencies in the TinyStories dataset.
    
    Args:
        split_name: 'train' or 'validation'
        batch_size: Number of texts to process in each batch
        max_workers: Number of parallel processes (default: cpu_count())
        sample_size: If provided, only analyze first N stories for testing
    """
    print(f"Loading TinyStories {split_name} split...")
    
    # Use streaming mode for memory efficiency
    if sample_size:
        # For samples, use regular loading
        dataset = load_dataset("roneneldan/TinyStories", split=split_name)
        dataset = dataset.select(range(min(sample_size, len(dataset))))
        dataset_size = len(dataset)
        print(f"Using sample of {dataset_size:,} stories for analysis")
    else:
        # For full dataset, use streaming mode
        dataset = load_dataset("roneneldan/TinyStories", split=split_name, streaming=True)
        # Get approximate size for progress tracking
        info = load_dataset("roneneldan/TinyStories", split=split_name, streaming=False)
        dataset_size = len(info)
        print(f"Dataset size: {dataset_size:,} stories (streaming mode)")
    
    # Set up multiprocessing - use more workers for better parallelism
    if max_workers is None:
        max_workers = min(cpu_count(), 6)  # Slightly more aggressive
    
    print(f"Using {max_workers} processes for analysis")
    print(f"Batch size: {batch_size:,} stories per batch")
    
    # Initialize counters
    total_char_counts = Counter()
    total_categories = defaultdict(Counter)
    total_chars_processed = 0
    
    # Process in batches to manage memory
    batch_texts = []
    processed_stories = 0
    
    # Create progress bar
    progress_bar = tqdm(total=dataset_size, desc="Processing stories", unit="stories")
    
    # Memory usage monitoring
    process = psutil.Process()
    start_time = time.time()
    
    with Pool(max_workers) as pool:
        for story in dataset:
            batch_texts.append(story['text'])
            processed_stories += 1
            
            # Process batch when it reaches target size
            if len(batch_texts) >= batch_size:
                # Split batch among workers for parallel processing
                chunk_size = max(1, len(batch_texts) // max_workers)
                text_chunks = [batch_texts[j:j+chunk_size] for j in range(0, len(batch_texts), chunk_size)]
                
                # Process chunks in parallel
                results = pool.map(process_batch, text_chunks)
                
                # Aggregate results
                for char_counts, categories in results:
                    total_char_counts.update(char_counts)
                    for category, chars in categories.items():
                        total_categories[category].update(chars)
                    total_chars_processed += sum(char_counts.values())
                
                # Update progress and clear batch
                progress_bar.update(len(batch_texts))
                batch_texts = []
                
                # Print memory usage every 10 batches
                if processed_stories % (batch_size * 10) == 0:
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    elapsed = time.time() - start_time
                    rate = processed_stories / elapsed
                    print(f"Memory: {memory_mb:.1f}MB, Rate: {rate:.1f} stories/sec")
    
        # Process remaining texts
        if batch_texts:
            chunk_size = max(1, len(batch_texts) // max_workers)
            text_chunks = [batch_texts[j:j+chunk_size] for j in range(0, len(batch_texts), chunk_size)]
            results = pool.map(process_batch, text_chunks)
            
            for char_counts, categories in results:
                total_char_counts.update(char_counts)
                for category, chars in categories.items():
                    total_categories[category].update(chars)
                total_chars_processed += sum(char_counts.values())
            
            progress_bar.update(len(batch_texts))
    
    progress_bar.close()
    
    # Final memory and timing stats
    elapsed = time.time() - start_time
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Analysis complete! Processed {processed_stories:,} stories in {elapsed:.1f}s")
    print(f"Peak memory usage: {memory_mb:.1f}MB")
    print(f"Processing rate: {processed_stories / elapsed:.1f} stories/sec")
    
    return total_char_counts, total_categories, total_chars_processed


def find_outliers(char_counts, threshold_percentile=99.9):
    """Find outlier characters based on frequency."""
    total_chars = sum(char_counts.values())
    frequencies = [(char, count, count/total_chars*100) for char, count in char_counts.items()]
    frequencies.sort(key=lambda x: x[1], reverse=True)
    
    # Characters below threshold percentile
    threshold_idx = int(len(frequencies) * threshold_percentile / 100)
    rare_chars = frequencies[threshold_idx:]
    
    return frequencies, rare_chars


def categorize_chars(char_counts):
    """Categorize characters by type."""
    categories = {
        'ascii_printable': [],
        'ascii_control': [],
        'unicode_letters': [],
        'unicode_punctuation': [],
        'unicode_symbols': [],
        'unicode_numbers': [],
        'unicode_other': [],
        'unknown': []
    }
    
    for char, count in char_counts.items():
        try:
            category = unicodedata.category(char)
            if ord(char) < 128:
                if char.isprintable():
                    categories['ascii_printable'].append((char, count, category))
                else:
                    categories['ascii_control'].append((char, count, category))
            else:
                if category.startswith('L'):  # Letter
                    categories['unicode_letters'].append((char, count, category))
                elif category.startswith('P'):  # Punctuation
                    categories['unicode_punctuation'].append((char, count, category))
                elif category.startswith('S'):  # Symbol
                    categories['unicode_symbols'].append((char, count, category))
                elif category.startswith('N'):  # Number
                    categories['unicode_numbers'].append((char, count, category))
                else:
                    categories['unicode_other'].append((char, count, category))
        except:
            categories['unknown'].append((char, count, 'UNKNOWN'))
    
    return categories


def main():
    print("=== TinyStories Character-Level Analysis ===\n")
    
    # Check if we should run a sample first
    import sys
    sample_mode = '--sample' in sys.argv
    sample_size = 50000 if sample_mode else None
    
    if sample_mode:
        print("Running in SAMPLE mode - analyzing first 50K stories only")
    
    # Analyze both splits
    splits = ['validation', 'train']  # Start with smaller validation split
    
    for split in splits:
        print(f"\n{'='*50}")
        print(f"Analyzing {split} split")
        print(f"{'='*50}")
        
        try:
            char_counts, unicode_categories, total_chars = analyze_dataset(split, sample_size=sample_size)
            
            print(f"\nBasic Statistics:")
            print(f"Total characters: {total_chars:,}")
            print(f"Unique characters: {len(char_counts):,}")
            
            # Find most and least common characters
            frequencies, rare_chars = find_outliers(char_counts)
            
            print(f"\nTop 20 most common characters:")
            for i, (char, count, percent) in enumerate(frequencies[:20]):
                char_repr = repr(char) if not char.isprintable() else char
                print(f"{i+1:2d}. {char_repr:>10} : {count:>10,} ({percent:>6.2f}%)")
            
            print(f"\nBottom 20 least common characters:")
            for i, (char, count, percent) in enumerate(frequencies[-20:]):
                char_repr = repr(char) if not char.isprintable() else char
                print(f"{i+1:2d}. {char_repr:>10} : {count:>10,} ({percent:>6.2f}%)")
            
            # Categorize characters
            char_categories = categorize_chars(char_counts)
            
            print(f"\nCharacter Categories:")
            for category, chars in char_categories.items():
                if chars:
                    total_count = sum(count for _, count, _ in chars)
                    percentage = total_count / total_chars * 100
                    print(f"{category:>20}: {len(chars):>6} unique chars, {total_count:>12,} total ({percentage:>6.2f}%)")
            
            # Show outlier details
            print(f"\nUnicode Categories Summary:")
            for category, char_counter in unicode_categories.items():
                total_count = sum(char_counter.values())
                percentage = total_count / total_chars * 100
                print(f"{category:>10}: {len(char_counter):>6} unique, {total_count:>12,} total ({percentage:>6.2f}%)")
            
            # Save detailed results
            output_file = f"{split}_char_analysis.json"
            results = {
                'total_chars': total_chars,
                'unique_chars': len(char_counts),
                'char_frequencies': [(char, count) for char, count in char_counts.most_common()],
                'unicode_categories': {cat: dict(counter) for cat, counter in unicode_categories.items()},
                'categorized_chars': {cat: [(char, count, unicode_cat) for char, count, unicode_cat in chars] 
                                    for cat, chars in char_categories.items()}
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\nDetailed results saved to: {output_file}")
            
        except Exception as e:
            print(f"Error analyzing {split} split: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()