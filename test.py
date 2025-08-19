from datasets import load_dataset
from collections import Counter
import re
import math
from data_process import *

def get_top_k_categories(values, k=5, min_count=1):
    """
    Get the top k most frequent categories from a list of values.
    
    Args:
        values: List of category values
        k: Number of top categories to return
        min_count: Minimum count threshold for categories
    
    Returns:
        list: Top k most frequent categories, sorted by frequency
    """
    # Count occurrences of each category
    counts = Counter([v if v is not None else "unknown" for v in values])
    
    # Filter by minimum count
    filtered_counts = {cat: count for cat, count in counts.items() if count >= min_count}
    
    # Get top k by frequency
    top_categories = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:k]
    
    # Extract just the category names
    top_cats = [cat for cat, count in top_categories]
    
    print(f"Top {k} categories (min_count={min_count}):")
    for i, (cat, count) in enumerate(top_categories):
        print(f"  {i+1}. {cat}: {count} occurrences")
    
    return top_cats

def build_global_vocabularies(posts_ds):
    """Build global language and location vocabularies from full dataset"""
    print(f"\n{'='*60}")
    print("BUILDING GLOBAL VOCABULARIES")
    print(f"{'='*60}")
    
    langs_all = posts_ds["post_text_language"]
    locs_all = posts_ds["post_location"]
    
    print(f"Total posts: {len(posts_ds)}")
    print(f"Language samples: {len(langs_all)}")
    print(f"Location samples: {len(locs_all)}")
    
    # Get top 5 most frequent categories
    print(f"\nLanguage categories:")
    lang_vocab = get_top_k_categories(langs_all, k=5, min_count=1)
    
    print(f"\nLocation categories:")
    loc_vocab = get_top_k_categories(locs_all, k=5, min_count=1)
    
    # Create the "other" categories for languages/locations not in top 5
    print(f"\nVocabulary summary:")
    print(f"Language vocabulary: {lang_vocab}")
    print(f"Location vocabulary: {loc_vocab}")
    
    return lang_vocab, loc_vocab

# Load Train dataset
ds_train_posts = load_dataset("smpchallenge/SMP-Video", 'posts')['train']

# Build global vocabularies
lang_vocab, loc_vocab = build_global_vocabularies(ds_train_posts)

print(f"\n✅ Global vocabularies created successfully!")
print(f"Use these vocabularies consistently across all splits to maintain feature dimensions.")