import os
import numpy as np
from datasets import load_dataset, load_from_disk
from video import build_video_text_embeddings

def cache_complete_embeddings(cache_dir="./video_text_complete_cache", batch_size=8):
    """
    Cache embeddings for the complete training dataset (all posts).
    This ensures the cache is always valid regardless of how you split the data.
    
    Args:
        cache_dir: Directory to save the complete embeddings cache
        batch_size: Batch size for processing
    
    Returns:
        HuggingFace Dataset: Complete embeddings dataset
    """
    print(f"\n{'='*80}")
    print("CACHING COMPLETE EMBEDDINGS DATASET")
    print(f"{'='*80}")
    
    # Load the complete training posts dataset
    ds_train_posts = load_dataset("smpchallenge/SMP-Video", 'posts')['train']
    print(f"Total posts to process: {len(ds_train_posts)}")
    
    # Check if cache already exists
    if os.path.exists(cache_dir):
        print(f"✅ Loading existing complete cache from: {cache_dir}")
        embeddings_full = load_from_disk(cache_dir)
        print(f"Loaded {len(embeddings_full)} cached embeddings")
        return embeddings_full
    
    # Cache doesn't exist, compute embeddings for all posts
    print(f"🔄 Computing embeddings for all posts...")
    print(f"This may take a while for {len(ds_train_posts)} posts...")
    
    # Compute embeddings for the complete dataset
    embeddings_full = build_video_text_embeddings(ds_train_posts, batch_size=batch_size)
    
    # Save to disk
    print(f"💾 Saving complete embeddings to: {cache_dir}")
    embeddings_full.save_to_disk(cache_dir)
    
    print(f"✅ Complete embeddings cached successfully!")
    print(f"Total embeddings: {len(embeddings_full)}")
    print(f"Cache location: {cache_dir}")
    
    return embeddings_full


def get_embeddings_for_pids(embeddings_full, target_pids, verbose=True):
    """
    Retrieve embeddings for specific PIDs from the complete embeddings dataset.
    
    Args:
        embeddings_full: Complete embeddings dataset
        target_pids: List/set of PIDs to retrieve
        verbose: Whether to print progress information
    
    Returns:
        HuggingFace Dataset: Filtered embeddings dataset with only target PIDs
    """
    if verbose:
        print(f"🔍 Retrieving embeddings for {len(target_pids)} PIDs from complete cache...")
    
    # Create a set for faster lookup
    target_pids_set = set(target_pids)
    
    # Find indices where pid is in target_pids
    matching_indices = []
    for i, pid in enumerate(embeddings_full["pid"]):
        if pid in target_pids_set:
            matching_indices.append(i)
    
    if verbose:
        print(f"Found {len(matching_indices)} matching embeddings")
    
    # Filter the dataset to only include matching PIDs
    filtered_embeddings = embeddings_full.select(matching_indices)
    
    if verbose:
        print(f"✅ Retrieved {len(filtered_embeddings)} embeddings")
    
    return filtered_embeddings


def get_embeddings_for_split(data_split, embeddings_full=None, cache_dir="./video_text_complete_cache", verbose=True):
    """
    Convenience function to get embeddings for a specific data split.
    
    Args:
        data_split: Dictionary containing 'posts' with PIDs
        embeddings_full: Complete embeddings dataset (if None, loads from cache)
        cache_dir: Cache directory for complete embeddings
        verbose: Whether to print progress information
    
    Returns:
        HuggingFace Dataset: Filtered embeddings for the split
    """
    # Load complete embeddings if not provided
    if embeddings_full is None:
        if not os.path.exists(cache_dir):
            print(f"❌ Complete cache not found at {cache_dir}")
            print("Please run cache_complete_embeddings() first")
            return None
        embeddings_full = load_from_disk(cache_dir)
    
    # Extract PIDs from the split
    target_pids = set(data_split["pid"])
    
    # Get embeddings for these PIDs
    return get_embeddings_for_pids(embeddings_full, target_pids, verbose=verbose)


def inspect_complete_cache(cache_dir="./video_text_complete_cache"):
    """
    Inspect the complete embeddings cache to verify its contents.
    
    Args:
        cache_dir: Path to the complete cache directory
    """
    if not os.path.exists(cache_dir):
        print(f"❌ Complete cache not found at: {cache_dir}")
        return
    
    print(f"\n{'='*80}")
    print("INSPECTING COMPLETE EMBEDDINGS CACHE")
    print(f"{'='*80}")
    
    embeddings_full = load_from_disk(cache_dir)
    
    print(f"Cache location: {cache_dir}")
    print(f"Total embeddings: {len(embeddings_full)}")
    print(f"Columns: {embeddings_full.column_names}")
    
    # Check for None/empty embeddings
    text_none_count = sum(1 for emb in embeddings_full["text_emb_f16"] if emb is None)
    video_none_count = sum(1 for emb in embeddings_full["video_emb_f16"] if emb is None)
    
    print(f"Text embeddings None: {text_none_count}/{len(embeddings_full)}")
    print(f"Video embeddings None: {video_none_count}/{len(embeddings_full)}")
    
    # Sample a few embeddings
    print(f"\nSample embeddings:")
    for i in range(min(3, len(embeddings_full))):
        pid = embeddings_full["pid"][i]
        text_emb = embeddings_full["text_emb_f16"][i]
        video_emb = embeddings_full["video_emb_f16"][i]
        
        text_len = len(text_emb) if text_emb else 0
        video_len = len(video_emb) if video_emb else 0
        
        print(f"  {pid}: text({text_len}), video({video_len})")
    # Show first 5 numbers of first 5 embeddings
    print(f"\nFirst 5 numbers of first 5 text and video embeddings:")
    for i in range(min(5, len(embeddings_full))):
        pid = embeddings_full["pid"][i]
        text_emb = embeddings_full["text_emb_f16"][i]
        video_emb = embeddings_full["video_emb_f16"][i]
        
        if text_emb and video_emb:
            text_preview = text_emb[:5] if len(text_emb) >= 5 else text_emb
            video_preview = video_emb[:5] if len(video_emb) >= 5 else video_emb
            
            print(f"  {pid}:")
            print(f"    text[:5] = {text_preview}")
            print(f"    video[:5] = {video_preview}")
        else:
            print(f"  {pid}: text=None, video=None")


if __name__ == "__main__":
    # Cache the complete dataset (run this once)
    print("🚀 Starting complete embeddings caching...")
    complete_embeddings = cache_complete_embeddings()
    
    # Inspect the cache
    inspect_complete_cache()
    
    print(f"\n✅ Complete embeddings setup complete!")
    print(f"Use get_embeddings_for_split() to retrieve embeddings for any data split.") 