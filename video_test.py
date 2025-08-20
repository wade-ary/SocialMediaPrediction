def preview_top_errors(posts_ds, k=10):
    """
    Runs the main build function on top-k rows, then prints:
      - status counts
      - per-row quick diagnostics for non-'ok'
      - verifies path existence inline
    """
    from collections import Counter

    sample_ds = posts_ds.select(range(min(k, len(posts_ds))))
    out = build_video_text_embeddings(sample_ds, batch_size=2, num_frames=8, target_size=224)

    statuses = Counter(out["proc_status"])
    print("\nStatus counts (top {k}):", dict(statuses))

    for i in range(len(out)):
        s = out[i]["proc_status"]
        if s != "ok":
            print(f"\nRow {i} pid={out[i]['pid']}")
            print(f"  status      : {s}")
            print(f"  path exists : {os.path.exists(out[i]['abs_video_path'])}")
            print(f"  abs path    : {out[i]['abs_video_path']}")
            # Shorten detail to keep log readable
            detail = out[i]["error_detail"] or ""
            print("  error_detail:", detail[:400] + ("..." if len(detail) > 400 else ""))

    # Sanity: show one success example if available
    for i in range(len(out)):
        if out[i]["proc_status"] == "ok":
            te = out[i]["text_emb_f16"]; ve = out[i]["video_emb_f16"]
            print(f"\nFirst OK row: idx={i} pid={out[i]['pid']}")
            print("  text_emb len :", None if te is None else len(te))
            print("  video_emb len:", None if ve is None else len(ve))
            break

def inspect_cached_embeddings(cache_dir="./video_text_cache", sample_size=10, num_values=5):
    """
    Load cached embeddings and display the first few values from the first few samples.
    
    Args:
        cache_dir: Path to the cached embeddings
        sample_size: Number of samples to inspect (default 10)
        num_values: Number of embedding values to show per sample (default 5)
    """
    try:
        from datasets import load_from_disk
        from video import compute_or_load_embeddings
        
        print(f"\n{'='*60}")
        print("INSPECTING CACHED EMBEDDINGS")
        print(f"{'='*60}")
        
        # Try to load from cache first
        if os.path.exists(cache_dir):
            print(f"Loading embeddings from cache: {cache_dir}")
            embeddings_ds = load_from_disk(cache_dir)
        else:
            print(f"Cache not found at {cache_dir}, computing fresh embeddings...")
            # Get a small subset of posts for testing
            sample_posts = ds_train_posts.select(range(min(sample_size, len(ds_train_posts))))
            embeddings_ds = compute_or_load_embeddings(sample_posts, cache_dir=cache_dir)
        
        print(f"✅ Loaded embeddings dataset with {len(embeddings_ds)} samples")
        print(f"Columns: {embeddings_ds.column_names}")
        
        # Inspect first few samples
        print(f"\nInspecting first {sample_size} samples:")
        print(f"{'='*80}")
        
        for i in range(min(sample_size, len(embeddings_ds))):
            pid = embeddings_ds["pid"][i]
            uid = embeddings_ds["uid"][i]
            
            # Get embeddings
            text_emb = embeddings_ds["text_emb_f16"][i]
            video_emb = embeddings_ds["video_emb_f16"][i]
            
            # Convert to lists if they're numpy arrays
            if hasattr(text_emb, 'tolist'):
                text_emb = text_emb.tolist()
            if hasattr(video_emb, 'tolist'):
                video_emb = video_emb.tolist()
            
            # Get first few values
            text_preview = text_emb[:num_values] if text_emb else [None] * num_values
            video_preview = video_emb[:num_values] if video_emb else [None] * num_values
            
            # Calculate norms if embeddings exist
            text_norm = None
            video_norm = None
            if text_emb:
                text_norm = sum(x*x for x in text_emb if x is not None) ** 0.5
            if video_emb:
                video_norm = sum(x*x for x in video_emb if x is not None) ** 0.5
            
            print(f"Sample {i+1}:")
            print(f"  PID: {pid}, UID: {uid}")
            print(f"  Text emb (first {num_values}): {text_preview}")
            print(f"  Video emb (first {num_values}): {video_preview}")
            print(f"  Text norm: {text_norm:.6f}" if text_norm else "  Text norm: None")
            print(f"  Video norm: {video_norm:.6f}" if video_norm else "  Video norm: None")
            print(f"  Text length: {len(text_emb) if text_emb else 0}")
            print(f"  Video length: {len(video_emb) if video_emb else 0}")
            print("-" * 80)
        
        return embeddings_ds
        
    except Exception as e:
        print(f"❌ Error inspecting cached embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None

# Run the inspection function
print(f"\n{'='*60}")
print("INSPECTING CACHED EMBEDDINGS")
print(f"{'='*60}")

cached_embeddings = inspect_cached_embeddings(cache_dir="./video_text_cache_train", sample_size=10, num_values=5)

if cached_embeddings:
    print(f"\n✅ Successfully loaded {len(cached_embeddings)} cached embeddings")
    print(f"Use 'cached_embeddings' variable to access the full dataset")
else:
    print(f"\n❌ Failed to load cached embeddings")


def inspect_cached_embedding_lengths(cache_dir="./video_text_cache_train"):
    """
    Load cached embeddings and report length consistency and None counts
    for both text and video embeddings.
    """
    try:
        if not os.path.exists(cache_dir):
            print(f"Cache directory not found: {cache_dir}")
            return
        ds = load_from_disk(cache_dir)
        print(f"Loaded cached embeddings from: {cache_dir} (rows={len(ds)})")

        # Text embeddings
        text_embeddings = ds["text_emb_f16"] if "text_emb_f16" in ds.column_names else []
        lens_text = [len(e) if isinstance(e, (list, np.ndarray)) else None for e in text_embeddings]
        print("Text emb unique lengths:", set(lens_text))
        print("Text emb None count:", sum(e is None for e in text_embeddings))

        # Video embeddings
        video_embeddings = ds["video_emb_f16"] if "video_emb_f16" in ds.column_names else []
        lens_video = [len(e) if isinstance(e, (list, np.ndarray)) else None for e in video_embeddings]
        print("Video emb unique lengths:", set(lens_video))
        print("Video emb None count:", sum(e is None for e in video_embeddings))

        # Optional: show mismatched rows indices (first few)
     
            
    except Exception as e:
        print(f"Failed to inspect cached embeddings: {e}")

inspect_cached_embedding_lengths()