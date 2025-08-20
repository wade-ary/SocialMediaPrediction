from data_process import build_post_features, build_target_features, build_user_features, build_video_features
from data_process import *
from datasets import load_dataset
from collections import Counter
import re
import math
import numpy as np

ds_train_posts = load_dataset("smpchallenge/SMP-Video", 'posts')['train']
ds_train_users = load_dataset("smpchallenge/SMP-Video", 'users')['train']
ds_train_videos = load_dataset("smpchallenge/SMP-Video", 'videos')['train']
ds_train_labels = load_dataset("smpchallenge/SMP-Video", 'labels')['train']



def create_train_val_test_split(labels_ds, posts_ds, users_ds, videos_ds, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Create deterministic train/val/test split grouped by pid to prevent leakage.
    Split the labels dataset first, then get corresponding data from other datasets.
    
    Args:
        labels_ds: Processed labels dataset with pid, uid, popularity_log1p
        posts_ds: Posts features dataset
        users_ds: Users features dataset  
        videos_ds: Video features dataset
        train_ratio: Training split ratio (default 0.8)
        val_ratio: Validation split ratio (default 0.1)
        test_ratio: Test split ratio (default 0.1)
        random_seed: Random seed for reproducibility (default 42)
    
    Returns:
        tuple: (train_data, val_data, test_data) where each contains all 4 datasets
    """
    import random
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Get unique post IDs from labels
    unique_pids = list(set(labels_ds["pid"]))
    print(f"Total unique posts: {len(unique_pids)}")
    
    # Shuffle posts deterministically
    random.shuffle(unique_pids)
    
    # Calculate split indices
    n_posts = len(unique_pids)
    n_train = int(n_posts * train_ratio)
    n_val = int(n_posts * val_ratio)
    n_test = n_posts - n_train - n_val  # Handle rounding
    
    # Split posts
    train_pids = set(unique_pids[:n_train])
    val_pids = set(unique_pids[n_train:n_train + n_val])
    test_pids = set(unique_pids[n_train + n_val:])
    
    print(f"Train posts: {len(train_pids)} ({len(train_pids)/n_posts*100:.1f}%)")
    print(f"Val posts: {len(val_pids)} ({len(val_pids)/n_posts*100:.1f}%)")
    print(f"Test posts: {len(test_pids)} ({len(test_pids)/n_posts*100:.1f}%)")
    
    # Function to filter dataset by post IDs
    def filter_by_pids(dataset, pids_to_keep):
        # Get indices where pid is in the set
        indices = [i for i, pid in enumerate(dataset["pid"]) if pid in pids_to_keep]
        return dataset.select(indices)
    
    # Split labels dataset
    train_labels = filter_by_pids(labels_ds, train_pids)
    val_labels = filter_by_pids(labels_ds, val_pids)
    test_labels = filter_by_pids(labels_ds, test_pids)
    
    # Split posts dataset
    train_posts = filter_by_pids(posts_ds, train_pids)
    val_posts = filter_by_pids(posts_ds, val_pids)
    test_posts = filter_by_pids(posts_ds, test_pids)
    
    # Split users dataset - users are filtered by uid, not pid
    # We need to get the uids from the labels first, then filter users
    train_uids = set(train_labels["uid"])
    val_uids = set(val_labels["uid"])
    test_uids = set(test_labels["uid"])
    
    def filter_by_uids(dataset, uids_to_keep):
        # Get indices where uid is in the set
        indices = [i for i, uid in enumerate(dataset["uid"]) if uid in uids_to_keep]
        return dataset.select(indices)
    
    train_users = filter_by_uids(users_ds, train_uids)
    val_users = filter_by_uids(users_ds, val_uids)
    test_users = filter_by_uids(users_ds, test_uids)
    
    # Split videos dataset
    train_videos = filter_by_pids(videos_ds, train_pids)
    val_videos = filter_by_pids(videos_ds, val_pids)
    test_videos = filter_by_pids(videos_ds, test_pids)
    
    # Create data dictionaries
    train_data = {
        "labels": train_labels,
        "posts": train_posts,
        "users": train_users,
        "videos": train_videos
    }
    
    val_data = {
        "labels": val_labels,
        "posts": val_posts,
        "users": val_users,
        "videos": val_videos
    }
    
    test_data = {
        "labels": test_labels,
        "posts": test_posts,
        "users": test_users,
        "videos": test_videos
    }
    
    # Print split statistics
    print(f"\nSplit Statistics:")
    print(f"Train: {len(train_labels)} samples")
    print(f"Val: {len(val_labels)} samples")
    print(f"Test: {len(test_labels)} samples")
    
    return train_data, val_data, test_data

# Create the splits
print(f"\n{'='*60}")
print("CREATING TRAIN/VAL/TEST SPLITS")
print(f"{'='*60}")

processed_targets, train_cap_value = build_target_features(ds_train_labels)
processed_posts = build_post_features(ds_train_posts)
processed_users = build_user_features(ds_train_users)
processed_videos = build_video_features(ds_train_videos)

train_data, val_data, test_data = create_train_val_test_split(
    processed_targets,
   processed_posts, 
    processed_users,
   processed_videos
)

print(f"\n✅ Train/Val/Test splits created successfully!")
print(f"Use train_data, val_data, test_data for your training pipeline.")



def build_master_train_table(train_data, batch_size=1000):
    """
    Build a master training table that joins all features by pid.
    Each row represents one post with all its features and embeddings.
    
    Args:
        train_data: Dictionary containing train splits for labels, posts, users, videos
        batch_size: Batch size for processing
    
    Returns:
        HuggingFace Dataset: Master table with all features joined by pid
    """
    print(f"\n{'='*60}")
    print("BUILDING MASTER TRAINING TABLE")
    print(f"{'='*60}")
    
    # Extract datasets
    train_labels = train_data["labels"]
    train_posts = train_data["posts"]
    train_users = train_data["users"]
    train_videos = train_data["videos"]
    
    print(f"Train labels: {len(train_labels)} samples")
    print(f"Train posts: {len(train_posts)} samples")
    print(f"Train users: {len(train_users)} samples")
    print(f"Train videos: {len(train_videos)} samples")

    post_idx  = {pid: i for i, pid in enumerate(train_posts["pid"])}
    user_idx  = {uid: i for i, uid in enumerate(train_users["uid"])}
    video_idx = {pid: i for i, pid in enumerate(train_videos["pid"])}
    emb_idx = {pid: i for i, pid in enumerate(train_posts["pid"])}
    
    # First, let's join labels with posts (they both have pid, uid)
    print("\nJoining labels with posts...")
    
    def join_labels_posts(batch):
        pids = batch["pid"]
        uids = batch["uid"]
        popularity = batch["popularity_log1p"]

        out = {
            "pid": pids,
            "uid": uids,
            "popularity_log1p": popularity,
        }

        def get_post(field, default):
            # O(1) dict lookup; no inner scans
            return [
                train_posts[field][post_idx[p]] if p in post_idx else default
                for p in pids
            ]

        out.update({
            "hour":          get_post("hour", 0.0),
            "minute":        get_post("minute", 0.0),
            "minute_of_day": get_post("minute_of_day", 0.0),
            "sin_time":      get_post("sin_time", 0.0),
            "cos_time":      get_post("cos_time", 1.0),

            "lang__en":      get_post("lang__en", 0.0),
            "lang__un":      get_post("lang__un", 0.0),
            "lang__es":      get_post("lang__es", 0.0),
            "lang__id":      get_post("lang__id", 0.0),
            "lang__pt":      get_post("lang__pt", 0.0),
            "lang__other":   get_post("lang__other", 0.0),

            "loc__us":       get_post("loc__us", 0.0),
            "loc__gb":       get_post("loc__gb", 0.0),
            "loc__ca":       get_post("loc__ca", 0.0),
            "loc__ph":       get_post("loc__ph", 0.0),
            "loc__au":       get_post("loc__au", 0.0),
            "loc__other":    get_post("loc__other", 0.0),
        })
        return out
    
    # Join labels with posts
    labels_posts_joined = train_labels.map(join_labels_posts, batched=True, batch_size=batch_size)
    
    print("✅ Labels + Posts joined")
    
    # Now join with user features
    print("\nJoining with user features...")
    
    def join_user_features(batch):
        uids = batch["uid"]
        out = {k: v for k, v in batch.items()}

        def get_user(field, default):
            return [
                train_users[field][user_idx[u]] if u in user_idx else default
                for u in uids
            ]

        out.update({
            "log1p_user_following_count": get_user("log1p_user_following_count", 0.0),
            "log1p_user_follower_count":  get_user("log1p_user_follower_count", 0.0),
            "log1p_user_likes_count":     get_user("log1p_user_likes_count", 0.0),
            "log1p_user_video_count":     get_user("log1p_user_video_count", 0.0),
            "log1p_user_digg_count":      get_user("log1p_user_digg_count", 0.0),
            "log1p_user_heart_count":     get_user("log1p_user_heart_count", 0.0),
            "log1p_follower_following_ratio": get_user("log1p_follower_following_ratio", 0.0),
            "log1p_likes_per_video":      get_user("log1p_likes_per_video", 0.0),
            "log1p_hearts_per_video":     get_user("log1p_hearts_per_video", 0.0),
            "log1p_diggs_per_video":      get_user("log1p_diggs_per_video", 0.0),
            "log1p_likes_per_follower":   get_user("log1p_likes_per_follower", 0.0),
        })
        return out
    
    # Join with user features
    all_features_joined = labels_posts_joined.map(join_user_features, batched=True, batch_size=batch_size)
    
    print("✅ User features joined")
    
    # Now join with video features
    print("\nJoining with video features...")
    
    def join_video_features(batch):
        pids = batch["pid"]
        out = {k: v for k, v in batch.items()}

        def get_video(field, default):
            return [
                train_videos[field][video_idx[p]] if p in video_idx else default
                for p in pids
            ]

        out["aspect_ratio"] = get_video("aspect_ratio", 1.0)  # default: square
        out["is_vertical"]  = get_video("is_vertical", 0.0)   # default: horizontal
        return out
    
    # Join with video features
    master_table = all_features_joined.map(join_video_features, batched=True, batch_size=batch_size)
    
    print("✅ Video features joined")
    
    # Now join with video-text embeddings
    print("\nJoining with video-text embeddings...")
    
    # Load embeddings dataset (assuming it's cached)
    try:
        from video import compute_or_load_embeddings
        # Get embeddings for the train posts
        train_posts_for_emb = train_data["posts"].select_columns(["pid", "uid", "video_path", "post_content", "post_suggested_words"])
        embeddings_ds = compute_or_load_embeddings(train_posts_for_emb, cache_dir="./video_text_cache_train")
        
        print(f"✅ Loaded embeddings dataset with {len(embeddings_ds)} samples")
    except Exception as e:
        print(f"⚠️  Warning: Could not load embeddings: {e}")
        print("Creating placeholder embeddings...")
        # Create placeholder embeddings dataset
        

        
    
    emb_idx = {pid: i for i, pid in enumerate(embeddings_ds["pid"])}
    text_embs  = embeddings_ds["text_emb_f16"]
    video_embs = embeddings_ds["video_emb_f16"]
    def join_embeddings(batch):
        pids = batch["pid"]
        out = {k: v for k, v in batch.items()}

        def first_dim(seq):
            for item in seq:
                if isinstance(item, list) and len(item) > 0:
                    return len(item)
            return 512

        text_dim = first_dim(text_embs)
        video_dim = first_dim(video_embs)

        ZERO_TEXT  = [0.0] * text_dim
        ZERO_VIDEO = [0.0] * video_dim

        def get_emb(field_list, default):
            return [
                field_list[emb_idx[p]] if (p in emb_idx and field_list[emb_idx[p]] is not None) else default
                for p in pids
            ]

        out["text_emb_f16"]  = get_emb(text_embs,  ZERO_TEXT)
        out["video_emb_f16"] = get_emb(video_embs, ZERO_VIDEO)
        return out
    
    # Join with embeddings
    master_table_with_emb = master_table.map(join_embeddings, batched=True, batch_size=batch_size)
    
    print("✅ Embeddings joined")
    
    # Print final table info
    print(f"\nMaster table shape: {master_table_with_emb.shape}")
    print(f"Master table columns: {master_table_with_emb.column_names}")
    print(f"Sample row:")
    print(master_table_with_emb[0])
    
    return master_table_with_emb

# Build the master training table
print(f"\n{'='*60}")
print("BUILDING MASTER TRAINING TABLE")
print(f"{'='*60}")

master_train_table = build_master_train_table(train_data)

print(f"\n✅ Master training table created successfully!")
print(f"Shape: {master_train_table.shape}")
print(f"Columns: {len(master_train_table.column_names)}")

def materialize_tensors(master_table, batch_size=1000):
    """
    Materialize training tensors from the master table with strict dtypes and shapes.
    
    Args:
        master_table: Master training table with all features
        batch_size: Batch size for processing
    
    Returns:
        dict: Contains tensors, dimensions, and metadata column order
    """
    print(f"\n{'='*60}")
    print("MATERIALIZING TRAINING TENSORS")
    print(f"{'='*60}")
    
    # Define metadata columns (frozen, no embeddings)
    metadata_columns = [
        # Time features
        "hour", "minute", "minute_of_day", "sin_time", "cos_time",
        # Language one-hots
        "lang__en", "lang__un", "lang__es", "lang__id", "lang__pt", "lang__other",
        # Location one-hots  
        "loc__us", "loc__gb", "loc__ca", "loc__ph", "loc__au", "loc__other",
        # User features
        "log1p_user_following_count", "log1p_user_follower_count", "log1p_user_likes_count",
        "log1p_user_video_count", "log1p_user_digg_count", "log1p_user_heart_count",
        "log1p_follower_following_ratio", "log1p_likes_per_video", "log1p_hearts_per_video",
        "log1p_diggs_per_video", "log1p_likes_per_follower",
        # Video features
        "aspect_ratio", "is_vertical"
    ]
    
    print(f"Metadata columns ({len(metadata_columns)}): {metadata_columns}")
    
    # Verify all columns exist
    missing_cols = [col for col in metadata_columns if col not in master_table.column_names]
    if missing_cols:
        raise ValueError(f"Missing metadata columns: {missing_cols}")
    
    # Get target column
    target_column = "popularity_log1p"
    if target_column not in master_table.column_names:
        raise ValueError(f"Target column '{target_column}' not found")
    
    print(f"Target column: {target_column}")
    
    # Process in batches to build tensors
    def process_batch(batch):
        batch_size = len(batch["pid"])
        
        # Extract metadata features
        meta_features = []
        for col in metadata_columns:
            values = batch[col]
            # Convert to float32, handle any non-numeric values
            try:
                meta_features.append([float(val) if val is not None else 0.0 for val in values])
            except (ValueError, TypeError):
                # If conversion fails, use zeros
                meta_features.append([0.0] * batch_size)
        
        # Extract target
        targets = [float(val) if val is not None else 0.0 for val in batch[target_column]]
        
        return {
            "meta_features": meta_features,
            "targets": targets,
            "pids": batch["pid"]
        }
    
    # Process all batches
    print("\nProcessing batches...")
    meta_batches = []
    all_targets = []
    all_pids = []
    
    for i in range(0, len(master_table), batch_size):
        end_idx = min(i + batch_size, len(master_table))
        batch = master_table.select(range(i, end_idx))
        processed = process_batch(batch)
        
        x_meta_batch = np.asarray(processed["meta_features"], dtype=np.float32).T
        # Safety: ensure rectangular inside the batch
        if x_meta_batch.ndim != 2 or x_meta_batch.shape[0] != len(processed["pids"]) or x_meta_batch.shape[1] != len(metadata_columns):
            lens = [len(col) for col in processed["meta_features"]]
            raise ValueError(f"Non-rectangular batch at rows [{i}:{end_idx}]. "
                            f"Per-column lengths={lens}, expected all == {len(processed['pids'])}")

        meta_batches.append(x_meta_batch)
        all_targets.extend(processed["targets"])
        all_pids.extend(processed["pids"])
        
        if (i // batch_size) % 10 == 0:
            print(f"Processed batch {i//batch_size + 1}/{(len(master_table) + batch_size - 1)//batch_size}")
    
    # Convert to numpy arrays
    print("\nConverting to numpy arrays...")
    
    # Metadata features: [D_meta, N_train] -> transpose to [N_train, D_meta]
    x_meta = np.vstack(meta_batches).astype(np.float32, copy=False)
    y = np.array(all_targets, dtype=np.float32)
    
    print(f"x_meta shape: {x_meta.shape}")
    print(f"y shape: {y.shape}")
    
    # Verify shapes
    N_train = len(all_pids)
    D_meta = len(metadata_columns)
    
    assert x_meta.shape == (N_train, D_meta), f"Expected x_meta shape ({N_train}, {D_meta}), got {x_meta.shape}"
    assert y.shape == (N_train,), f"Expected y shape ({N_train},), got {y.shape}"
    
    # Check for NaNs
    if np.isnan(x_meta).any():
        print("⚠️  Warning: NaN values found in metadata features")
        # Replace NaNs with 0
        x_meta = np.nan_to_num(x_meta, nan=0.0)
    
    if np.isnan(y).any():
        print("⚠️  Warning: NaN values found in targets")
        # Replace NaNs with 0
        y = np.nan_to_num(y, nan=0.0)
    
    # Verify one row per pid
    unique_pids = set(all_pids)
    print(f"Unique PIDs: {len(unique_pids)}")
    print(f"Total rows: {N_train}")
    
    if len(unique_pids) != N_train:
        print("⚠️  Warning: Duplicate PIDs found - not one row per pid")
    
    # Extract real embeddings from the master table
    print("\nExtracting real embeddings...")
    
    # Check if embeddings exist in the master table
    if "text_emb_f16" in master_table.column_names and "video_emb_f16" in master_table.column_names:
        print("✅ Found text and video embeddings in master table")
        
        # Check for excessive zero embeddings (indicates serious issue)
        zero_text_count = 0
        zero_video_count = 0
        total_embeddings = len(master_table)
        
        for i in range(len(master_table)):
            text_emb = master_table["text_emb_f16"][i]
            video_emb = master_table["video_emb_f16"][i]
            
            # Check if embeddings are all zeros
            if text_emb and all(x == 0.0 for x in text_emb):
                zero_text_count += 1
            if video_emb and all(x == 0.0 for x in video_emb):
                zero_video_count += 1
        
        # Print zero embedding statistics
        print(f"Zero embeddings check: text={zero_text_count}/{total_embeddings} ({zero_text_count/total_embeddings*100:.1f}%), video={zero_video_count}/{total_embeddings} ({zero_video_count/total_embeddings*100:.1f}%)")
        
        # Initialize lists for processing
        text_embeddings = []
        video_embeddings = []
        
        # Continue with normal processing...
        for i in range(len(master_table)):
            text_emb = master_table["text_emb_f16"][i]
            video_emb = master_table["video_emb_f16"][i]
            
            # Convert to float32 and ensure they're lists/arrays
            if isinstance(text_emb, list):
                text_emb = np.array(text_emb, dtype=np.float32)
            if isinstance(video_emb, list):
                video_emb = np.array(video_emb, dtype=np.float32)
            
            # Ensure embeddings are 1D arrays
            text_emb = text_emb.flatten()
            video_emb = video_emb.flatten()
            
            text_embeddings.append(text_emb)
            video_embeddings.append(video_emb)
        
        # Convert to numpy arrays
        text_emb = np.array(text_embeddings, dtype=np.float32)
        video_emb = np.array(video_embeddings, dtype=np.float32)
        
        # Get embedding dimensions
        D_text = text_emb.shape[1] if len(text_emb.shape) > 1 else 1
        D_video = video_emb.shape[1] if len(video_emb.shape) > 1 else 1
        
        print(f"Text embeddings shape: {text_emb.shape}")
        print(f"Video embeddings shape: {video_emb.shape}")
        
        # For now, concatenate text and video embeddings as pair_emb
        # You can modify this based on how you want to use the embeddings
        if D_text == D_video:
            pair_emb = np.concatenate([text_emb, video_emb], axis=1)
            D_emb = D_text + D_video
        else:
            # If dimensions don't match, use text embeddings as primary
            pair_emb = text_emb
            D_emb = D_text
            
    else:
        print("⚠️  No embeddings found in master table, creating placeholders...")
        D_emb = 1024  # Default embedding dimension
        pair_emb = np.zeros((N_train, D_emb), dtype=np.float32)
    
    print(f"Final pair_emb shape: {pair_emb.shape}")
    print(f"Embedding dimension: {D_emb}")
    
    # Final verification
    print(f"\nFinal tensor shapes:")
    print(f"x_meta: {x_meta.shape} (float32)")
    print(f"pair_emb: {pair_emb.shape} (float32)") 
    print(f"y: {y.shape} (float32)")
    
    # Record dimensions and metadata info
    tensor_info = {
        "x_meta": x_meta,
        "pair_emb": pair_emb,
        "y": y,
        "meta_cont_dim": D_meta,
        "emb_in_dim": D_emb,
        "metadata_columns": metadata_columns,
        "N_train": N_train,
        "pids": all_pids
    }
    
    # Add individual embeddings if they exist
    if "text_emb_f16" in master_table.column_names and "video_emb_f16" in master_table.column_names:
        tensor_info["text_emb"] = text_emb
        tensor_info["video_emb"] = video_emb
        tensor_info["text_emb_dim"] = D_text
        tensor_info["video_emb_dim"] = D_video
        print(f"✅ Added individual embeddings: text({D_text}), video({D_video})")
    
    return tensor_info

# Materialize training tensors
print(f"\n{'='*60}")
print("MATERIALIZING TRAINING TENSORS")
print(f"{'='*60}")

tensor_info = materialize_tensors(master_train_table)

print(f"\n✅ Training tensors materialized successfully!")
print(f"Metadata dimension: {tensor_info['meta_cont_dim']}")
print(f"Embedding dimension: {tensor_info['emb_in_dim']}")
print(f"Training samples: {tensor_info['N_train']}")
print(f"Metadata columns: {len(tensor_info['metadata_columns'])}")