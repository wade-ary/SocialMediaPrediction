from data_process import *
from datasets import load_dataset
from collections import Counter
import re
import math
import numpy as np

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

train_data, val_data, test_data = create_train_val_test_split(
    ds_train_labels_processed,
    ds_train_post_features, 
    ds_train_users_features,
    ds_train_video_features
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
    
    # First, let's join labels with posts (they both have pid, uid)
    print("\nJoining labels with posts...")
    
    def join_labels_posts(batch):
        # Get the batch data
        pids = batch["pid"]
        uids = batch["uid"]
        popularity = batch["popularity_log1p"]
        
        # Find corresponding post features
        post_features = {}
        for i, pid in enumerate(pids):
            # Find post with matching pid
            post_idx = None
            for j, post_pid in enumerate(train_posts["pid"]):
                if post_pid == pid:
                    post_idx = j
                    break
            
            if post_idx is not None:
                post_features[i] = {
                    "hour": train_posts["hour"][post_idx],
                    "minute": train_posts["minute"][post_idx],
                    "minute_of_day": train_posts["minute_of_day"][post_idx],
                    "sin_time": train_posts["sin_time"][post_idx],
                    "cos_time": train_posts["cos_time"][post_idx],
                    "lang__en": train_posts["lang__en"][post_idx],
                    "lang__un": train_posts["lang__un"][post_idx],
                    "lang__es": train_posts["lang__es"][post_idx],
                    "lang__id": train_posts["lang__id"][post_idx],
                    "lang__pt": train_posts["lang__pt"][post_idx],
                    "lang__other": train_posts["lang__other"][post_idx],
                    "loc__us": train_posts["loc__us"][post_idx],
                    "loc__gb": train_posts["loc__gb"][post_idx],
                    "loc__ca": train_posts["loc__ca"][post_idx],
                    "loc__ph": train_posts["loc__ph"][post_idx],
                    "loc__au": train_posts["loc__au"][post_idx],
                    "loc__other": train_posts["loc__other"][post_idx]
                }
            else:
                # Default values if post not found
                post_features[i] = {
                    "hour": 0, "minute": 0, "minute_of_day": 0,
                    "sin_time": 0.0, "cos_time": 1.0,
                    "lang__en": 0, "lang__un": 0, "lang__es": 0, "lang__id": 0, "lang__pt": 0, "lang__other": 0,
                    "loc__us": 0, "loc__gb": 0, "loc__ca": 0, "loc__ph": 0, "loc__au": 0, "loc__other": 0
                }
        
        # Build output batch
        out = {
            "pid": pids,
            "uid": uids,
            "popularity_log1p": popularity
        }
        
        # Add post features
        for feature in ["hour", "minute", "minute_of_day", "sin_time", "cos_time",
                       "lang__en", "lang__un", "lang__es", "lang__id", "lang__pt", "lang__other",
                       "loc__us", "loc__gb", "loc__ca", "loc__ph", "loc__au", "loc__other"]:
            out[feature] = [post_features[i][feature] for i in range(len(pids))]
        
        return out
    
    # Join labels with posts
    labels_posts_joined = train_labels.map(join_labels_posts, batched=True, batch_size=batch_size)
    
    print("✅ Labels + Posts joined")
    
    # Now join with user features
    print("\nJoining with user features...")
    
    def join_user_features(batch):
        pids = batch["pid"]
        uids = batch["uid"]
        
        # Find corresponding user features
        user_features = {}
        for i, uid in enumerate(uids):
            # Find user with matching uid
            user_idx = None
            for j, user_uid in enumerate(train_users["uid"]):
                if user_uid == uid:
                    user_idx = j
                    break
            
            if user_idx is not None:
                user_features[i] = {
                    "log1p_user_following_count": train_users["log1p_user_following_count"][user_idx],
                    "log1p_user_follower_count": train_users["log1p_user_follower_count"][user_idx],
                    "log1p_user_likes_count": train_users["log1p_user_likes_count"][user_idx],
                    "log1p_user_video_count": train_users["log1p_user_video_count"][user_idx],
                    "log1p_user_digg_count": train_users["log1p_user_digg_count"][user_idx],
                    "log1p_user_heart_count": train_users["log1p_user_heart_count"][user_idx],
                    "log1p_follower_following_ratio": train_users["log1p_follower_following_ratio"][user_idx],
                    "log1p_likes_per_video": train_users["log1p_likes_per_video"][user_idx],
                    "log1p_hearts_per_video": train_users["log1p_hearts_per_video"][user_idx],
                    "log1p_diggs_per_video": train_users["log1p_diggs_per_video"][user_idx],
                    "log1p_likes_per_follower": train_users["log1p_likes_per_follower"][user_idx]
                }
            else:
                # Default values if user not found
                user_features[i] = {
                    "log1p_user_following_count": 0.0, "log1p_user_follower_count": 0.0,
                    "log1p_user_likes_count": 0.0, "log1p_user_video_count": 0.0,
                    "log1p_user_digg_count": 0.0, "log1p_user_heart_count": 0.0,
                    "log1p_follower_following_ratio": 0.0, "log1p_likes_per_video": 0.0,
                    "log1p_hearts_per_video": 0.0, "log1p_diggs_per_video": 0.0,
                    "log1p_likes_per_follower": 0.0
                }
        
        # Add user features to existing batch
        out = {k: v for k, v in batch.items()}
        
        for feature in ["log1p_user_following_count", "log1p_user_follower_count", "log1p_user_likes_count",
                       "log1p_user_video_count", "log1p_user_digg_count", "log1p_user_heart_count",
                       "log1p_follower_following_ratio", "log1p_likes_per_video", "log1p_hearts_per_video",
                       "log1p_diggs_per_video", "log1p_likes_per_follower"]:
            out[feature] = [user_features[i][feature] for i in range(len(pids))]
        
        return out
    
    # Join with user features
    all_features_joined = labels_posts_joined.map(join_user_features, batched=True, batch_size=batch_size)
    
    print("✅ User features joined")
    
    # Now join with video features
    print("\nJoining with video features...")
    
    def join_video_features(batch):
        pids = batch["pid"]
        
        # Find corresponding video features
        video_features = {}
        for i, pid in enumerate(pids):
            # Find video with matching pid
            video_idx = None
            for j, video_pid in enumerate(train_videos["pid"]):
                if video_pid == pid:
                    video_idx = j
                    break
            
            if video_idx is not None:
                video_features[i] = {
                    "aspect_ratio": train_videos["aspect_ratio"][video_idx],
                    "is_vertical": train_videos["is_vertical"][video_idx]
                }
            else:
                # Default values if video not found
                video_features[i] = {
                    "aspect_ratio": 1.0,  # Default square aspect ratio
                    "is_vertical": 0       # Default horizontal
                }
        
        # Add video features to existing batch
        out = {k: v for k, v in batch.items()}
        
        out["aspect_ratio"] = [video_features[i]["aspect_ratio"] for i in range(len(pids))]
        out["is_vertical"] = [video_features[i]["is_vertical"] for i in range(len(pids))]
        
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
        from datasets import Dataset
        placeholder_embeddings = {
            "pid": train_data["posts"]["pid"],
            "uid": train_data["posts"]["uid"],
            "text_emb_f16": [[0.0] * 512] * len(train_data["posts"]),
            "video_emb_f16": [[0.0] * 512] * len(train_data["posts"])
        }
        embeddings_ds = Dataset.from_dict(placeholder_embeddings)
    
    def join_embeddings(batch):
        pids = batch["pid"]
        
        # Find corresponding embeddings
        embeddings = {}
        for i, pid in enumerate(pids):
            # Find embedding with matching pid
            emb_idx = None
            for j, emb_pid in enumerate(embeddings_ds["pid"]):
                if emb_pid == pid:
                    emb_idx = j
                    break
            
            if emb_idx is not None:
                embeddings[i] = {
                    "text_emb_f16": embeddings_ds["text_emb_f16"][emb_idx],
                    "video_emb_f16": embeddings_ds["video_emb_f16"][emb_idx]
                }
            else:
                # Default values if embedding not found
                embeddings[i] = {
                    "text_emb_f16": [0.0] * 512,
                    "video_emb_f16": [0.0] * 512
                }
        
        # Add embeddings to existing batch
        out = {k: v for k, v in batch.items()}
        
        out["text_emb_f16"] = [embeddings[i]["text_emb_f16"] for i in range(len(pids))]
        out["video_emb_f16"] = [embeddings[i]["video_emb_f16"] for i in range(len(pids))]
        
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
    all_meta_features = []
    all_targets = []
    all_pids = []
    
    for i in range(0, len(master_table), batch_size):
        end_idx = min(i + batch_size, len(master_table))
        batch = master_table.select(range(i, end_idx))
        processed = process_batch(batch)
        
        all_meta_features.extend(processed["meta_features"])
        all_targets.extend(processed["targets"])
        all_pids.extend(processed["pids"])
        
        if (i // batch_size) % 10 == 0:
            print(f"Processed batch {i//batch_size + 1}/{(len(master_table) + batch_size - 1)//batch_size}")
    
    # Convert to numpy arrays
    print("\nConverting to numpy arrays...")
    
    # Metadata features: [D_meta, N_train] -> transpose to [N_train, D_meta]
    x_meta = np.array(all_meta_features, dtype=np.float32).T
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
        
        # Extract embeddings and convert to numpy arrays
        text_embeddings = []
        video_embeddings = []
        
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