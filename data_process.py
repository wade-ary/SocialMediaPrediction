from datasets import load_dataset
from collections import Counter
import re
import math

# Load Train dataset
ds_train_posts = load_dataset("smpchallenge/SMP-Video", 'posts')['train']
ds_train_users = load_dataset("smpchallenge/SMP-Video", 'users')['train']
ds_train_videos = load_dataset("smpchallenge/SMP-Video", 'videos')['train']
ds_train_labels = load_dataset("smpchallenge/SMP-Video", 'labels')['train']

# Load test dataset
ds_test_posts = load_dataset("smpchallenge/SMP-Video", 'posts')['test']
ds_test_users = load_dataset("smpchallenge/SMP-Video", 'users')['test']
ds_test_videos = load_dataset("smpchallenge/SMP-Video", 'videos')['test']
print("\nSample train user features:")
print(ds_train_users[0]) 
import numpy as np

def build_user_features(users_ds):
    """
    Returns a NEW HuggingFace Dataset with user feature columns added.
    Original users_ds is not modified.
    """
    required = [
        "user_following_count",
        "user_follower_count",
        "user_likes_count",
        "user_video_count",
        "user_digg_count",
        "user_heart_count",
    ]
    missing = [c for c in required if c not in users_ds.column_names]
    if missing:
        raise ValueError(f"Missing required columns in users_ds: {missing}")

    def _add_features(batch):
        # Pull as numpy, clip negatives to 0 (defensive), then smooth with +1 where needed
        following = np.maximum(np.asarray(batch["user_following_count"], dtype=np.float64), 0.0)
        followers = np.maximum(np.asarray(batch["user_follower_count"], dtype=np.float64), 0.0)
        likes     = np.maximum(np.asarray(batch["user_likes_count"],     dtype=np.float64), 0.0)
        videos    = np.maximum(np.asarray(batch["user_video_count"],     dtype=np.float64), 0.0)
        diggs     = np.maximum(np.asarray(batch["user_digg_count"],      dtype=np.float64), 0.0)
        hearts    = np.maximum(np.asarray(batch["user_heart_count"],     dtype=np.float64), 0.0)

        # ---- Ratios (with +1 smoothing) ----
        follower_following_ratio = (followers + 1.0) / (following + 1.0)
        likes_per_video          = (likes     + 1.0) / (videos    + 1.0)
        hearts_per_video         = (hearts    + 1.0) / (videos    + 1.0)
        diggs_per_video          = (diggs     + 1.0) / (videos    + 1.0)
        likes_per_follower       = (likes     + 1.0) / (followers + 1.0)

        # ---- Log1p of raw counts ----
        log1p_following = np.log1p(following)
        log1p_followers = np.log1p(followers)
        log1p_likes     = np.log1p(likes)
        log1p_videos    = np.log1p(videos)
        log1p_diggs     = np.log1p(diggs)
        log1p_hearts    = np.log1p(hearts)

        # ---- Log1p of ratios----
        log1p_follower_following_ratio = np.log1p(follower_following_ratio)
        log1p_likes_per_video          = np.log1p(likes_per_video)
        log1p_hearts_per_video         = np.log1p(hearts_per_video)
        log1p_diggs_per_video          = np.log1p(diggs_per_video)
        log1p_likes_per_follower       = np.log1p(likes_per_follower)

        return {
            # ratios
            "follower_following_ratio": follower_following_ratio.tolist(),
            "likes_per_video":          likes_per_video.tolist(),
            "hearts_per_video":         hearts_per_video.tolist(),
            "diggs_per_video":          diggs_per_video.tolist(),
            "likes_per_follower":       likes_per_follower.tolist(),

            # log1p raw counts
            "log1p_user_following_count": log1p_following.tolist(),
            "log1p_user_follower_count":  log1p_followers.tolist(),
            "log1p_user_likes_count":     log1p_likes.tolist(),
            "log1p_user_video_count":     log1p_videos.tolist(),
            "log1p_user_digg_count":      log1p_diggs.tolist(),
            "log1p_user_heart_count":     log1p_hearts.tolist(),

            # log1p ratios
            "log1p_follower_following_ratio": log1p_follower_following_ratio.tolist(),
            "log1p_likes_per_video":          log1p_likes_per_video.tolist(),
            "log1p_hearts_per_video":         log1p_hearts_per_video.tolist(),
            "log1p_diggs_per_video":          log1p_diggs_per_video.tolist(),
            "log1p_likes_per_follower":       log1p_likes_per_follower.tolist(),
        }

    # Map returns a NEW dataset; original users_ds remains unchanged
    users_features = users_ds.map(_add_features, batched=True, batch_size=1000)
    return users_features

ds_train_users_features = build_user_features(ds_train_users)
ds_test_users_features = build_user_features(ds_test_users)


print("Train users shape:", ds_train_users.shape)
print("Test  users shape:", ds_test_users_features.shape)

print("\nSample train user features:")
print(ds_train_users_features[0])  # first row

print("\nNew columns added:")
print([col for col in ds_train_users_features.column_names if "log1p" in col or "_per_" in col])

def build_video_features(videos_ds, batch_size=1000, vertical_threshold=1.0):
    """
    Add engineered video features to a NEW HuggingFace Dataset:
      - aspect_ratio = video_width / video_height  (safe divide)
      - is_vertical = 1 if (video_height / video_width) >= vertical_threshold else 0
    Keeps original columns like video_height and video_width.
    """
    required = ["video_height", "video_width"]
    missing = [c for c in required if c not in videos_ds.column_names]
    if missing:
        raise ValueError(f"Missing required columns in videos_ds: {missing}")

    def _add_features(batch):
        height = np.asarray(batch["video_height"], dtype=np.float64)
        width  = np.asarray(batch["video_width"],  dtype=np.float64)

        # Safe divide: avoid divide-by-zero without altering originals
        aspect_ratio = np.divide(width, np.maximum(height, 1.0))

        # Vertical flag: 1 if height/width >= threshold (default 1.0 means height >= width)
        hv = np.divide(height, np.maximum(width, 1.0))
        is_vertical = (hv >= vertical_threshold).astype(np.int64)

        return {
            "aspect_ratio": aspect_ratio.tolist(),
            "is_vertical":  is_vertical.tolist(),
        }

    return videos_ds.map(_add_features, batched=True, batch_size=batch_size)



ds_train_video_features = build_video_features(ds_train_videos)



print("Train users shape:", ds_train_video_features.shape)


print("\nSample train user features:")
print(ds_train_video_features[0])  # first row

print("\nNew columns added:")
print([col for col in ds_train_video_features.column_names if "aspect_ratio" in col or "is_vertical" in col])



def _slug(s: str) -> str:
    # Safe column name: lowercase, alnum+underscore only
    return re.sub(r'[^0-9a-zA-Z]+', '_', (s or 'unknown').strip().lower()).strip('_')



def _build_one_hot_vocab(values, top_k=None, min_count=1):
    """
    Build a list of category names to one-hot encode.
    If top_k is set, keep the top_k most frequent; else keep all with count >= min_count.
    """
    counts = Counter([v if v is not None else "unknown" for v in values])
    if top_k is not None and top_k > 0:
        cats = [c for c, _ in counts.most_common(top_k)]
    else:
        cats = [c for c, n in counts.items() if n >= min_count]
    # Ensure deterministic order
    cats = sorted(cats, key=lambda x: (_slug(x)))
    return cats, counts

def build_post_features(posts_ds, *, batch_size=1000, top_k_lang=None, top_k_loc=None, min_count=1):
    """
    Returns a NEW HF Dataset with:
      - time features: hour, minute, minute_of_day, sin_time, cos_time
      - one-hot for post_text_language and post_location (with optional top-K + OTHER bucket)
    Identifiers (pid, uid) are kept but not transformed.
    Columns ignored for now: post_content, post_suggested_words, video_path.
    """
    required = ["post_time", "post_text_language", "post_location"]
    missing = [c for c in required if c not in posts_ds.column_names]
    if missing:
        raise ValueError(f"Missing required columns in posts_ds: {missing}")

    # ----- Build vocabularies for one-hot (on full dataset) -----
    langs_all = posts_ds["post_text_language"]
    locs_all  = posts_ds["post_location"]

    lang_vocab, lang_counts = _build_one_hot_vocab(langs_all, top_k=top_k_lang, min_count=min_count)
    loc_vocab,  loc_counts  = _build_one_hot_vocab(locs_all,  top_k=top_k_loc,  min_count=min_count)

    lang_set = set(lang_vocab)
    loc_set  = set(loc_vocab)

    # Column names for one-hot
    lang_cols = [f"lang__{_slug(c)}" for c in lang_vocab]
    loc_cols  = [f"loc__{_slug(c)}"  for c in loc_vocab]
    has_lang_other = True
    has_loc_other  = True
    lang_other_col = "lang__other" if has_lang_other else None
    loc_other_col  = "loc__other"  if has_loc_other else None

    # ----- Mapper -----
    def _add_features(batch):
        times = batch["post_time"]
        langs = batch["post_text_language"]
        locs  = batch["post_location"]

        n = len(times)

        # Time features
        hour = np.zeros(n, dtype=np.int16)
        minute = np.zeros(n, dtype=np.int16)
        minute_of_day = np.zeros(n, dtype=np.int32)

        for i, ts in enumerate(times):
            if ts is None:
                h, m = 0, 0
            else:
                # If it's already a Python datetime
                if hasattr(ts, "hour") and hasattr(ts, "minute"):
                    h, m = ts.hour, ts.minute
                else:
                    # If it's a NumPy datetime64
                    dt = np.datetime64(ts, 's').astype(object)
                    h, m = dt.hour, dt.minute
            hour[i] = h
            minute[i] = m
            minute_of_day[i] = h * 60 + m

        # Cyclical encoding (period = 1440 minutes)
        angle = (2.0 * math.pi * minute_of_day.astype(np.float64)) / 1440.0
        sin_time = np.sin(angle)
        cos_time = np.cos(angle)

        # One-hot encoding for language/location
        # Initialize zeros for all defined columns in this batch
        out = {
            "hour": hour.tolist(),
            "minute": minute.tolist(),
            "minute_of_day": minute_of_day.tolist(),
            "sin_time": sin_time.tolist(),
            "cos_time": cos_time.tolist(),
        }

        # Prepare zero arrays for one-hots
        for col in lang_cols:
            out[col] = [0] * n
        if has_lang_other:
            out[lang_other_col] = [0] * n

        for col in loc_cols:
            out[col] = [0] * n
        if has_loc_other:
            out[loc_other_col] = [0] * n

        # Fill one-hots
        for i in range(n):
            lang = langs[i] if langs[i] is not None else "unknown"
            loc  = locs[i]  if locs[i]  is not None else "unknown"

            if lang in lang_set:
                out[f"lang__{_slug(lang)}"][i] = 1
            elif has_lang_other:
                out[lang_other_col][i] = 1

            if loc in loc_set:
                out[f"loc__{_slug(loc)}"][i] = 1
            elif has_loc_other:
                out[loc_other_col][i] = 1

        return out

    posts_features = posts_ds.map(_add_features, batched=True, batch_size=batch_size)
    return posts_features


ds_train_post_features = build_post_features(ds_train_posts)
print("Train users shape:", ds_train_post_features.shape)


print("\nSample train user features:")
print(ds_train_post_features[0])  # first row

print("\nNew columns added:")
print([col for col in ds_train_post_features.column_names])