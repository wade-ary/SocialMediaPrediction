import os
import re
import numpy as np
import torch
import av
from torchvision.transforms.functional import resize
from transformers import XCLIPProcessor, XCLIPModel
from datasets import load_dataset

#data
ds_train_posts = load_dataset("smpchallenge/SMP-Video", 'posts')['train']
ds_train_users = load_dataset("smpchallenge/SMP-Video", 'users')['train']
ds_train_videos = load_dataset("smpchallenge/SMP-Video", 'videos')['train']
ds_train_labels = load_dataset("smpchallenge/SMP-Video", 'labels')['train']

# Load test dataset
ds_test_posts = load_dataset("smpchallenge/SMP-Video", 'posts')['test']
ds_test_users = load_dataset("smpchallenge/SMP-Video", 'users')['test']
ds_test_videos = load_dataset("smpchallenge/SMP-Video", 'videos')['test']
# -----------------------
# Model init
# -----------------------
MODEL_NAME = "microsoft/xclip-base-patch32"
processor = XCLIPProcessor.from_pretrained(MODEL_NAME)
model = XCLIPModel.from_pretrained(MODEL_NAME)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -----------------------
# Text helpers
# -----------------------
def _as_list(obj):
    if obj is None:
        return []
    if isinstance(obj, list):
        return [str(x) for x in obj]
    return [str(obj)]

def clean_and_join_text(post_content, post_suggested_words):
    """
    Build paired text from post_content + post_suggested_words.
    - lowercase
    - replace commas with spaces
    - collapse whitespace
    """
    parts = []
    for s in _as_list(post_content):
        s = s.lower().replace(",", " ")
        parts.append(s)

    sw = _as_list(post_suggested_words)
    # If a single comma-joined string, split
    if len(sw) == 1 and ("," in sw[0]):
        sw = [w.strip() for w in sw[0].split(",") if w.strip()]
    sw = [w.lower().replace(",", " ") for w in sw]
    if sw:
        parts.append(" ".join(sw))

    text = " ".join(parts)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------
# Video frame sampling
# -----------------------
def sample_video_frames(video_path, num_frames=8, target_size=224):
    """
    Evenly sample num_frames across the video using PyAV.
    Returns list of PIL RGB images (resized). Pads by repeating last frame if needed.
    Raises on decode error so caller can mark status.
    """
    frames = []
    container = av.open(video_path)
    stream = container.streams.video[0]
    total = stream.frames if stream.frames else 0

    # Fallback if frame count is unknown: iterate and count once
    if total <= 0:
        raw = [f.to_image().convert("RGB") for f in container.decode(video=0)]
        total = len(raw)
        if total == 0:
            raise RuntimeError("No frames decoded")
        # choose indices from this decoded list
        idx = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = [resize(raw[j], [target_size, target_size]) for j in idx]
        return frames

    indices = set(np.linspace(0, total - 1, num_frames, dtype=int).tolist())
    collected = 0
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            img = frame.to_image().convert("RGB")
            img = resize(img, [target_size, target_size])
            frames.append(img)
            collected += 1
            if collected == num_frames:
                break

    if len(frames) == 0:
        raise RuntimeError("No frames selected")
    # pad if we got fewer than requested
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames

# -----------------------
# HF map function
# -----------------------
def build_video_text_embeddings(posts_ds, *, batch_size=8, num_frames=8, target_size=224):
    """
    Adds columns to a NEW HF Dataset:
      - paired_text (str)
      - text_emb_f16 (List[float16] or None)
      - video_emb_f16 (List[float16] or None)
      - proc_status (str: 'ok', 'missing_video', 'decode_error', 'empty_text', 'processor_error')
    """
    required = ["pid", "uid", "video_path", "post_content", "post_suggested_words"]
    missing = [c for c in required if c not in posts_ds.column_names]
    if missing:
        raise ValueError(f"Missing required columns in posts_ds: {missing}")

    downloads_root = os.path.expanduser("~/Downloads/video_file")

    def _mapper(batch):
        pids   = batch["pid"]
        uids   = batch["uid"]
        paths  = batch["video_path"]
        conts  = batch["post_content"]
        suggs  = batch["post_suggested_words"]

        n = len(pids)

        paired_texts = []
        abs_paths = []
        status = ["ok"] * n

        # Build paired text and absolute paths
        for i in range(n):
            text = clean_and_join_text(conts[i], suggs[i])
            if not text:
                status[i] = "empty_text"
            paired_texts.append(text)

            rel_path = str(paths[i]) if paths[i] is not None else ""
            # prepend ~/Downloads/, ensure no accidental '//' joins
            abs_path = os.path.join(downloads_root, rel_path.lstrip("/"))
            if not os.path.exists(abs_path):
                status[i] = "missing_video"
            abs_paths.append(abs_path)

        # Sample frames where possible
        frames_list = [None] * n
        valid_idx = []
        for i in range(n):
            if status[i] != "ok":
                continue
            try:
                frames_list[i] = sample_video_frames(abs_paths[i], num_frames=num_frames, target_size=target_size)
                valid_idx.append(i)
            except Exception:
                status[i] = "decode_error"
                frames_list[i] = None

        # Prepare model inputs for valid rows
        text_emb_out  = [None] * n
        video_emb_out = [None] * n

        if valid_idx:
            try:
                vids_batch = [frames_list[i] for i in valid_idx]
                txts_batch = [paired_texts[i] for i in valid_idx]

                inputs = processor(
                    text=txts_batch,
                    videos=vids_batch,
                    return_tensors="pt",
                    padding=True
                )
                # Move to device
                for k in inputs:
                    if isinstance(inputs[k], torch.Tensor):
                        inputs[k] = inputs[k].to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    # (B, D)
                    vemb = outputs.video_embeds
                    temb = outputs.text_embeds

                    # L2-normalize
                    vemb = torch.nn.functional.normalize(vemb, p=2, dim=1)
                    temb = torch.nn.functional.normalize(temb, p=2, dim=1)

                    # to float16 CPU lists
                    vemb = vemb.detach().to("cpu").to(torch.float16).numpy()
                    temb = temb.detach().to("cpu").to(torch.float16).numpy()

                # Scatter back
                for j, i in enumerate(valid_idx):
                    # Get individual embeddings and ensure they're 1D
                    video_emb = vemb[j].reshape(-1)  # Flatten from (1, 512) to (512,)
                    text_emb = temb[j].reshape(-1)   # Flatten from (1, 512) to (512,)
                    
                    # Re-normalize after reshaping to ensure unit norm
                    video_emb = video_emb / np.linalg.norm(video_emb)
                    text_emb = text_emb / np.linalg.norm(text_emb)
                    
                    video_emb_out[i] = video_emb.tolist()
                    text_emb_out[i]  = text_emb.tolist()

            except Exception:
                # If processor/model fails for the whole sub-batch
                for i in valid_idx:
                    status[i] = "processor_error"
                    video_emb_out[i] = None
                    text_emb_out[i]  = None

        return {
            "paired_text": paired_texts,
            "text_emb_f16": text_emb_out,
            "video_emb_f16": video_emb_out,
            "proc_status": status,
        }

    # batched=True uses our _mapper on chunks of rows;
    # batch_size here controls how many rows per map-call (unrelated to XCLIP frame count).
    video_text_features = posts_ds.map(_mapper, batched=True, batch_size=batch_size)
    return video_text_features




N = 1 # try first 10 posts to find one that processes OK
test_slice = ds_train_posts.select(range(min(N, len(ds_train_posts))))
emb_slice = build_video_text_embeddings(test_slice, batch_size=2, num_frames=8)

print("Columns now:", emb_slice.column_names)

# --- 2) Find the first successful row ---
ok_idxs = [i for i, s in enumerate(emb_slice["proc_status"]) if s == "ok"]
if not ok_idxs:
    raise RuntimeError("No successful rows in the test slice. Try increasing N or check video paths.")
i = ok_idxs[0]
row = emb_slice[i]

print("\nSample OK row index:", i)
print("pid:", row["pid"], "uid:", row["uid"])
print("proc_status:", row["proc_status"])
print("paired_text (first 140 chars):", (row["paired_text"] or "")[:140])

# --- 3) Basic validity checks on embeddings ---
v = np.array(row["video_emb_f16"], dtype=np.float16)
t = np.array(row["text_emb_f16"], dtype=np.float16)

# Debug: print shapes and types
print(f"Video embedding shape: {v.shape}, type: {type(v)}, dtype: {v.dtype}")
print(f"Text embedding shape: {t.shape}, type: {type(t)}, dtype: {t.dtype}")
print(f"Video embedding content: {row['video_emb_f16'][:5] if row['video_emb_f16'] else 'None'}")  # First 5 elements
print(f"Text embedding content: {row['text_emb_f16'][:5] if row['text_emb_f16'] else 'None'}")   # First 5 elements

assert v.ndim == 1 and t.ndim == 1, "Embeddings should be 1D vectors"
assert len(v) == len(t) and len(v) > 0, "Video/Text dims must match and be > 0"
assert v.dtype == np.float16 and t.dtype == np.float16, "Embeddings should be float16"

# Check (approx) unit norm after L2-normalize in pipeline
v_norm = np.linalg.norm(v.astype(np.float32))
t_norm = np.linalg.norm(t.astype(np.float32))
print("video norm:", round(float(v_norm), 4), "| text norm:", round(float(t_norm), 4))
assert 0.9 < v_norm < 1.1 and 0.9 < t_norm < 1.1, "Expected near unit-norm embeddings"

# --- 4) Cosine similarity sanity check (should be finite and reasonable) ---
cos_sim = float(np.dot(v.astype(np.float32), t.astype(np.float32)))
print("cosine(video, text):", round(cos_sim, 4))

print("\n✅ Single-item embedding test passed.")
