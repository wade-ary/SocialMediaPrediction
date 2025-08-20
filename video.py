import os
import re
import numpy as np
import torch
import av
from torchvision.transforms.functional import resize
from transformers import XCLIPProcessor, XCLIPModel
from datasets import load_dataset
from datasets import load_from_disk


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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    model = XCLIPModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
else:
    model = XCLIPModel.from_pretrained(MODEL_NAME).to(device)
model.eval()



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
      - text_emb_f16 (List[float16] or None)
      - video_emb_f16 (List[float16] or None)
      - proc_status (str: 'ok', 'empty_text', 'missing_video', 'decode_error', 'processor_error')
      - error_detail (str short trace / message for debugging)
      - abs_video_path (str absolute path used for video)
    """
    import traceback

    required = ["pid", "uid", "video_path", "post_content", "post_suggested_words"]
    missing = [c for c in required if c not in posts_ds.column_names]
    if missing:
        raise ValueError(f"Missing required columns in posts_ds: {missing}")

    downloads_root = os.path.expanduser("~/Downloads/video_file/")

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
        err    = [""] * n

        # Build paired text and absolute paths
        for i in range(n):
            text = clean_and_join_text(conts[i], suggs[i])
            if not text:
                status[i] = "empty_text"
            paired_texts.append(text)

            rel_path = str(paths[i]) if paths[i] is not None else ""
            abs_path = os.path.join(downloads_root, rel_path.lstrip("/"))
            if not os.path.exists(abs_path):
                status[i] = "missing_video"
                err[i] = f"not found: {abs_path}"
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
            except Exception as e:
                status[i] = "decode_error"
                err[i] = (repr(e) + "\n" + traceback.format_exc())[:2000]
                frames_list[i] = None

        text_emb_out  = [None] * n
        video_emb_out = [None] * n

        # Processor + model on valid rows
        if valid_idx:
            try:
                vids_batch = [frames_list[i] for i in valid_idx]
                txts_batch = [paired_texts[i] for i in valid_idx]

                text_inputs = processor.tokenizer(
                    txts_batch,
                    padding=True,
                    truncation=True,
                    max_length=77,        # X-CLIP text max positions = 77
                    return_tensors="pt",
                )

                # build video inputs (image processor only)
                video_inputs = processor.image_processor(
                    images=vids_batch,     # list/array of videos or frames in VideoMAE format
                    return_tensors="pt",
                )

                # merge for the model
                inputs = {**text_inputs, **video_inputs}

                # Move to device
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device, non_blocking=True)

                with torch.inference_mode():
                    if device.type == "cuda":
                        with torch.autocast("cuda", dtype=torch.float16):
                            outputs = model(**inputs)
                    else:
                        outputs = model(**inputs)

                # IMPORTANT: extract outside the device branch
                vemb = outputs.video_embeds
                temb = outputs.text_embeds

                if vemb.ndim == 3:                   # (B, T, D)
                    vemb = vemb.mean(dim=1)          # -> (B, D)
                if temb.ndim == 3:                   # (B, T, D) some HF versions broadcast text over frames
                    temb = temb.mean(dim=1)          # -> (B, D)

                # Sanity: now both must be (B, D)
                assert vemb.ndim == 2 and temb.ndim == 2, f"Unexpected shapes: v={vemb.shape}, t={temb.shape}"

                # To CPU float16 lists
                vemb = vemb.detach().to("cpu").to(torch.float16).numpy()
                temb = temb.detach().to("cpu").to(torch.float16).numpy()

                # Scatter back (no normalization here; keep pipeline pure)
                for j, i in enumerate(valid_idx):
                    video_emb_out[i] = vemb[j].reshape(-1).tolist()
                    text_emb_out[i]  = temb[j].reshape(-1).tolist()

            except Exception as e:
                # Mark processor/model failure for *those* valid rows
                msg = (repr(e) + "\n" + traceback.format_exc())[:2000]
                for i in valid_idx:
                    status[i] = "processor_error"
                    err[i] = msg
                    video_emb_out[i] = None
                    text_emb_out[i]  = None

        return {
            "pid": pids,
            "uid": uids,
            "text_emb_f16": text_emb_out,
            "video_emb_f16": video_emb_out,
            "proc_status": status,
            "error_detail": err,
            "abs_video_path": abs_paths,
        }

    video_text_features = posts_ds.map(_mapper, batched=True, batch_size=batch_size)
    # Keep debug columns!
    keep_cols = ["pid", "uid", "text_emb_f16", "video_emb_f16", "proc_status", "error_detail", "abs_video_path"]
    video_text_features = video_text_features.select_columns(keep_cols)
    return video_text_features


def compute_or_load_embeddings(posts_ds, cache_dir="./video_text_cache_train", **kwargs):
    """
    Load cached dataset if it exists; else compute with build_video_text_embeddings(),
    save to disk once, and return it.
    """
    if os.path.exists(cache_dir):
        return load_from_disk(cache_dir)

    ds = build_video_text_embeddings(posts_ds, **kwargs)
    ds.save_to_disk(cache_dir)
    return ds






