import cv2
import torch
from transformers import CLIPModel, CLIPImageProcessorFast, CLIPTokenizerFast
from torchvision.transforms import ToPILImage

# Load CLIP
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPImageProcessorFast.from_pretrained(model_name)
model.eval()

# Video settings
video_path = "/Users/aryamanwade/Desktop/bodyai/VIDEO00001442.mp4"
num_frames = 9

def extract_frames_uniform(video_path, num_frames=9):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError(f"Video appears to have no frames: {video_path}")

    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frame_indices = list(set(frame_indices))  # avoid duplicates for very short videos

    frames = []
    current_idx = 0
    to_pil = ToPILImage()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in frame_indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = to_pil(torch.from_numpy(frame_rgb).permute(2, 0, 1).byte())
            frames.append(pil_image)
        current_idx += 1

    cap.release()
    return frames

# 1. Extract frames
frames = extract_frames_uniform(video_path, num_frames=num_frames)

print(f"Extracted {len(frames)} frames")
if len(frames) == 0:
    raise ValueError("No frames extracted from video. Check video path or codec.")

# 2. Preprocess with CLIPImageProcessorFast
inputs = processor(images=frames, return_tensors="pt")

# 3. Extract image features
with torch.no_grad():
    image_features = model.get_image_features(**inputs)  # shape: (9, 512)

# 4. Mean-pool over frames
video_feat = image_features.mean(dim=0)  # shape: (512,)

print("Video embedding shape:", video_feat.shape)
print("First 10 values of video embedding:", video_feat[:10].tolist())


# Example text (caption, hashtags, etc.)
text = "a cat playing with a toy"  # could also be caption + hashtags
tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
model.eval()
# Tokenize
inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)

# Get text embedding
with torch.no_grad():
    text_features = model.get_text_features(**inputs)  # shape: (1, 512)

# Optional: normalize embedding
text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)

# Output
print("Text embedding shape:", text_features.shape)
print("First 10 values:", text_features[0][:10].tolist())