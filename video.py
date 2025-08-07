import torch
import numpy as np
from transformers import XCLIPProcessor, XCLIPModel
import av
from torchvision.transforms.functional import resize

# Choose pretrained XCLIP model
model_name = "microsoft/xclip-base-patch32"

# Load the processor and model
processor = XCLIPProcessor.from_pretrained(model_name)
model = XCLIPModel.from_pretrained(model_name)
model.eval()

def sample_video_frames(video_path, num_frames=8, target_size=224):
    container = av.open(video_path)
    total = container.streams.video[0].frames
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            img = frame.to_image().convert("RGB")
            img = resize(img, [target_size, target_size])
            frames.append(img)
        if len(frames) == num_frames:
            break
    return frames

# 2. Load frames
frames = sample_video_frames("/Users/aryamanwade/Desktop/bodyai/VIDEO00001442.mp4", num_frames=8)

caption = "a person dancing in a club"

# Prepare inputs for the model
inputs = processor(text=[caption], videos=[frames], return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)

# Extract embeddings
video_embedding = outputs.video_embeds  # shape: (batch_size, hidden_dim)
text_embedding = outputs.text_embeds    # [CLS] for text

print("Video embedding shape:", video_embedding.shape)
print("Text embedding shape:", text_embedding.shape)