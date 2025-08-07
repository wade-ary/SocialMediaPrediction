from datasets import load_dataset

# Load Train dataset
ds_train_posts = load_dataset("smpchallenge/SMP-Video", 'posts')['train']
ds_train_users = load_dataset("smpchallenge/SMP-Video", 'users')['train']
ds_train_videos = load_dataset("smpchallenge/SMP-Video", 'videos')['train']
ds_train_labels = load_dataset("smpchallenge/SMP-Video", 'labels')['train']

# Load test dataset
ds_test_posts = load_dataset("smpchallenge/SMP-Video", 'posts')['test']
ds_test_users = load_dataset("smpchallenge/SMP-Video", 'users')['test']
ds_test_videos = load_dataset("smpchallenge/SMP-Video", 'videos')['test']

# Examine the train videos dataset
print("=== Train Videos Dataset Info ===")
print(f"Dataset size: {len(ds_train_videos)}")
print(f"Features: {ds_train_videos.features}")
print(f"Column names: {ds_train_videos.column_names}")

# Show first few examples
print("\n=== First 3 examples ===")
for i in range(min(3, len(ds_train_videos))):
    print(f"\nExample {i+1}:")
    example = ds_train_videos[i]
    for key, value in example.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}...")
        else:
            print(f"  {key}: {value}")

# Show some statistics
print(f"\n=== Dataset Statistics ===")
print(f"Total videos: {len(ds_train_videos)}")

