import shutil, os
cache_dir = "./video_text_cache_train"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)