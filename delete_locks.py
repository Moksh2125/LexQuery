import os
import glob
import shutil

lock_dir = r"D:\.cache\huggingface\hub\.locks"
if os.path.exists(lock_dir):
    for root, dirs, files in os.walk(lock_dir):
        for file in files:
            if file.endswith('.lock'):
                path = os.path.join(root, file)
                try:
                    os.remove(path)
                    print(f"Removed: {path}")
                except Exception as e:
                    print(f"Failed to remove {path}: {e}")
else:
    print("Lock directory not found.")
