import os
import cv2
import hashlib

# this removes broken or unreadable images
def remove_broken_images(folder):
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        try:
            img = cv2.imread(path)
            if img is None:
                print(f"Broken file removed: {filename}")
                os.remove(path)
        except:
            print(f"Failed to read image: {filename}")
            os.remove(path)

# this removes duplicate images by comparing hashes
def remove_duplicates(folder):
    hashes = set()
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        with open(path, 'rb') as f:
            filehash = hashlib.md5(f.read()).hexdigest()
        if filehash in hashes:
            print(f"Duplicate removed: {filename}")
            os.remove(path)
        else:
            hashes.add(filehash)

# clean both folders
for folder in ["data/hanni", "data/others"]:
    print(f"\nCleaning {folder}...")
    remove_broken_images(folder)
    remove_duplicates(folder)