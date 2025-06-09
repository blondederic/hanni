# import ts for file system, pulling images from ddg, downloading files from url
import os
from duckduckgo_search import DDGS
import urllib.request

# function to download images with a search, folder name, and how many to get
def download_images(query, folder_name, max_images=200):
    
    # make the folder if it doesn’t exist already
    os.makedirs(folder_name, exist_ok=True)

    # use duckduckgo to get images based on the search
    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=max_images)

        # go through each result and grab the image link
        for i, result in enumerate(results):
            url = result["image"]
            try:
                # save the image and name it like "Hanni_0.jpg", "Hanni_1.jpg", etc
                file_path = os.path.join(folder_name, f"{query.replace(' ', '_')}_{i}.jpg")
                urllib.request.urlretrieve(url, file_path)
                print(f"Downloaded {file_path}")
            except Exception as e:
                print(f"Failed to download {url}: {e}")

# run the function for both Hanni and Chaewon images
download_images("haerin newjeans face", "data/others", max_images=50)
download_images("hyein newjeans face", "data/others", max_images=50)