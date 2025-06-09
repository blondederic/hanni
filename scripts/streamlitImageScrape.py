import streamlit as st
import os
import urllib.request
from duckduckgo_search import DDGS
from PIL import Image

# download fn that takes in search, folder, how many images
def download_images(query, folder_name, max_images=30):
    os.makedirs(folder_name, exist_ok=True)

    count = 0
    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=max_images)
        for i, result in enumerate(results):
            if count >= max_images:
                break
            url = result.get("image")
            try:
                file_path = os.path.join(folder_name, f"{query.replace(' ', '_')}_{i}.jpg")
                urllib.request.urlretrieve(url, file_path)
                count += 1
            except Exception as e:
                print(f"❌ failed to download {url}: {e}")
    return count

# streamlit ui
st.title("ricky's image scraper")

query = st.text_input("what do u wanna download?")
folder = st.text_input("what folder?")
image_amount = st.slider("how many images do u want??", 1, 300, 30)

if st.button("start download"):
    if not query or not folder:
        st.warning("please fill in all fields.")
    else:
        with st.spinner("downloading..."):
            count = download_images(query, folder, image_amount)
        st.success(f"✅ downloaded {count} images to '{folder}'")

# 
st.image("https://i.pinimg.com/736x/c1/07/fc/c107fc62e2ec18026ebf840506ace957.jpg")
