import kagglehub
import shutil
import os

# 1. Download the dataset (Kagglehub handles the 2GB download to a cache folder)
print("Downloading dataset... Please wait (this depends on your internet).")
path = kagglehub.dataset_download("hadikp/resume-data-pdf")
print(f" Dataset downloaded to cache: {path}")

# 2. Define the exact source and your destination
# Based on the dataset structure, the resumes are inside a 'Resumes PDF' or 'data' subfolder
# Let's check the path to be sure
src_folder_name = "DataScience" 
dst = "data/raw/data/Data_Science"

# We search for the 'Data Science' folder within the downloaded path
src = None
for root, dirs, files in os.walk(path):
    if src_folder_name in dirs:
        src = os.path.join(root, src_folder_name)
        break

if src:
    print(f"Found source folder at: {src}")
    
    # Create project directory if it doesn't exist
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    
    # Clear old data if it exists to save space
    if os.path.exists(dst):
        shutil.rmtree(dst)
    
    # 3. Copy only what you need
    shutil.copytree(src, dst)
    print(f"Successfully copied Data Science resumes to: {dst}")

    count = len([f for f in os.listdir(dst) if f.endswith('.pdf')])
    print(f"📄 Total PDFs copied: {count}")
    
    # 4. CRITICAL FOR 4GB RAM: Instructions to clear cache
    print("\nTo save 2GB of disk space, you can now manually delete the cache at:")
    print(path)
else:
    print(f"Could not find folder '{src_folder_name}' in the downloaded dataset.")
    print("Available folders were:", os.listdir(path))