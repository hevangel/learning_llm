import os
import zipfile

# Step 1: Move to the parent directory
os.chdir(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
parent_dir = os.getcwd()

# Step 2: Loop through each item in the parent directory
for item in os.listdir(parent_dir):
    item_path = os.path.join(parent_dir, item)
    
    # Only zip directories, skip if directory is called 'models'
    if os.path.isdir(item_path) and item != "models":
        zip_filename = f"{item}.zip"
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(item_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, item_path)
                    zipf.write(file_path, arcname)
        
        print(f"✅ Zipped: {zip_filename}")
