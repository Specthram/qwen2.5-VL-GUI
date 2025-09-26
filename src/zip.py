import os
import zipfile


"""Creates a zip archive of the input directory."""
def create_zip_archive(input_folder: str):
    zip_path = "collection.zip"
    print("Creating ZIP archive...")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(input_folder):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=file)
    print(f"Archive created: {zip_path}")
    return zip_path
