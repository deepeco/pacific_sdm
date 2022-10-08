import os
import zipfile

SOURCE_PATH = '..\\wcdata'


for dir_, dirnames, filenames in os.walk(SOURCE_PATH, followlinks=True):
    for filename in filenames:
        if filename.lower().endswith(".zip"):
            file_path = os.path.join(dir_, filename)
            print(f"Extracting {file_path}...")
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(dir_)
            except:
                print(f"something went wrong with file: {file_path}")








