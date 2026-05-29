import os
import shutil
import kagglehub
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

path = kagglehub.dataset_download("gabriellecharlton/vancouver-based-scrunchie-e-commerce-business")

print("Path to dataset files:", path)

# Where you want the files — change this to whatever you like
dst = Path("/Users/busegungor/model1/data")

dst.mkdir(parents=True, exist_ok=True)
shutil.copytree(path, dst, dirs_exist_ok=True)
print("Saved to:", dst)


