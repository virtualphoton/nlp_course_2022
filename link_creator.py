import os
import shutil

from pathlib import Path

cur_dir = Path(".")
storage = Path("/mnt/10A6-8381/books/SDA/3/00_future/01_NLP/storage")

for subdir in cur_dir.glob("*"):
    if subdir.is_dir() and subdir.stem.startswith("week"):
        sub_storage = storage / subdir.stem
        if not sub_storage.exists():
            sub_storage.mkdir()
        new_path = subdir.absolute() / "storage"
        if not new_path.exists():
            os.symlink(sub_storage, subdir.absolute() / "storage")
            
        os.symlink("/home/photon/PyProjects/06_ds_utils", subdir / "ds_utils")
        for file in subdir.rglob("*"):
            if file.suffix in [".pt", ".pth"]:
                print(file)
                shutil.move(file, sub_storage / file.name)
