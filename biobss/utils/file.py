import os
import urllib.request
import zipfile
from pathlib import Path


def unzip(file_path: str, dest_dir: str):
    """
    Unzips compressed .zip file.
    Example inputs:
        file_path: 'data/01_alb_id.zip'
        dest_dir: 'data/'
    """

    # unzip file
    with zipfile.ZipFile(file_path) as zf:
        zf.extractall(dest_dir)


def download_from_url(from_url: str, to_path: str):
    Path(to_path).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(to_path):
        urllib.request.urlretrieve(
            from_url,
            to_path,
        )
