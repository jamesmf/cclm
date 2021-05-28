from importlib.metadata import version
import requests
import os
import tarfile

cclm_version = "v" + version("cclm")


class Shelf:
    """
    A Shelf is a place to store components and preprocessors. The Shelf
    class facilitates pulling from and pushing to a repository.
    """

    def __init__(
        self,
        repo_url: str = "https://github.com/jamesmf/cclm-shelf/raw/{tag}/{item_type}/{filename}",
    ):
        self.repo_url = repo_url

    def fetch(
        self,
        identifier: str,
        item_type: str,
        tag: str = str(cclm_version),
        cache_dir=".shelf",
    ):
        filename = identifier + ".tar.gz"
        path = os.path.join(cache_dir, identifier, item_type)
        if not os.path.exists(path):
            url = self.repo_url.format(tag=tag, item_type=item_type, filename=filename)
            print(url)

            resp = requests.get(url)
            print(resp)

            os.makedirs(path, exist_ok=True)
            tar_path = os.path.join(path, filename)
            with open(tar_path, "wb") as f:
                f.write(resp.content)
            tar = tarfile.open(tar_path)
            tar.extractall(path=path)