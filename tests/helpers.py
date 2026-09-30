import os
import shutil
import socket
import urllib
import zipfile

import VISSSlib


def get_test_path():
    path = os.path.dirname(__file__)
    return path


def get_test_data_path():
    path = os.path.join(get_test_path(), "data")
    return path


def readTestSettings(settings):
    testRoot = get_test_data_path()
    settings = f"{testRoot}/{settings}"
    if not os.path.exists(settings):
        downloadData()
    config = VISSSlib.tools.readSettings(settings)
    config["tmpPath"] = f"{os.path.dirname(settings)}/tmp_{socket.gethostname()}"
    config["fileQueue"] = f"{config['tmpPath']}/fileQueue"

    return config


def makeSyntheticConfig(tmp_path, **overrides):
    """Build a minimal VISSSlib config for tests that only exercise
    control-flow (DAG wiring, path templating, ...) and must not touch
    the network or the downloaded sample dataset. Values mirror sample.yaml.
    """
    import yaml

    settings = {
        "computers": ["testcomputer", "testcomputer"],
        "fps": 140,
        "frame_height": 1024,
        "frame_width": 1280,
        "leader": "leader_test",
        "follower": "follower_test",
        "nThreads": 1,
        "path": str(tmp_path / "raw" / "{level}"),
        "pathOut": str(tmp_path / "products" / "{level}"),
        "pathQuicklooks": str(tmp_path / "quicklooks" / "{level}"),
        "visssGen": "visss",
        "site": "test",
        "start": "2026-01-01",
        "end": "today",
        "name": "Synthetic Test Site",
        "model": "M1280",
    }
    settings.update(overrides)

    settingsFile = tmp_path / "settings.yaml"
    with open(settingsFile, "w") as f:
        yaml.dump(settings, f)

    return VISSSlib.tools.readSettings(str(settingsFile))


def downloadData():
    test_path = get_test_path()
    test_data_path = get_test_data_path()
    # remove old data dir if present
    if os.path.exists(test_data_path):
        shutil.rmtree(test_data_path)
    url = "https://speicherwolke.uni-leipzig.de/public.php/dav/files/PJ8dt77ND9tmaB2/?accept=zip"
    zip_path = os.path.join(test_path, "data.zip")
    print(f"Downloading test data from {url} to {zip_path}")
    urllib.request.urlretrieve(url, zip_path)
    print(f"Extracting test data to {test_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(test_path)
    print(f"Removing zip file {zip_path}")
    os.remove(zip_path)
    return
