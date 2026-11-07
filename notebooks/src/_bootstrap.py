# Bootstrap. The repo is private: put a GitHub token (repo read access) in Colab Secrets as
# GH_TOKEN. Set REPO_URL = None to copy the code from Drive instead (MyDrive/marwari-eot/code).
REPO_URL = "https://github.com/krishagarwal314/marwari-eot.git"
REPO_DIR = "/content/marwari-eot"
DRIVE_CODE = "/content/drive/MyDrive/marwari-eot/code"

import os, subprocess, sys
if REPO_URL:
    url = REPO_URL
    try:
        from google.colab import userdata
        tok = userdata.get("GH_TOKEN")
        url = REPO_URL.replace("https://", f"https://{tok}@")
    except Exception:
        pass
    if not os.path.isdir(f"{REPO_DIR}/.git"):
        subprocess.run(["git", "clone", "-q", url, REPO_DIR], check=True)
    else:
        subprocess.run(["git", "-C", REPO_DIR, "pull", "-q", "--ff-only"], check=False)
else:
    from google.colab import drive
    drive.mount("/content/drive")
    subprocess.run(["rsync", "-a", "--delete", f"{DRIVE_CODE}/", f"{REPO_DIR}/"], check=True)
sys.path.insert(0, REPO_DIR)
%pip install -q -r {REPO_DIR}/requirements.txt
