"""Deposit derived datasets + code archive to OSF node ruhy8 (private).

Idempotent/resumable: skips any file already present in the target folder with the same
byte size; overwrites (new version) if size differs. Folder layout follows DATA_MANIFEST.md.
Run:  uv run python osf_deposit.py
Token read from OSF_TOKEN.txt (gitignored).
"""
import os
import sys
import time
import requests
from urllib.parse import urlsplit, urlunsplit, urlencode

NODE = "ruhy8"
TOK = open("OSF_TOKEN.txt").read().strip()
WB = f"https://files.osf.io/v1/resources/{NODE}/providers/osfstorage/"
H = {"Authorization": f"Bearer {TOK}"}

# folder -> list of local files to place in it
SHA = os.environ.get("CODE_SHA", "490a527")
PLAN = {
    "primary-2025-09": [
        "data/selected.parquet", "data/depth_traj.parquet",
        "data/maia_q.parquet", "data/train.pt", "data/model.pt",
    ],
    "replication-2026-05": [
        "data/repl/selected.parquet", "data/repl/depth_traj.parquet",
        "data/repl/maia_q.parquet", "data/repl/train.pt",
    ],
    "replication-2026-04": [
        "data/repl04/selected.parquet", "data/repl04/depth_traj.parquet",
        "data/repl04/maia_q.parquet", "data/repl04/train.pt",
    ],
    "chesscom": [
        "data/chesscom/selected.parquet", "data/chesscom/depth_traj.parquet",
        "data/chesscom/maia_q.parquet", "data/chesscom/train.pt",
    ],
    "code": [f"/tmp/depth-of-satisficing-code-{SHA}.zip"],
}


def _base(url):
    """Strip any query string; WaterButler upload links already carry ?kind=file."""
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, p.path, "", ""))


def _with(url, **params):
    return _base(url) + "?" + urlencode(params)


def _req(method, url, upload_path=None, timeout=120):
    """HTTP with retry+backoff on transient 5xx / 429 / connection errors.
    For uploads pass upload_path (reopened each attempt, since the body stream is consumed)."""
    for attempt in range(6):
        try:
            if upload_path is not None:
                with open(upload_path, "rb") as f:
                    r = requests.request(method, url, headers=H, data=f, timeout=timeout)
            else:
                r = requests.request(method, url, headers=H, timeout=timeout)
        except requests.exceptions.RequestException as e:
            if attempt == 5:
                raise
            why = type(e).__name__
        else:
            if r.status_code < 500 and r.status_code != 429:
                return r
            if attempt == 5:
                return r
            why = f"HTTP {r.status_code}"
        wait = min(2 ** attempt, 30)
        print(f" [retry {attempt + 1}/6 after {why}; {wait}s]", end="", flush=True)
        time.sleep(wait)
    return r


def list_dir(url):
    r = _req("GET", _base(url))
    r.raise_for_status()
    return {d["attributes"]["name"]: d for d in r.json()["data"]}


def ensure_folder(name):
    root = list_dir(WB)
    if name in root and root[name]["attributes"]["kind"] == "folder":
        return root[name]["links"]["upload"]
    r = _req("PUT", _with(WB, kind="folder", name=name))
    r.raise_for_status()
    return r.json()["data"]["links"]["upload"]


def upload(folder_url, path):
    name = os.path.basename(path)
    sz = os.path.getsize(path)
    existing = list_dir(folder_url)
    if name in existing and existing[name]["attributes"].get("size") == sz:
        print(f"  skip  {name:32s} ({sz/1048576:8.1f} MB, already present)", flush=True)
        return
    print(f"  PUT   {name:32s} ({sz/1048576:8.1f} MB) ...", end="", flush=True)
    if name in existing:  # exists, different size -> overwrite (new version) via its own link
        url = _with(existing[name]["links"]["upload"], kind="file")
    else:                 # new file into the folder
        url = _with(folder_url, kind="file", name=name)
    r = _req("PUT", url, upload_path=path, timeout=3600)
    r.raise_for_status()
    print(" done", flush=True)


def main():
    # sanity: confirm token + node
    me = requests.get("https://api.osf.io/v2/users/me/", headers=H, timeout=60)
    me.raise_for_status()
    print("auth OK:", me.json()["data"]["attributes"]["full_name"], "| node:", NODE, flush=True)
    for folder, files in PLAN.items():
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            print(f"WARNING {folder}: missing locally -> {missing}", flush=True)
        files = [f for f in files if os.path.exists(f)]
        if not files:
            continue
        print(f"[{folder}]", flush=True)
        url = ensure_folder(folder)
        for f in files:
            upload(url, f)
    print("DEPOSIT COMPLETE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
