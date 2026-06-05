"""
Stage 0: fetch one month of the Lichess open database (standard rated, .pgn.zst).

Lichess throttles single connections hard (~35 KiB/s); it allows a higher AGGREGATE over
many connections (~0.26 MiB/s at 16, with diminishing returns). So we download with a
front-first parallel-segment scheme: N curl workers each pull a byte range into the correct
offset of one preallocated file, working from the front so the start of the file completes
first. That means once the leading ~2 GB is on disk you can already parse a prefix
(parse_games.py --max-bytes) while the same download keeps filling toward the full month.

Resume is automatic: a sidecar `.progress` bitmap records completed 16 MiB chunks, so
re-running continues where it left off. The whole month is ~27-30 GB compressed; we never
expand it (parse_games stream-decompresses the .zst directly).

Run:
    python src/download_data.py --config config.yaml                  # full month, 16 conns
    python src/download_data.py --config config.yaml --check          # resolve URL + size only
    python src/download_data.py --config config.yaml --segments 16 --max-bytes 2147483648
"""
import argparse
import os
import shutil
import subprocess
import threading
import time
import urllib.request

import yaml

CHUNK = 4 * 1024 * 1024  # bytes per segment request (override with --chunk-mib)


def lichess_url(month):
    return f"https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"


def human(n):
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024


def remote_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers.get("Content-Length", 0))


def _bitmap_path(dest):
    return dest + ".progress"


def _load_bitmap(path, nchunks):
    bm = bytearray(nchunks)
    if os.path.exists(path):
        prev = open(path, "rb").read()
        bm[:min(len(prev), nchunks)] = prev[:nchunks]
    return bm


def _contiguous_front_bytes(bm, total):
    """Bytes of the file that are complete contiguously from offset 0 (the parseable prefix)."""
    n = 0
    for done in bm:
        if not done:
            break
        n += 1
    return min(n * CHUNK, total)


def _fetch_chunk(url, start, end, retries=6):
    size = end - start + 1
    for attempt in range(retries):
        p = subprocess.run(
            ["curl", "-s", "--fail", "--max-time", "600", "-r", f"{start}-{end}", url],
            capture_output=True,
        )
        if p.returncode == 0 and len(p.stdout) == size:
            return p.stdout
        time.sleep(min(2 ** attempt, 30))
    return None


def _download_segmented(url, dest, total, segments, max_bytes):
    import concurrent.futures

    nchunks = (total + CHUNK - 1) // CHUNK
    targeted = nchunks if not max_bytes else min(nchunks, (max_bytes + CHUNK - 1) // CHUNK)
    bm_path = _bitmap_path(dest)
    bm = _load_bitmap(bm_path, nchunks)

    fd = os.open(dest, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if os.fstat(fd).st_size != total:
            os.ftruncate(fd, total)  # sparse preallocation; offset writes land correctly
    except OSError:
        pass

    todo = [i for i in range(targeted) if not bm[i]]
    print(f"segmented download: {targeted}/{nchunks} chunks x {human(CHUNK)} targeted, "
          f"{targeted - len(todo)} already done, {len(todo)} to fetch over {segments} conns",
          flush=True)
    if not todo:
        os.close(fd)
        return True

    lock = threading.Lock()
    stop = threading.Event()
    t0 = time.time()
    fetched_bytes = [0]

    def flusher():
        while not stop.wait(5):
            with lock:
                open(bm_path, "wb").write(bytes(bm))
                front = _contiguous_front_bytes(bm, total)
                done = sum(bm[:targeted])
            rate = fetched_bytes[0] / max(time.time() - t0, 1e-6)
            left = (targeted - done) * CHUNK
            eta = f"{left / rate / 60:.0f}min" if rate > 1024 else "?"
            print(f"  {done}/{targeted} chunks  front-ready={human(front)}  "
                  f"{human(rate)}/s  eta~{eta}", flush=True)

    def work(i):
        if stop.is_set():
            return
        data = _fetch_chunk(url, i * CHUNK, min((i + 1) * CHUNK, total) - 1)
        if data is None:
            return
        os.pwrite(fd, data, i * CHUNK)
        with lock:
            bm[i] = 1
            fetched_bytes[0] += len(data)

    ft = threading.Thread(target=flusher, daemon=True)
    ft.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=segments) as ex:
            list(ex.map(work, todo))  # submission order = front-first
    finally:
        stop.set()
        with lock:
            open(bm_path, "wb").write(bytes(bm))
        os.close(fd)

    remaining = [i for i in range(targeted) if not bm[i]]
    if remaining:
        print(f"  {len(remaining)} chunks still missing (network errors) -- re-run to resume",
              flush=True)
        return False
    if targeted == nchunks:
        os.remove(bm_path)  # complete file; drop the sidecar
    return True


def _download_urllib(url, dest, total):
    """Single-stream stdlib fallback (slow under lichess throttling; resumable via Range)."""
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if total and have >= total:
        return True
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as r:
        mode = "ab"
        if getattr(r, "status", r.getcode()) != 206:
            have, mode = 0, "wb"
        done = have
        with open(dest, mode) as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {human(done)}/{human(total)} ({100*done/total:.1f}%)",
                          end="", flush=True)
        print()
    return True


def download_hf(repo, month, shards, out_dir):
    """Pull the first `shards` parquet shards of a month from the official Lichess HF dataset.
    Fast (HF is not throttled like database.lichess.org) and keeps inline [%clk]/[%eval]."""
    from huggingface_hub import HfFileSystem, hf_hub_download

    year, mon = month.split("-")
    fs = HfFileSystem()
    base = f"datasets/{repo}/data/year={year}/month={mon}"
    try:
        files = sorted(e["name"] for e in fs.ls(base, detail=True)
                       if e["name"].endswith(".parquet"))
    except FileNotFoundError:
        raise SystemExit(f"{month} not found on HF ({base}); the HF mirror lags the raw site")
    if not files:
        raise SystemExit(f"no parquet shards under {base}")
    picked = files[:shards]
    os.makedirs(out_dir, exist_ok=True)
    print(f"HF {repo} {month}: {len(files)} shards available, fetching {len(picked)}")
    local = []
    for f in picked:
        rel = f.split(f"datasets/{repo}/", 1)[1]   # 'data/year=YYYY/month=MM/train-....parquet'
        path = hf_hub_download(repo_id=repo, filename=rel, repo_type="dataset", local_dir=out_dir)
        local.append(path)
        print(f"  got {os.path.basename(path)} ({os.path.getsize(path)/1e6:.0f} MB)")
    return out_dir


def main(cfg, args):
    if args.source == "hf":
        month = args.month or cfg["data"]["lichess_month"]
        repo = cfg["data"].get("hf_repo", "Lichess/standard-chess-games")
        shards = args.shards or cfg["data"].get("hf_shards", 2)
        out_dir = os.path.join(cfg["data"]["raw_dir"], "hf", month)
        download_hf(repo, month, shards, out_dir)
        print("\nnext:")
        print(f"  python src/parse_games.py --config config.yaml --parquet {out_dir}")
        return

    month = args.month or cfg["data"]["lichess_month"]
    raw_dir = cfg["data"]["raw_dir"]
    url = lichess_url(month)
    dest = args.out or os.path.join(raw_dir, f"lichess_db_standard_rated_{month}.pgn.zst")

    try:
        total = remote_size(url)
    except Exception as exc:
        raise SystemExit(f"could not resolve {url}\n  {exc}")

    print(f"month   : {month}")
    print(f"url     : {url}")
    print(f"size    : {human(total)} ({total} bytes)")
    print(f"dest    : {dest}")
    if args.check:
        return

    os.makedirs(raw_dir, exist_ok=True)
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if total and have == total and not os.path.exists(_bitmap_path(dest)) and not args.force:
        print("already complete; skipping download.")
    else:
        if args.force:
            for p in (dest, _bitmap_path(dest)):
                if os.path.exists(p):
                    os.remove(p)
        if args.segments > 1 and shutil.which("curl"):
            ok = _download_segmented(url, dest, total, args.segments, args.max_bytes)
        else:
            ok = _download_urllib(url, dest, total)
        if not ok:
            raise SystemExit("download incomplete -- re-run this command to resume")

    print("\nnext:")
    print(f"  python src/parse_games.py --config config.yaml --pgn {dest}"
          + ("  --max-bytes <prefix-bytes>  (to parse the front before the full file finishes)"
             if args.max_bytes or os.path.exists(_bitmap_path(dest)) else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", choices=["hf", "http"], default="hf",
                    help="hf = official Lichess parquet on HuggingFace (fast); http = database.lichess.org .zst (throttled)")
    ap.add_argument("--shards", type=int, default=0, help="(hf) number of month shards to fetch (0 = cfg.data.hf_shards)")
    ap.add_argument("--month", default=None, help="override cfg.data.lichess_month (YYYY-MM)")
    ap.add_argument("--out", default=None, help="(http) override destination path")
    ap.add_argument("--check", action="store_true", help="(http) resolve URL + size, then exit")
    ap.add_argument("--force", action="store_true", help="discard any existing partial file")
    ap.add_argument("--segments", type=int, default=16, help="parallel connections (1 = single stream)")
    ap.add_argument("--max-bytes", type=int, default=0, help="only fetch the first N bytes (front prefix)")
    ap.add_argument("--chunk-mib", type=int, default=4, help="segment size in MiB (smaller = finer progress/prefix)")
    a = ap.parse_args()
    CHUNK = a.chunk_mib * 1024 * 1024
    main(yaml.safe_load(open(a.config)), a)
