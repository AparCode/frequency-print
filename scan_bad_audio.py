"""Utility script to scan audio files for decode problems before training."""

import argparse
from collections import Counter
import csv
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import preprocess as pp

try:
    import shutil
    import subprocess
except Exception:
    shutil = None
    subprocess = None


def load_with_ffmpeg(audio_path, target_sr):
    """Decode one audio file to mono float samples using ffmpeg."""
    if shutil is None or subprocess is None:
        raise RuntimeError("ffmpeg fallback is unavailable")

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg not found in PATH")

    cmd = [
        ffmpeg_bin,
        "-v",
        "error",
        "-nostdin",
        "-i",
        audio_path,
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0 or not proc.stdout:
        stderr_msg = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(stderr_msg or "ffmpeg failed to decode audio")

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("Decoded empty audio stream")
    return audio


def check_file(audio_path: str, target_sr: int = 32000) -> tuple[str, str]:
    """Check decode health for one file with librosa and ffmpeg fallback."""
    try:
        librosa.load(audio_path, sr=target_sr, mono=True)
        return "ok", "librosa"
    except Exception as first_err:
        try:
            load_with_ffmpeg(audio_path, target_sr)
            return "fallback", f"librosa failed: {type(first_err).__name__}; ffmpeg ok"
        except Exception as second_err:
            return "bad", f"librosa failed: {type(first_err).__name__}; ffmpeg failed: {type(second_err).__name__}: {second_err}"


def build_manifest(args: argparse.Namespace):
    """Build a manifest for scan targets based on CLI options."""
    if args.paths and args.labels:
        if len(args.paths) != len(args.labels):
            raise ValueError("--paths and --labels must have the same length")
        return pp.build_master_list(args.paths, args.labels, filetype=args.filetypes)

    return pp.build_master_list(
        paths=["data/fake", "data/real"],
        labels=[0, 1],
        filetype=args.filetypes,
    )


def main() -> None:
    """CLI entry point for decode-health scanning."""
    parser = argparse.ArgumentParser(description="Scan audio files for decode problems before training.")
    parser.add_argument("--paths", nargs="*", default=None, help="Audio root directories")
    parser.add_argument("--labels", nargs="*", type=int, default=None, help="Labels matching --paths")
    parser.add_argument("--filetypes", nargs="*", default=["wav", "mp3"], help="Extensions to scan")
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many files; 0 means scan all")
    parser.add_argument("--csv", default=None, help="Optional path to write bad/fallback files as CSV")
    args = parser.parse_args()

    manifest = build_manifest(args)
    if manifest.empty:
        print("No files found.")
        return

    paths = manifest["path"].tolist()
    if args.limit and args.limit > 0:
        paths = paths[: args.limit]

    counts = Counter()
    rows = []
    for path in paths:
        status, message = check_file(path)
        counts[status] += 1
        if status != "ok":
            rows.append({"path": path, "status": status, "message": message})
            print(f"{status.upper()}: {path}\n  {message}")

    print("Summary:")
    print(f"  ok: {counts.get('ok', 0)}")
    print(f"  fallback: {counts.get('fallback', 0)}")
    print(f"  bad: {counts.get('bad', 0)}")

    if args.csv and rows:
        out_path = Path(args.csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "status", "message"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()
