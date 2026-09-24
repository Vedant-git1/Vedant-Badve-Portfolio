"""
Decode-level checks for the intro clip, plus poster-frame extraction.

What it answers
---------------
1. Does the clip actually decode all the way through?  (frame count / fps / duration)
2. Are the first frames black?  A clip that starts on a black frame looks like a
   buffering stall even when the stream is perfectly healthy, so if that is the
   case we want the intro to cover it with a poster still instead.
3. Which early frame is the best candidate for `intro-poster.jpg`?
4. After a container rewrite (`mp4_faststart.py`), do the decoded pixels still
   match the source frame for frame?

Usage
-----
    python tools/video_check.py Neutron_Stars.mp4
    python tools/video_check.py Neutron_Stars.mp4 --compare Neutron_Stars.faststart.mp4
    python tools/video_check.py Neutron_Stars.mp4 --poster intro-poster.jpg
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import cv2

POSTER_MAX_WIDTH = 1600
POSTER_QUALITY = 84


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"error: could not open {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frames / fps if fps else 0.0

    brightness = []
    hashes = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hashes.append(hashlib.md5(frame.tobytes()).hexdigest())
        if index < int(fps * 4) if fps else index < 120:  # first ~4s only
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness.append((index, float(gray.mean())))
        index += 1
    cap.release()

    duration = index / fps if fps else duration
    decoded = index
    print(f"--- {path.name} ---")
    print(f"  resolution : {width}x{height} ({width * height / 1_000_000:.2f} MP)")
    print(f"  fps        : {fps:.3f}")
    print(f"  duration   : {duration:.2f} s (container reports {frames} frames)")
    print(f"  decoded    : {decoded} frames")

    if decoded != frames:
        print(f"  WARNING: decoded {decoded} of {frames} reported frames")

    if brightness:
        first_second = [b for b in brightness if fps and b[0] <= fps]
        mean_all = sum(b for _, b in first_second) / max(1, len(first_second))
        peak_index, peak_value = max(brightness, key=lambda item: item[1])
        print(f"  mean luma of the first second : {mean_all:.1f}/255")
        print(f"  brightest early frame         : #{peak_index} ({peak_index / fps:.2f}s) "
              f"luma {peak_value:.1f}")
        if mean_all < 25:
            print("  NOTE: the clip opens on (near) black - a poster still should cover it")

    return {"fps": fps, "frames": decoded, "duration": duration,
            "width": width, "height": height, "hashes": hashes,
            "brightness": brightness}


def compare(a: Path, b: Path) -> bool:
    print(f"--- comparing {a.name} vs {b.name} ---")
    left = probe(a)
    right = probe(b)
    if left["hashes"] == right["hashes"]:
        print("  decoded frames are pixel-identical  ->  PASS")
        return True
    mismatch = next((i for i, (x, y) in enumerate(zip(left["hashes"], right["hashes"])) if x != y), None)
    print(f"  first differing frame: {mismatch} of {len(left['hashes'])}  ->  FAIL")
    return False


def write_poster(path: Path, target: Path, seconds: float | None = None) -> None:
    """Write a poster frame.

    With no explicit timestamp we grab the brightest frame of the first seconds.
    The clip opens on deep space (mean luma 27/255), so frame 0 is a poor poster
    and the intro would flip between two near-black images.
    """
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if seconds is None:
        index, seconds = best_frame_index(cap, fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(seconds * fps))))
    ok, frame = cap.read()
    if not ok:
        # fall back to the very first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"error: could not read a frame from {path}")

    height, width = frame.shape[:2]
    if width > POSTER_MAX_WIDTH:
        scale = POSTER_MAX_WIDTH / width
        frame = cv2.resize(frame, (POSTER_MAX_WIDTH, int(round(height * scale))),
                           interpolation=cv2.INTER_AREA)

    if target.suffix.lower() in (".jpg", ".jpeg"):
        cv2.imwrite(str(target), frame, [int(cv2.IMWRITE_JPEG_QUALITY), POSTER_QUALITY])
    else:
        cv2.imwrite(str(target), frame)
    print(f"  poster written: {target} ({target.stat().st_size / 1024:.0f} KB, "
          f"{frame.shape[1]}x{frame.shape[0]}, source frame @{seconds:.2f}s)")


def best_frame_index(cap, fps: float, window_seconds: float = 4.0):
    """Brightest frame within the first `window_seconds` -> (index, seconds)."""
    limit = max(1, int(round(fps * window_seconds))) if fps else 120
    best_index, best_luma = 0, -1.0
    for index in range(limit):
        ok, frame = cap.read()
        if not ok:
            break
        luma = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        if luma > best_luma:
            best_index, best_luma = index, luma
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return best_index, (best_index / fps if fps else 0.0)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--compare", type=Path, help="second file to compare decoded frames with")
    parser.add_argument("--poster", type=Path, help="write a poster frame to this path")
    parser.add_argument("--poster-time", type=float, default=None,
                        help="seconds into the clip to grab the poster frame from "
                             "(default: brightest frame of the first 4 s)")
    args = parser.parse_args(argv)

    info = probe(args.video)
    ok = True
    if args.compare:
        ok &= compare(args.video, args.compare)
    if args.poster:
        write_poster(args.video, args.poster, args.poster_time)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
