"""
Lossless "faststart" MP4 optimizer + container report.

Why this exists
---------------
An MP4 keeps its sample data in `mdat` and the index that describes every
sample (the tables a decoder needs up front) in `moov`.  If `moov` is written
*after* `mdat` - what most editors export by default - a browser cannot decode
a single frame until the *entire* file has been downloaded.  That is exactly
what makes a video intro look like it is buffering: an empty rectangle, a
wordless wait, then a jump into the clip.

This tool rewrites the container so the box order becomes

    ftyp, [any other header boxes], moov, mdat

and patches the chunk offset tables (`stco` 32-bit / `co64` 64-bit) by the
number of bytes the `mdat` payload moved.  Sample data is copied byte for
byte, so the result is visually identical to the source (no re-encode).

Usage
-----
    python tools/mp4_faststart.py Neutron_Stars.mp4              # report only
    python tools/mp4_faststart.py in.mp4 out.mp4                 # write optimized copy

Exit codes: 0 = ok, 1 = already optimized / nothing to do, 2 = error.
"""

from __future__ import annotations

import hashlib
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

# Boxes whose payload is a list of child boxes.
CONTAINERS = {
    "moov", "trak", "mdia", "minf", "stbl", "edts", "dinf", "udta", "mvex", "moof", "traf",
}
# Offset tables that point into mdat.
OFFSET_TABLES = ("stco", "co64")


@dataclass
class Box:
    offset: int
    type: str
    size: int
    header: int

    @property
    def body(self) -> int:
        return self.offset + self.header

    @property
    def end(self) -> int:
        return self.offset + self.size


def iter_boxes(data: bytes, start: int, end: int):
    """Yield the boxes inside data[start:end]."""
    off = start
    while off + 8 <= end:
        size = struct.unpack_from(">I", data, off)[0]
        box_type = data[off + 4:off + 8].decode("latin1", "replace")
        header = 8
        if size == 1:
            if off + 16 > end:
                break
            size = struct.unpack_from(">Q", data, off + 8)[0]
            header = 16
        elif size == 0:
            size = end - off
        if size < header or off + size > end:
            break
        yield Box(off, box_type, size, header)
        off += size


def walk(data: bytes, start: int, end: int):
    """Recursively yield (box, path) for every box, container or leaf."""
    for box in iter_boxes(data, start, end):
        yield box, ()
        if box.type in CONTAINERS:
            for child, path in walk(data, box.body, box.end):
                yield child, (box.type,) + path


def find_top_level(data: bytes):
    return list(iter_boxes(data, 0, len(data)))


def chunk_offsets(data: bytes, moov: Box):
    """Return [(table_start, entry_count, entry_size, path)] for every stco/co64."""
    tables = []
    for box, path in walk(data, moov.body, moov.end):
        if box.type in OFFSET_TABLES:
            entry_size = 4 if box.type == "stco" else 8
            count = struct.unpack_from(">I", data, box.body + 4)[0]
            tables.append((box.body + 8, count, entry_size, path + (box.type,)))
    return tables


def read_offsets(data: bytes, tables) -> list[int]:
    """Flatten every stco/co64 entry into one list of absolute file offsets."""
    values: list[int] = []
    for table_start, count, entry_size, _ in tables:
        fmt = ">I" if entry_size == 4 else ">Q"
        for i in range(count):
            values.append(struct.unpack_from(fmt, data, table_start + i * entry_size)[0])
    return values


def report(data: bytes, label: str) -> dict:
    boxes = find_top_level(data)
    print(f"--- {label} ({len(data):,} bytes) ---")
    for box in boxes:
        print(f"  @{box.offset:>10,}  {box.type:<6} size={box.size:,}")

    moov = next((b for b in boxes if b.type == "moov"), None)
    mdat = next((b for b in boxes if b.type == "mdat"), None)
    if moov is None or mdat is None:
        raise SystemExit("error: not a plain MP4 (missing moov and/or mdat)")

    faststart = moov.offset < mdat.offset
    print(f"  moov before mdat (streamable / faststart): {faststart}")

    tables = chunk_offsets(data, moov)
    offsets = read_offsets(data, tables)
    inside = sum(1 for value in offsets if mdat.body <= value < mdat.end)
    print(f"  chunk tables : {', '.join('/'.join(t[3]) for t in tables) or 'none'} "
          f"-> {len(offsets)} entries, {inside} inside mdat")
    if offsets:
        print(f"  chunk range  : {min(offsets):,} .. {max(offsets):,} "
              f"(mdat payload starts at {mdat.body:,})")

    # Encrypted / auxiliary information offsets also address mdat; bail out rather
    # than write a file whose offsets we did not patch.
    risky = sorted({b.type for b, _ in walk(data, moov.body, moov.end)
                    if b.type in ("saio", "senc", "sbgp", "sgpd")})
    if risky:
        raise SystemExit(f"error: moov contains {risky}; offsets would also need patching "
                         "- use `ffmpeg -c copy -movflags +faststart` instead")
    return {"boxes": boxes, "moov": moov, "mdat": mdat, "faststart": faststart,
            "tables": tables, "offsets": offsets}


def remux(data: bytes, info: dict) -> tuple[bytes, int]:
    """Move moov in front of mdat. Returns (new_file_bytes, bytes_shifted)."""
    moov: Box = info["moov"]
    mdat: Box = info["mdat"]
    if info["faststart"]:
        raise SystemExit(1)

    delta = moov.size  # every mdat byte moves forward by exactly the moov size

    moov_bytes = bytearray(data[moov.offset:moov.end])
    for table_start, count, entry_size, _ in info["tables"]:
        fmt = ">I" if entry_size == 4 else ">Q"
        limit = (1 << 32) - 1 if entry_size == 4 else (1 << 64) - 1
        for i in range(count):
            pos = table_start - moov.offset + i * entry_size
            value = struct.unpack_from(fmt, moov_bytes, pos)[0]
            if value + delta > limit:
                raise SystemExit(f"error: chunk offset {value} overflows {fmt}")
            struct.pack_into(fmt, moov_bytes, pos, value + delta)

    before = [b for b in info["boxes"] if b.offset < mdat.offset and b.type != "moov"]
    after = [b for b in info["boxes"] if b.offset > mdat.offset and b.type != "moov"]

    out = bytearray()
    for box in before:
        out += data[box.offset:box.end]
    out += moov_bytes
    out += data[mdat.offset:mdat.end]
    for box in after:
        out += data[box.offset:box.end]

    return bytes(out), delta


def verify(original: bytes, optimized: bytes, info: dict) -> bool:
    """Structural checks: box order, untouched sample data, patched offsets."""
    print(f"--- verifying rewritten file ({len(optimized):,} bytes) ---")
    dst = find_top_level(optimized)
    order = [b.type for b in dst]
    print(f"  new box order: {order}")

    ok = order.index("moov") < order.index("mdat")
    if not ok:
        print("  FAIL: moov is still behind mdat")

    src_mdat = info["mdat"]
    dst_mdat = next(b for b in dst if b.type == "mdat")
    src_hash = hashlib.sha256(original[src_mdat.body:src_mdat.end]).hexdigest()
    dst_hash = hashlib.sha256(optimized[dst_mdat.body:dst_mdat.end]).hexdigest()
    same = src_hash == dst_hash
    print(f"  sample data byte-identical: {same}")
    ok &= same

    new_moov = next(b for b in dst if b.type == "moov")
    new_offsets = read_offsets(optimized, chunk_offsets(optimized, new_moov))
    delta = new_moov.size
    shifted = [b - a for a, b in zip(info["offsets"], new_offsets)]
    offsets_ok = bool(shifted) and set(shifted) == {delta}
    print(f"  every chunk offset shifted by {delta:,} bytes: {offsets_ok}")
    ok &= offsets_ok

    inside = sum(1 for value in new_offsets if dst_mdat.body <= value < dst_mdat.end)
    print(f"  chunk offsets inside new mdat: {inside}/{len(new_offsets)}")
    ok &= inside == len(new_offsets)

    size_ok = len(original) == len(optimized)
    print(f"  file size unchanged ({len(original):,} bytes): {size_ok}")
    ok &= size_ok
    return ok


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    src_path = Path(argv[0])
    if not src_path.is_file():
        print(f"error: {src_path} not found")
        return 2

    data = src_path.read_bytes()
    info = report(data, src_path.name)

    if len(argv) < 2:
        print("\nReport only (no output path given). "
              "Pass an output path to write the optimized file.")
        return 1 if info["faststart"] else 0

    out_path = Path(argv[1])
    if info["faststart"]:
        print("\nAlready faststart - nothing to do.")
        return 1

    optimized, delta = remux(data, info)
    print(f"\nMoving moov ahead of mdat (chunk offsets shift by {delta:,} bytes)")
    print(f"Writing {out_path} ...")
    out_path.write_bytes(optimized)
    ok = verify(data, optimized, info)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
