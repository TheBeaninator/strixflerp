#!/usr/bin/env python3
"""Write `loom-swarm stream` frames (u64 file offset, u64 length, bytes; LE; end = offset 2**64-1, length 0) from
stdin into a sparse file of the given size at their offsets. The fallback of gate/ornith_p5.py (--source file): the
file goes on the RAM disk and the loader reads it with --data-file --punch-behind.

    loom-swarm stream ... | python3 swarm_frames_to_file.py OUT SIZE
Exit 0 only after the end frame.
"""
import os
import struct
import sys


def main():
    out, size = sys.argv[1], int(sys.argv[2])
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd, size)
    src = sys.stdin.buffer
    total = 0
    while True:
        head = src.read(16)
        if len(head) != 16:
            sys.exit("stream ended without its end frame")
        off, n = struct.unpack("<QQ", head)
        if off == 2 ** 64 - 1 and n == 0:
            break
        if off + n > size:
            sys.exit(f"frame [{off}, +{n}) past the end ({size})")
        data = src.read(n)
        if len(data) != n:
            sys.exit("stream broke inside a frame")
        os.pwrite(fd, data, off)
        total += n
    os.close(fd)
    print(f'{{"event":"frames_written","bytes":{total},"path":"{out}"}}')


if __name__ == "__main__":
    main()
