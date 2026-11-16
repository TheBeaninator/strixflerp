#!/usr/bin/env python3
"""Scan a tree for what the public export must not carry: private network addresses, home directories, extra words.

The public repository is a scrubbed export of the internal one (docs/PROVENANCE.md). The export runs this on the
public tree after scrubbing and publishes only when it exits 0; a reviewer can run it on any tree. It reports:

- IPv4 addresses in the private ranges 10/8, 172.16/12 and 192.168/16 (the export maps hosts to the RFC 5737
  documentation ranges 192.0.2.0/24, 198.51.100.0/24 and 203.0.113.0/24, which are not reported);
- home directories other than the placeholders /home/user and /home/pod;
- with --deny FILE, every line of FILE as a case-insensitive word (hostnames, user names, keys). The deny list
  stays outside the repository: it names exactly what must not appear.

Vendored source (catalog/<origin other than flerp>/*/*/src/) and .git are skipped, as the export leaves them as
they were published upstream. On the internal tree it reports many hits: that is the point of the export.

    python3 tools/public-check.py [ROOT] [--deny FILE] [--list]
Exit 0 when nothing is found, 1 otherwise.
"""

import argparse
import os
import re
import sys

VENDORED = re.compile(r"^catalog/(?!flerp/)[^/]+/[^/]+/[^/]+/src/")
PRIVATE_V4 = re.compile(r"(?<![\d.])(10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}(?![\d.]*\d)")
# any case, Linux and macOS layouts: a capitalised name under /home and any name under /Users count too
HOME = re.compile(r"/(?:home|users)/(?!user\b|pod\b)[a-z_][a-z0-9_-]*", re.I)


def scan(root, deny):
    words = [re.compile(r"(?<![A-Za-z0-9])" + re.escape(w) + r"(?![A-Za-z0-9])", re.I) for w in deny]
    hits = {"private address": {}, "home directory": {}, "deny-list word": {}}
    unread = []
    for d, dirs, files in os.walk(root):
        if ".git" in dirs:
            dirs.remove(".git")
        for f in files:
            p = os.path.join(d, f)
            rel = os.path.relpath(p, root)
            if VENDORED.match(rel) or os.path.islink(p):
                continue
            try:
                with open(p, encoding="utf-8") as fh:
                    text = fh.read()
            except (UnicodeDecodeError, OSError):
                unread.append(rel)  # binary or unreadable: not scanned, so counted and listed
                continue
            for kind, found in (("private address", PRIVATE_V4.search(text)), ("home directory", HOME.search(text)),
                                ("deny-list word", next((m for w in words if (m := w.search(text))), None))):
                if found:
                    hits[kind][rel] = found.group(0) if kind != "deny-list word" else "(deny-list match)"
    return hits, unread


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--deny", help="file of words, one per line, that must not appear")
    ap.add_argument("--list", action="store_true", help="print every file with a hit")
    a = ap.parse_args()
    deny = []
    if a.deny:
        with open(a.deny) as f:
            deny = [w.strip() for w in f if w.strip() and not w.startswith("#")]
    hits, unread = scan(a.root, deny)
    print(f"not scanned (binary or not UTF-8, outside vendored source): {len(unread)} file(s)")
    if a.list:
        for rel in sorted(unread):
            print(f"  {rel}")
    total = 0
    for kind, files in hits.items():
        total += len(files)
        print(f"{kind}: {len(files)} file(s)")
        if a.list:
            for rel, sample in sorted(files.items()):
                print(f"  {rel}: {sample}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
