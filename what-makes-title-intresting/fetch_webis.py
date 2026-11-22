"""Fetch only the JSONL files (posts + labels) from the Webis-Clickbait-17 zips on Zenodo.

The zips are ~1 GB each because they bundle post images; HTTP range reads let zipfile pull just
instances.jsonl / truth.jsonl (a few MB). Source: https://zenodo.org/record/5530410 (CC BY 4.0).
Usage: python fetch_webis.py <out_dir>
"""
import io
import sys
import zipfile
import urllib.request
from pathlib import Path

BASE = "https://zenodo.org/api/records/5530410/files/{}/content"
ZIPS = ["clickbait17-train-170331.zip", "clickbait17-train-170630.zip", "clickbait17-test-170720.zip"]


class HTTPRangeFile(io.RawIOBase):
    """A read-only, seekable 'file' whose bytes live on a web server.

    zipfile needs a seekable file: it first reads the zip's table of contents at the END of the archive,
    then jumps to the members it wants. Each read() here becomes an HTTP Range request
    ("give me bytes 1000-1999"), so only the few MB we need are downloaded, not the ~1 GB zip.
    """
    def __init__(self, url):
        """Ask the server for byte 0 only; the Content-Range reply header tells us the total file size."""
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Range"].split("/")[-1])

    # tell io/zipfile that random access (seek) is supported
    def seekable(self): return True
    # tell io/zipfile that reading is supported
    def readable(self): return True
    # current read position in bytes
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        """Move the read position. whence 0 = from start, 1 = from current position, 2 = from end of file."""
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        """Download n bytes starting at the current position with one HTTP Range request (n<0 = until the end)."""
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        """Fill the given buffer b with downloaded bytes (used by io.BufferedReader); returns the number of bytes."""
        data = self.read(len(b)); b[:len(data)] = data
        return len(data)


def main(out_dir):
    """For each of the 3 Webis zips: open it remotely, list its members, and save only the .jsonl
    files (instances = posts, truth = clickbait labels) under out_dir/<zip name>/.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for name in ZIPS:
        f = HTTPRangeFile(BASE.format(name))
        z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20))
        members = [m for m in z.namelist() if m.endswith(".jsonl")]
        print(name, f"{f.size / 1e6:.0f} MB zip ->", members)
        for m in members:
            target = out / name.replace(".zip", "") / Path(m).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(m))
            print("  wrote", target, f"{target.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
