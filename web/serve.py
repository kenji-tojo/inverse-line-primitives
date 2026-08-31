#!/usr/bin/env python3
"""web/serve.py - serve the web viewer over HTTP and print the URL to open.

Also prints a LAN URL so a phone on the same WiFi can connect.

Responses carry "Cache-Control: no-store" so edits to main.js / index.html
are picked up on reload.  Bundles are keyed by filename, so re-fetching them
is only a cost while iterating.

Usage
-----
    python web/serve.py                       # scene picker visible
    python web/serve.py --port 8080
    python web/serve.py --no-ui --scene fuzzy_kiwi_sh1.bin

``--no-ui`` hides every overlay, including the scene picker.  It is meant for
screen capture, and because the picker is then unavailable the bundle has to
be named with ``--scene``.
"""

from __future__ import annotations

import argparse
import http.server
import json
import socket
import socketserver
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("  %s\n" % (fmt % args))


def lan_ip() -> str | None:
    """Best-effort LAN address, found by opening a UDP socket that never sends."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("192.0.2.1", 1))   # TEST-NET-1, never routed
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return None


def available_scenes() -> list[str]:
    manifest = DATA / "manifest.json"
    if not manifest.exists():
        return []
    try:
        return [s["file"] for s in json.loads(manifest.read_text()).get("scenes", [])]
    except (ValueError, KeyError):
        return []


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-ui", action="store_true",
                    help="hide every overlay including the scene picker; "
                         "requires --scene")
    ap.add_argument("--scene", type=str, default=None,
                    help="bundle in data/ to load instead of the manifest's "
                         "first entry")
    args = ap.parse_args()

    if not (DATA / "manifest.json").exists():
        print(f"error: {DATA}/manifest.json missing.", file=sys.stderr)
        print("       Run:  python web/precompute.py", file=sys.stderr)
        return 1

    scenes = available_scenes()
    if args.scene and args.scene not in scenes:
        print(f"error: scene {args.scene!r} is not in the manifest.", file=sys.stderr)
        print(f"       available: {', '.join(scenes) or '(none)'}", file=sys.stderr)
        return 2
    if args.no_ui and not args.scene:
        print("error: --no-ui hides the scene picker, so --scene is required.",
              file=sys.stderr)
        print(f"       available: {', '.join(scenes) or '(none)'}", file=sys.stderr)
        return 2

    query = []
    if args.no_ui:
        query.append("ui=0")
    if args.scene:
        query.append(f"scene={args.scene}")
    suffix = ("?" + "&".join(query)) if query else "/"
    if query:
        suffix = "/" + suffix

    ip = lan_ip()
    print()
    print(f"Serving {HERE}  on  http://0.0.0.0:{args.port}")
    print(f"Laptop:              http://localhost:{args.port}{suffix}")
    if ip:
        print(f"Phone (same WiFi):   http://{ip}:{args.port}{suffix}")
    print()
    print("Ctrl-C to stop.")
    print()

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", args.port), NoCacheHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
