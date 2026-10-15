"""Poll the native probe's live-status every interval and write each readout.

This is a defensive polling snapshot used to observe an online battle across
its load/teardown without crashing the game.  Readouts are appended as compact
JSON lines.  The `--watch-idle` flag only prints the compact tick/readable line
so the same command can watch the lobby without flooding the shell.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import time


def send(command: str, host: str, port: int, timeout: float) -> str:
    try:
        with socket.create_connection((host, port), timeout=timeout) as connection:
            connection.sendall(command.encode("utf-8") + b"\n")
            connection.settimeout(timeout)
            chunks: list[bytes] = []
            while True:
                block = connection.recv(65536)
                if not block:
                    break
                chunks.append(block)
                if block.endswith(b"\n"):
                    break
        return b"".join(chunks).decode("utf-8", errors="replace").strip()
    except OSError as error:
        return json.dumps({"ok": False, "error": str(error)})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=26789)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--watch-idle", action="store_true", help="compact one-line tick/readable output")
    parser.add_argument("--count", type=int, default=0, help="number of polls (0 = run until keyboard interrupt)")
    args = parser.parse_args(argv)

    try:
        count = 0
        while args.count == 0 or count < args.count:
            count += 1
            reply = send("live-status", args.host, args.port, args.timeout)
            if args.watch_idle:
                tick = re.search(r'"tick":(-?\d+)', reply)
                readable = re.search(r'"resultReadable":(true|false)', reply)
                world = re.search(r'"world":"(0x[0-9a-f]+)"', reply)
                print("tick={} readable={} world={}".format(
                    tick.group(1) if tick else "?",
                    readable.group(1) if readable else "?",
                    world.group(1) if world else "",
                ), flush=True)
            else:
                print(reply, flush=True)
            time.sleep(max(0.0, args.interval))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())