"""Send one raw line to the native probe control port and print the reply.

Used for reconnaissance against a running client: the probe speaks one request
per line (``status``, ``attest``, ``observe``, ``observe-rich``, ...).  See
``handle_control_command`` in ``native_runner/probe/cr_replay_probe.cpp`` for
the full command set.
"""

from __future__ import annotations

import argparse
import socket
import sys


def send(command: str, host: str, port: int, timeout: float) -> str:
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="+", help="raw probe command line")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=26789)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--repeat", type=int, default=1, help="repeat the command N times")
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between repeats")
    args = parser.parse_args(argv)

    command = " ".join(args.command)
    for index in range(max(1, args.repeat)):
        if index:
            import time

            time.sleep(max(0.0, args.interval))
        try:
            reply = send(command, args.host, args.port, args.timeout)
        except OSError as error:
            print(f"connection error: {error}", file=sys.stderr)
            return 2
        print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
