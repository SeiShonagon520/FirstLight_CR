"""Live (server-driven) battle harness for FirstLight CR.

A battle the client starts from its own network path is owned by the game, not
by the probe, so this harness drives the probe's live commands instead of the
offline match lifecycle:

* ``live-status``          - locate the live manager/world/players
* ``live-observe-atomic``  - the ordinary + rich observation pair a V4 policy needs
* ``live-deploy OWNER HAND X Y`` - submit one card play into the live battle

Modes
-----
``capture``  Poll a running battle and record observations (default).
``probe``    Resolve one hand deploy without committing it, to check legality.
``deploy``   Queue one deploy at OWNER HAND X Y (one shot per invocation).

The service never owns the battle and never pauses the game; it only reads and
optionally injects through the game thread's own command path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
import time
from typing import Any, Mapping


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 26789


class ProbeConnection:
    """One request per line against the native probe control port."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 20.0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)

    def send(self, command: str) -> dict[str, Any]:
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as connection:
            connection.sendall(command.encode("utf-8") + b"\n")
            connection.settimeout(self.timeout)
            chunks: list[bytes] = []
            while True:
                block = connection.recv(1 << 16)
                if not block:
                    break
                chunks.append(block)
                # Control replies are one JSON object terminated by a newline.
                if block.endswith(b"\n"):
                    break
        text = b"".join(chunks).decode("utf-8", errors="replace").strip()
        if not text:
            raise RuntimeError(f"empty probe reply for {command!r}")
        try:
            reply = json.loads(text)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"probe reply is not JSON for {command!r}: {text[:200]!r}") from error
        if not isinstance(reply, dict):
            raise RuntimeError(f"probe reply is not an object for {command!r}")
        return reply


def _player_summary(raw: Mapping[str, Any], owner: int) -> dict[str, Any]:
    players = raw.get("players")
    if not isinstance(players, list):
        return {"owner": owner, "readable": False}
    for entry in players:
        if isinstance(entry, Mapping) and int(entry.get("owner", -1)) == owner:
            return dict(entry)
    return {"owner": owner, "readable": False}


def battle_summary(status: Mapping[str, Any]) -> dict[str, Any]:
    live = status.get("live")
    if not isinstance(live, Mapping):
        return {"manager": None, "world": None, "tick": None, "players": []}
    world = live.get("world")
    return {
        "manager": live.get("manager"),
        "world": world,
        "tick": live.get("tick"),
        "resultReadable": live.get("resultReadable"),
        "finalized": live.get("finalized"),
        "players": [_player_summary(live, owner) for owner in (0, 1)],
    }


def describe(summary: Mapping[str, Any]) -> str:
    tick = summary.get("tick")
    world = summary.get("world")
    parts = [f"tick={tick}", f"world={world}"]
    for player in summary.get("players", ()):
        owner = player.get("owner")
        if player.get("readable"):
            hand = player.get("handCount")
            deck = player.get("deckCount")
            elixir = player.get("elixirRaw")
            elixir_x10 = None if elixir is None else round(float(elixir) / 10_000.0, 2)
            parts.append(f"p{owner}(hand={hand},deck={deck},elixir={elixir_x10})")
        else:
            parts.append(f"p{owner}(unreadable)")
    return " ".join(parts)


def wait_for_battle(probe: ProbeConnection, timeout: float, interval: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            status = probe.send("live-status")
        except OSError as error:
            print(f"[wait] probe unreachable: {error}", flush=True)
            time.sleep(interval)
            continue
        summary = battle_summary(status)
        last = summary
        readable = summary.get("world") not in (None, "0x0") and any(
            player.get("readable") for player in summary.get("players", ())
        )
        if readable:
            return summary
        time.sleep(interval)
    raise RuntimeError(f"no readable live battle appeared within {timeout:.0f}s (last={last})")


def run_capture(probe: ProbeConnection, args: argparse.Namespace) -> int:
    print("[capture] waiting for a live battle…", flush=True)
    summary = wait_for_battle(probe, args.timeout, args.interval)
    print(f"[capture] battle found: {describe(summary)}", flush=True)

    artifact = Path(args.artifact_dir)
    artifact.mkdir(parents=True, exist_ok=True)
    samples_path = artifact / "live-samples.jsonl"
    atomic_path = artifact / "live-atomic.json"
    last_tick: int | None = None
    started = time.monotonic()
    with samples_path.open("w", encoding="utf-8", buffering=1) as samples:
        while time.monotonic() - started < args.duration:
            try:
                status = probe.send("live-status")
            except OSError as error:
                print(f"[capture] probe unreachable: {error}", flush=True)
                break
            live_summary = battle_summary(status)
            tick = live_summary.get("tick")
            world = live_summary.get("world")
            if world in (None, "0x0"):
                print("[capture] battle ended (world cleared)", flush=True)
                break
            if tick != last_tick:
                last_tick = tick
                samples.write(json.dumps(live_summary, ensure_ascii=False) + "\n")
                print(f"[capture] {describe(live_summary)}", flush=True)
            if args.observe_every > 0 and last_tick is not None and last_tick % args.observe_every < 20:
                try:
                    atomic = probe.send("live-observe-atomic")
                except OSError as error:
                    print(f"[capture] atomic query failed: {error}", flush=True)
                else:
                    atomic_path.write_text(json.dumps(atomic, ensure_ascii=False), encoding="utf-8")
                    ok = atomic.get("ok")
                    print(f"[capture] live-observe-atomic ok={ok} schema={atomic.get('schema')}", flush=True)
            time.sleep(args.interval)
    print(f"[capture] samples={samples_path}", flush=True)
    return 0


def run_deploy(probe: ProbeConnection, args: argparse.Namespace) -> int:
    summary = wait_for_battle(probe, args.timeout, args.interval)
    print(f"[deploy] battle: {describe(summary)}", flush=True)
    command = f"live-deploy {args.owner} {args.hand} {args.x} {args.y}"
    print(f"[deploy] sending: {command}", flush=True)
    reply = probe.send(command)
    print(f"[deploy] reply: {json.dumps(reply, ensure_ascii=False)}", flush=True)
    return 0 if reply.get("ok") else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for a live battle")
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--artifact-dir", type=Path, default=Path("runs/live"))
    sub = parser.add_subparsers(dest="mode")
    capture = sub.add_parser("capture", help="record observations from a live battle")
    capture.add_argument("--duration", type=float, default=600.0)
    capture.add_argument("--observe-every", type=int, default=100, help="capture atomic observation every N ticks")
    deploy = sub.add_parser("deploy", help="queue one live deploy")
    deploy.add_argument("--owner", type=int, choices=(0, 1), required=True)
    deploy.add_argument("--hand", type=int, required=True)
    deploy.add_argument("--x", type=int, required=True)
    deploy.add_argument("--y", type=int, required=True)
    args = parser.parse_args(argv)
    if args.mode is None:
        args.mode = "capture"
        args.duration = 600.0
        args.observe_every = 100

    probe = ProbeConnection(args.host, args.port)
    if args.mode == "capture":
        return run_capture(probe, args)
    return run_deploy(probe, args)


if __name__ == "__main__":
    raise SystemExit(main())
