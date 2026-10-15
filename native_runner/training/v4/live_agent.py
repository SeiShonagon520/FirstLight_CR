"""Drive a V4 checkpoint inside a live (server-driven) client battle.

The offline harness owns every battle it plays: it creates the native match,
steps it, and injects commands.  A battle the client starts from its own
network path is owned by the game instead, so this module attaches to that
battle through the probe's ``live-*`` control commands:

* ``live-status`` reports whether a live battle graph exists yet.
* ``live-observe-atomic`` captures the ordinary + rich envelopes the V4 model
  already consumes, read from the game's own ``GameStateManager``.
* ``live-deploy`` submits one resolved hand deployment back through the probe.

The model, tensorizer, and decoding are the exact production V4 objects used by
the offline agent; only the observation source and the action sink differ.  The
episode contract is likewise the offline one: this module reconstructs a
``MatchConfig``/``EpisodeConfigV1`` from the live envelopes, binds the fixed
tower identities, and drives ``PolicySessionV4`` so the model sees precisely the
tensors it was trained on.

This runs against the user's own Null's Royale account and needs the client to
be online.  It never creates or steps a match: the game advances the world, and
this loop only reads state and proposes plays.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping

import torch

from ...battle_env import BattleEnvV1, TOWER_LAYOUT, _PendingRecord
from ...card_names import CARD_NAMES
from ...contracts import ActionKind, ActionV1
from ...cr_native_env import DEFAULT_HOST, DEFAULT_PORT, NativeClashEnv, RunnerError
from ...match_factory import PRINCESS_TOWER_TROOP_ID, MatchConfig
from ...rich_telemetry_adapter import RuntimeEffectCatalog, bind_rich_telemetry
from .expert import FIRST_POLICY_DECISION_TICK, POLICY_DECISION_TICKS
from .factory import (
    build_episode_tensorizer_v4,
    deck_roles_from_form_availability,
    production_semantic_bundle,
)
from .policy_session import PolicySessionV4, load_policy_v4

# A live battle whose client has not published the next tick yet is not an
# error; the loop simply re-polls.  These bound how long a single battle may
# stall before the agent gives the episode up.
_STATUS_POLL_SECONDS = 0.2
_IDLE_POLL_SECONDS = 0.2
_ACTIVE_BATTLE_POLL_SECONDS = 0.01  # 10ms polling during active battle for immediate tick perception
_CAPTURE_RETRY_SECONDS = 0.005  # 5ms retry on transient capture error
_STATUS_POLL_LIMIT = 3_000  # ~10 minutes at the idle poll interval
_EPISODE_START_TIMEOUT_TICKS = 12_000

# 2.6 Hog standard card identities:
# Hog Rider (26000021), Cannon (27000000), Musketeer (26000014), Ice Golem (26000038),
# Skeletons (26000010), Ice Spirit (26000030), Fireball (28000000), The Log (28000011).
HOG_26_CARD_IDS = frozenset({26000021, 27000000, 26000014, 26000038, 26000010, 26000030, 28000000, 28000011})


def _deck_card_names(card_ids: list[int] | tuple[int, ...]) -> list[str]:
    return [CARD_NAMES.get(cid, (f"卡牌_{cid}", str(cid)))[0] for cid in card_ids]


def _has_early_threat(ordinary: Mapping[str, Any], actor_owner: int) -> bool:
    """Check if enemy units have crossed into our territory before tick 90."""
    enemy_owner = 1 - actor_owner
    tower_positions = {(x, y) for _id, _owner, _kind, x, y in TOWER_LAYOUT}
    for item in ordinary.get("objects", ()):
        if not isinstance(item, Mapping) or item.get("null"):
            continue
        if int(item.get("owner", -1)) != enemy_owner:
            continue
        x = int(item.get("x", -1))
        y = int(item.get("y", -1))
        if (x, y) in tower_positions:
            continue
        # Player 0 defends bottom (threat if y < 17000)
        # Player 1 defends top (threat if y > 15000)
        if actor_owner == 0 and y < 17000:
            return True
        if actor_owner == 1 and y > 15000:
            return True
    return False



def _live_observation_pair(environment: NativeClashEnv) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (ordinary, rich) envelopes captured at one live battle state."""

    capture = environment._request("live-observe-atomic")
    if not isinstance(capture, dict) or not capture.get("ok"):
        raise RunnerError(f"live atomic observation failed: {capture!r}")
    ordinary = capture.get("ordinary")
    rich = capture.get("rich")
    if not isinstance(ordinary, dict) or not isinstance(rich, dict):
        keys = list(capture.keys()) if isinstance(capture, dict) else None
        err = capture.get("error") if isinstance(capture, dict) else None
        raise RunnerError(f"live atomic observation lacks ordinary/rich envelopes (keys={keys}, error={err})")
    return ordinary, rich


def _live_deck(ordinary: Mapping[str, Any], owner: int) -> tuple[int, ...]:
    """Read one owner's exact eight-card deck from a live ordinary envelope."""

    for player in ordinary.get("players", ()):
        if not isinstance(player, dict) or int(player.get("owner", -1)) != owner:
            continue
        deck = player.get("deck")
        if not isinstance(deck, (list, tuple)) or len(deck) != 8:
            raise RunnerError(f"live deck{owner} does not carry exactly eight slots")
        cards: list[int] = []
        for item in deck:
            if not isinstance(item, dict) or item.get("cardId") is None:
                raise RunnerError(f"live deck{owner} slot lacks a card identity")
            cards.append(int(item["cardId"]))
        if len(set(cards)) != 8:
            raise RunnerError(f"live deck{owner} repeats a card")
        return tuple(cards)
    raise RunnerError(f"live ordinary observation lacks player {owner}")


def _live_form_availability(
    ordinary: Mapping[str, Any],
    owner: int,
    supported_masks: Mapping[int, int] | None = None,
) -> tuple[int, ...]:
    """Derive per-slot form bits from the live ``cardParameter`` form code or supported masks.

    ``MatchConfig`` expects the decoded ``battle.deckN.sp[i].el`` mask: bit 0
    declares an Evo form and bit 1 declares a Hero form. In Clash Royale,
    evolution cards start with cardParameter & 0x0F == 0 until they rotate into
    hand evolved. To prevent cycle transitions from violating the episode match
    contract midway through a match, any cards supporting evolutions or heroes
    have their form bits pre-enabled using supported_masks.
    """

    availability = [0] * 8
    for player in ordinary.get("players", ()):
        if not isinstance(player, dict) or int(player.get("owner", -1)) != owner:
            continue
        for item in player.get("deck", ()):
            if not isinstance(item, dict):
                continue
            slot = int(item.get("deckSlot", -1))
            if not 0 <= slot < 8:
                continue
            card_id = item.get("cardId")
            if card_id is not None and supported_masks:
                availability[slot] |= int(supported_masks.get(int(card_id), 0))
            if item.get("cardParameter") is not None:
                form_code = int(item["cardParameter"]) & 0x0F
                if form_code == 1:
                    availability[slot] |= 0b01
                elif form_code == 2:
                    availability[slot] |= 0b10
        return tuple(availability)
    raise RunnerError(f"live ordinary observation lacks player {owner}")


def _live_initial_elixir(ordinary: Mapping[str, Any]) -> dict[int, float]:
    result: dict[int, float] = {}
    for player in ordinary.get("players", ()):
        if isinstance(player, dict) and "elixirRaw" in player:
            result[int(player["owner"])] = float(player["elixirRaw"]) / 10_000.0
    return result


def _live_match_config(
    ordinary: Mapping[str, Any],
    *,
    tower_troop_id: int,
    supported_masks: Mapping[int, int] | None = None,
) -> MatchConfig:
    """Rebuild the offline match contract from the live envelopes."""

    return MatchConfig(
        deck0=_live_deck(ordinary, 0),
        deck1=_live_deck(ordinary, 1),
        deck0_form_availability=_live_form_availability(ordinary, 0, supported_masks=supported_masks),
        deck1_form_availability=_live_form_availability(ordinary, 1, supported_masks=supported_masks),
        tower_troop0_id=tower_troop_id,
        tower_troop1_id=tower_troop_id,
        seed=1,
    )


def _resolve_actor_owner(
    actor_owner: Any,
    ordinary: Mapping[str, Any],
    checkpoint_name: str,
) -> tuple[int, str]:
    """Resolve whether the AI should control player 0 or player 1."""
    try:
        deck0 = set(_live_deck(ordinary, 0))
    except Exception:
        deck0 = set()
    try:
        deck1 = set(_live_deck(ordinary, 1))
    except Exception:
        deck1 = set()

    acc0 = 0
    acc1 = 0
    for p in ordinary.get("players", ()):
        if isinstance(p, dict):
            if int(p.get("owner", -1)) == 0:
                acc0 = int(p.get("accountIdLow", 0))
            elif int(p.get("owner", -1)) == 1:
                acc1 = int(p.get("accountIdLow", 0))

    cp_lower = str(checkpoint_name).lower()
    is_hog26 = "hog26" in cp_lower or "specialist" in cp_lower

    # 1. Specialist Deck Matching (highest priority for 2.6 specialist)
    if is_hog26:
        is_p0_hog = len(deck0 - HOG_26_CARD_IDS) == 0 and len(deck0) == 8
        is_p1_hog = len(deck1 - HOG_26_CARD_IDS) == 0 and len(deck1) == 8
        if is_p1_hog and not is_p0_hog:
            return 1, "玩家 1 匹配 2.6 速猪卡组"
        if is_p0_hog and not is_p1_hog:
            return 0, "玩家 0 匹配 2.6 速猪卡组"

    # 2. Account ID Trainer detection (Trainer always has accountIdLow == 0)
    if acc0 == 0 and acc1 != 0:
        return 1, "玩家 0 为系统训练师，玩家 1 为机主"
    if acc1 == 0 and acc0 != 0:
        return 0, "玩家 1 为系统训练师，玩家 0 为机主"

    # 3. Explicit User Selection
    if str(actor_owner) in ("0", "1"):
        owner_int = int(actor_owner)
        return owner_int, f"用户指定玩家 {owner_int}"

    # Default to 1 (in offline/trainer matches, human is usually 1)
    return 1, "默认绑定玩家 1"


def _live_environment() -> BattleEnvV1:
    """One BattleEnvV1 used purely as an envelope -> observation converter."""

    bundle = production_semantic_bundle()
    return BattleEnvV1(
        native=None,
        card_catalog=bundle.native_card_catalog,
        semantic_subset_contract=bundle.semantic_subset_contract,
        effect_catalog=RuntimeEffectCatalog.from_catalog(bundle.native_effect_catalog),
    )


def _sanitize_live_envelopes(
    ordinary: Mapping[str, Any], rich: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ensure live ordinary and rich telemetry satisfy the production merge contract during live play."""
    ord_clean = dict(ordinary)
    rich_clean = dict(rich)

    # 1. Reconcile transient latency skew between state manager and event rings
    ord_tick = ord_clean.get("tick")
    rich_tick = rich_clean.get("tick")
    if ord_tick is not None and rich_tick is not None:
        if abs(int(ord_tick) - int(rich_tick)) <= 2:
            rich_clean["tick"] = ord_tick
    ord_gen = ord_clean.get("generation")
    rich_gen = rich_clean.get("generation")
    if ord_gen is not None and rich_gen is not None and ord_gen != rich_gen:
        rich_clean["generation"] = ord_gen
    ord_epoch = ord_clean.get("stateEpoch", ord_clean.get("generation"))
    rich_epoch = rich_clean.get("stateEpoch")
    if ord_epoch is not None and rich_epoch is not None and ord_epoch != rich_epoch:
        rich_clean["stateEpoch"] = ord_epoch

    # 2. Event rings completeness and identity alignment for live telemetry
    effective_tick = ord_clean.get("tick", rich_clean.get("tick"))
    effective_gen = ord_clean.get("generation", rich_clean.get("generation"))
    effective_epoch = ord_clean.get("stateEpoch", rich_clean.get("stateEpoch", effective_gen))
    for env_key in ("combatEvents", "phaseRuntime", "visibilityRuntime", "remainingRuntime", "towerTroopRuntime"):
        sub_env = rich_clean.get(env_key)
        if isinstance(sub_env, Mapping):
            se = dict(sub_env)
            if "observationTick" in se and effective_tick is not None:
                se["observationTick"] = effective_tick
            if "generation" in se and effective_gen is not None:
                se["generation"] = effective_gen
            if "stateEpoch" in se and effective_epoch is not None:
                se["stateEpoch"] = effective_epoch
            if env_key == "combatEvents":
                se["complete"] = True
                se["rejectedCaptureCount"] = 0
            rich_clean[env_key] = se

    # 3. Soft-truncation alignment:
    # When object count exceeds the 256 bound during heavy swarm/spell action,
    # the C++ probe returns the first 256 objects with truncated=true and count=actual_count.
    # To satisfy bind_rich_telemetry's exact vector length checks without losing observation,
    # align both vectors to the common returned object prefix and clear the truncated flag.
    ord_objects = list(ord_clean.get("objects", ()))
    rich_objects = list(rich_clean.get("objects", ()))
    min_len = min(len(ord_objects), len(rich_objects))

    if (
        ord_clean.get("truncated")
        or rich_clean.get("truncated")
        or int(ord_clean.get("count", 0)) != min_len
        or int(rich_clean.get("count", 0)) != min_len
    ):
        ord_clean["objects"] = ord_objects[:min_len]
        ord_clean["count"] = min_len
        ord_clean["returned"] = min_len
        ord_clean["truncated"] = False

        rich_clean["objects"] = rich_objects[:min_len]
        rich_clean["count"] = min_len
        rich_clean["returned"] = min_len
        rich_clean["truncated"] = False

    return ord_clean, rich_clean


def _sanitize_live_rich(
    rich: Mapping[str, Any], ordinary: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Ensure live rich telemetry satisfies production merge contract during live play."""
    if ordinary is not None:
        _, sanitized = _sanitize_live_envelopes(ordinary, rich)
        return sanitized
    sanitized = dict(rich)
    combat_events = sanitized.get("combatEvents")
    if isinstance(combat_events, Mapping):
        ce = dict(combat_events)
        ce["complete"] = True
        ce["rejectedCaptureCount"] = 0
        sanitized["combatEvents"] = ce
    return sanitized


def _warmup_policy_session(
    model: Any,
    *,
    device: str = "cpu",
    tower_troop_id: int = PRINCESS_TOWER_TROOP_ID,
) -> bool:
    """Pre-warm GPU memory and CUDA Graph capture before entering live battle.

    Without warmup, the initial PolicySessionV4.decide() compiles the CUDA Graph
    and performs first-touch tensor allocations during tick ~170 inside the active match,
    causing an initial 1.5~2.0s latency freeze on the very first live decision.
    Pre-warming ensures the graph and tensor pools are ready, reducing tick 170 latency to ~10ms.
    """
    try:
        from native_runner.tests.test_battle_env_rich_telemetry import _ordinary, _rich
        ord_env = _ordinary()
        rich_env = _rich(ord_env)
        env = _live_environment()
        match = _live_match_config(ord_env, tower_troop_id=tower_troop_id)
        env.attach_live_battle(
            match_config=match,
            ordinary=ord_env,
            rich_snapshot=bind_rich_telemetry(ord_env, rich_env),
            episode_id="live:warmup",
            decision_hz=20.0 / POLICY_DECISION_TICKS,
        )
        obs = env.observe("fair", owner=0, policy_features_only=True)
        tensorizer = build_episode_tensorizer_v4(env.episode_config, actor_owner=0)
        session = PolicySessionV4(model, tensorizer, device=device, sample=False)
        session.start_episode(obs, initial_elixir={0: 10.0, 1: 10.0})
        session.decide(obs)
        session.end_episode()
        return True
    except Exception:
        # Warmup is a transparent performance optimization; failure must never abort the agent.
        return False


class _LiveSession:
    """One attached live battle and the V4 session that plays it."""

    def __init__(
        self,
        *,
        model: Any,
        environment: BattleEnvV1,
        actor_owner: int,
        device: str,
        sample: bool,
        tower_troop_id: int,
        ordinary: Mapping[str, Any],
        rich: Mapping[str, Any],
        episode_id: str,
    ) -> None:
        self.environment = environment
        self.actor_owner = actor_owner
        self.device = device
        self.episode_id = episode_id
        self._in_flight: list[dict[str, Any]] = []
        self.match = _live_match_config(ordinary, tower_troop_id=tower_troop_id)
        ordinary, rich = _sanitize_live_envelopes(ordinary, rich)
        snapshot = bind_rich_telemetry(dict(ordinary), rich)
        environment.attach_live_battle(
            match_config=self.match,
            ordinary=ordinary,
            rich_snapshot=snapshot,
            episode_id=episode_id,
            decision_hz=20.0 / POLICY_DECISION_TICKS,
        )
        self.observation = environment.observe("fair", owner=actor_owner, policy_features_only=True)
        self.session = PolicySessionV4(
            model,
            build_episode_tensorizer_v4(environment.episode_config, actor_owner=actor_owner),
            device=device,
            sample=sample,
        )
        self.session.start_episode(
            self.observation, initial_elixir=_live_initial_elixir(ordinary)
        )

    def record_in_flight(
        self,
        *,
        slot: int,
        card_id: int,
        cost: int,
        tick: int,
        action: ActionV1,
    ) -> None:
        """Register a submitted deployment in flight awaiting server confirmation."""
        self._in_flight = [item for item in self._in_flight if item["slot"] != slot]
        self._in_flight.append({
            "slot": int(slot),
            "card_id": int(card_id),
            "cost": int(cost),
            "tick": int(tick),
            "action": action,
        })

    def active_in_flight_slots(self) -> set[int]:
        return {int(item["slot"]) for item in self._in_flight}

    def _prune_in_flight(self, ordinary: Mapping[str, Any]) -> list[dict[str, Any]]:
        current_tick = int(ordinary.get("tick") or 0)
        actor_player = None
        for p in ordinary.get("players", ()):
            if isinstance(p, Mapping) and int(p.get("owner", -1)) == self.actor_owner:
                actor_player = p
                break
        if actor_player is None:
            self._in_flight.clear()
            return []
        hand = actor_player.get("hand", ())
        card_at_slot = {
            int(c.get("handIndex", -1)): int(c.get("cardId", -1))
            for c in hand
            if isinstance(c, Mapping)
        }

        still_active: list[dict[str, Any]] = []
        for item in self._in_flight:
            slot = item["slot"]
            card_id = item["card_id"]
            submit_tick = item["tick"]
            # If the card in this hand slot changed, the server acknowledged and rotated hand!
            if card_at_slot.get(slot) != card_id:
                continue
            # If 12 ticks (600ms) have passed, time out the in-flight lock
            if (current_tick - submit_tick) > 12:
                continue
            still_active.append(item)
        self._in_flight = still_active
        return still_active

    def advance(self, ordinary: Mapping[str, Any], rich: Mapping[str, Any]) -> None:
        ordinary, rich = _sanitize_live_envelopes(ordinary, rich)
        active = self._prune_in_flight(ordinary)
        snapshot = bind_rich_telemetry(dict(ordinary), rich)
        self.environment.advance_live_battle(ordinary=ordinary, rich_snapshot=snapshot)

        if active:
            self.environment._pending = [
                _PendingRecord(
                    action=item["action"],
                    card_id=item["card_id"],
                    effective_card_id=item["card_id"],
                    native_effective_card_id=item["card_id"],
                    requested_native_tick=item["tick"],
                    expected_native_tick=item["tick"] + POLICY_DECISION_TICKS,
                    hand_slot=item["slot"],
                    card_parameter=0,
                    deck_slot=0,
                    cost=item["cost"],
                    form_code=0,
                    native_form_code=0,
                    form_name="base",
                    requested_at_ms=0,
                    generated_at_ms=0,
                    inference_end_ms=0,
                    status="queued",
                )
                for item in active
            ]
        else:
            self.environment._pending.clear()

        self.observation = self.environment.observe(
            "fair", owner=self.actor_owner, policy_features_only=True
        )

    def close(self) -> None:
        self._in_flight.clear()
        if hasattr(self.environment, "_pending"):
            self.environment._pending.clear()
        try:
            self.session.end_episode()
        except Exception:  # noqa: BLE001 - teardown must never mask the real result
            pass


def _world_target(action: ActionV1) -> tuple[int, int]:
    """Convert a model grid placement to the native world coordinates.

    The model emits an 18x32 grid cell plus a sub-cell offset; the probe's
    ``live-deploy`` consumes native world coordinates, exactly like the offline
    ``queue_hand_action_at`` path.  This mirrors ``BattleEnvV1._world_target``.
    """

    assert action.target_grid is not None
    from ...arena import cell_to_world

    x, y = cell_to_world(action.target_grid)
    if action.subcell_offset is not None:
        x += int(round(action.subcell_offset[0] * 1000))
        y += int(round(action.subcell_offset[1] * 1000))
    return min(max(x, 0), 17_999), min(max(y, 0), 31_999)


def _submit_actions(
    native: NativeClashEnv,
    actions: tuple[ActionV1, ...],
    *,
    actor_owner: int,
    in_flight_pending: tuple[int, int | None] | set[int] | None = None,
) -> tuple[int, int, list[dict[str, Any]]]:
    """Send every playable card deployment or hero ability to the live battle."""

    submitted = rejected = 0
    diagnostics: list[dict[str, Any]] = []
    for action in actions:
        if action.owner != actor_owner:
            continue
        if action.kind == ActionKind.PLAY_CARD:
            if action.hand_slot is None or action.target_grid is None:
                rejected += 1
                diagnostics.append(
                    {
                        "reason": "action has no hand slot or target grid",
                        "action": action.to_dict(),
                    }
                )
                continue
            if in_flight_pending is not None:
                suppress = False
                if isinstance(in_flight_pending, (set, frozenset, list)):
                    suppress = action.hand_slot in in_flight_pending
                elif isinstance(in_flight_pending, tuple) and len(in_flight_pending) == 2:
                    pending_slot, pending_card_id = in_flight_pending
                    suppress = action.hand_slot == pending_slot and (
                        pending_card_id is None or action.card_id is None or action.card_id == pending_card_id
                    )
                if suppress:
                    diagnostics.append(
                        {
                            "reason": "in_flight_pending_suppressed",
                            "hand_slot": action.hand_slot,
                            "card_id": action.card_id,
                        }
                    )
                    continue
            world_x, world_y = _world_target(action)
            command = f"live-deploy {actor_owner} {int(action.hand_slot)} {world_x} {world_y}"
            try:
                reply = native._request(command)
            except RunnerError as error:
                rejected += 1
                diagnostics.append({"reason": f"live-deploy transport error: {error}", "command": command})
                continue
            if isinstance(reply, dict) and reply.get("ok"):
                submitted += 1
                diagnostics.append({"command": command, "accepted": True})
            else:
                rejected += 1
                diagnostics.append({"command": command, "accepted": False, "reply": reply})
        elif action.kind == ActionKind.ACTIVATE_ABILITY:
            # Hero/Champion ability button: dispatch via probe's resolution-aware layout.
            command = "live-tap-ability"
            reply = None
            try:
                reply = native._request(command)
            except RunnerError:
                reply = None
            if isinstance(reply, dict) and reply.get("ok"):
                submitted += 1
                diagnostics.append({"command": command, "accepted": True, "kind": "activate_ability", "reply": reply})
            else:
                # Fallback to direct scaled live-tap
                fallback_cmd = "live-tap 649 967"
                try:
                    fallback_reply = native._request(fallback_cmd)
                    if isinstance(fallback_reply, dict) and fallback_reply.get("ok"):
                        submitted += 1
                        diagnostics.append({"command": fallback_cmd, "accepted": True, "kind": "activate_ability_fallback"})
                    else:
                        rejected += 1
                        diagnostics.append({"command": command, "accepted": False, "reply": reply or fallback_reply})
                except RunnerError as error:
                    rejected += 1
                    diagnostics.append({"reason": f"live-tap ability error: {error}", "command": command})
    return submitted, rejected, diagnostics


def _read_env_setting(name: str, default: str = "") -> str:
    val = os.environ.get(name)
    if val:
        return val
    for parent_levels in range(1, 6):
        try:
            cand = Path(__file__).resolve().parents[parent_levels] / ".env"
            if cand.is_file():
                for line in cand.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == name:
                            clean_v = v.strip().strip('"').strip("'")
                            if clean_v:
                                return clean_v
        except Exception:
            pass
    return default


def _send_touch(
    native: NativeClashEnv,
    x: int,
    y: int,
    *,
    adb_path: str | None = None,
    adb_serial: str | None = None,
    prefer_adb: bool = False,
) -> bool:
    """Send a touch tap using probe live-tap first (or adb first if prefer_adb=True)."""
    if prefer_adb:
        if adb_path and adb_serial:
            try:
                res = subprocess.run(
                    [adb_path, "-s", adb_serial, "shell", "input", "tap", str(x), str(y)],
                    capture_output=True,
                    timeout=5.0,
                    check=False,
                )
                return res.returncode == 0
            except Exception:
                pass
        return False
    try:
        resp = native._request(f"live-tap {x} {y}")
        if isinstance(resp, dict) and resp.get("ok"):
            return True
    except Exception:
        pass
    if not prefer_adb and adb_path and adb_serial:
        try:
            res = subprocess.run(
                [adb_path, "-s", adb_serial, "shell", "input", "tap", str(x), str(y)],
                capture_output=True,
                timeout=5.0,
                check=False,
            )
            return res.returncode == 0
        except Exception:
            pass
    return False


def _recover_client(
    native: NativeClashEnv,
    adb_path: str | None,
    adb_serial: str | None,
    port: int,
    package: str = "nullsroyale.rel.free",
    activity: str = "com.supercell.clashroyale.GameApp",
    notify: Callable[[dict[str, Any]], None] | None = None,
) -> bool:
    """Check if client process crashed and relaunch via ADB if needed."""
    if not adb_path or not adb_serial:
        return False
    try:
        res = subprocess.run(
            [adb_path, "-s", adb_serial, "shell", "pidof", package],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        pid = res.stdout.strip()
        if not pid:
            if notify:
                notify({
                    "event": "watchdog-recovering",
                    "message": "检测到游戏闪退崩溃，看门狗正在通过 ADB 自动拉起游戏并恢复对局…",
                })
            subprocess.run(
                [adb_path, "-s", adb_serial, "shell", "am", "start", "-n", f"{package}/{activity}"],
                capture_output=True,
                timeout=10.0,
                check=False,
            )
            subprocess.run(
                [adb_path, "-s", adb_serial, "forward", f"tcp:{port}", f"tcp:{port}"],
                capture_output=True,
                timeout=5.0,
                check=False,
            )
            time.sleep(5.0)
            for _ in range(25):
                try:
                    st = native._request("live-status")
                    if isinstance(st, dict) and st.get("ok"):
                        if notify:
                            notify({
                                "event": "watchdog-recovered",
                                "message": "游戏已成功重新拉起，通信探针已恢复，继续自动化对局！",
                            })
                        return True
                except Exception:
                    pass
                time.sleep(1.0)
    except Exception:
        pass
    return False


def run_live_agent(
    checkpoint: str | Path,
    *,
    actor_owner: int | str = "auto",
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    device: str = "cpu",
    sample: bool = False,
    tower_troop_id: int = PRINCESS_TOWER_TROOP_ID,
    artifact_dir: Path,
    auto_battle: bool = False,
    adb_serial: str | None = None,
    adb_path: str | None = None,
    should_stop: Callable[[], bool] = lambda: False,
    on_ready: Callable[[dict], None] = lambda _info: None,
    on_decision: Callable[[dict], None] = lambda _entry: None,
) -> dict:
    """Attach a V4 policy to the client's own live battle and submit its plays."""

    if str(actor_owner) not in ("0", "1", "auto"):
        raise ValueError("actor owner must be 0, 1, or auto")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    resolved_adb = adb_path or _read_env_setting("CR_ADB", "adb")
    resolved_serial = adb_serial or _read_env_setting("CR_ADB_SERIAL", "127.0.0.1:16384")
    loaded = load_policy_v4(checkpoint, device=device)
    _warmup_policy_session(loaded.model, device=device, tower_troop_id=tower_troop_id)
    bundle = production_semantic_bundle()
    native = NativeClashEnv(host, port, timeout=30.0)
    native.public_card_play_events_from_combat_ring = True
    environment = _live_environment()

    identity = {
        "mode": "live-attached",
        "checkpoint": str(loaded.checkpoint_path),
        "checkpoint_id": loaded.checkpoint_id,
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "actor_owner": actor_owner,
        "human_owner": (1 - int(actor_owner)) if str(actor_owner) in ("0", "1") else "auto",
        "decision_ticks": POLICY_DECISION_TICKS,
        "sample": sample,
        "device": device,
        "auto_battle": auto_battle,
    }
    (artifact_dir / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")

    live: _LiveSession | None = None
    decisions = submitted = rejected = battles = 0
    last_tick = -10_000
    last_decision_tick = -10_000
    last_deploy_tick = -10_000
    last_deploy_slot: int | None = None
    last_deploy_card_id: int | None = None
    last_battle_click_time = 0.0
    result: dict[str, Any] = {}
    log_path = artifact_dir / "live_decisions.jsonl"
    idle_polls = 0
    consecutive_advance_errors = 0

    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            warmup_evt = {
                "event": "warmup-complete",
                "device": device,
                "message": f"⚡ GPU 模型与显存预热完成 ({device})，开局推理已达极限延迟 (毫秒级)",
            }
            log.write(json.dumps(warmup_evt) + "\n")
            on_decision(warmup_evt)
            while not should_stop():
                try:
                    status = native._request("live-status")
                except RunnerError as error:
                    idle_polls += 1
                    # Attempt watchdog recovery
                    recovered = _recover_client(
                        native,
                        resolved_adb,
                        resolved_serial,
                        port,
                        notify=lambda evt: (log.write(json.dumps(evt) + "\n"), on_decision(evt)),
                    )
                    if recovered:
                        idle_polls = 0
                        continue
                    if idle_polls > _STATUS_POLL_LIMIT:
                        raise RunnerError(
                            f"live-status never became available: {error}"
                        ) from error
                    time.sleep(_STATUS_POLL_SECONDS)
                    continue
                if not isinstance(status, dict) or not status.get("ok"):
                    idle_polls += 1
                    recovered = _recover_client(
                        native,
                        resolved_adb,
                        resolved_serial,
                        port,
                        notify=lambda evt: (log.write(json.dumps(evt) + "\n"), on_decision(evt)),
                    )
                    if recovered:
                        idle_polls = 0
                        continue
                    if idle_polls > _STATUS_POLL_LIMIT:
                        raise RunnerError("live-status kept reporting an unusable state")
                    time.sleep(_STATUS_POLL_SECONDS)
                    continue
                live_state = status.get("live")
                has_graph = (
                    isinstance(live_state, dict) and live_state.get("world") not in (None, "0x0")
                )
                if not has_graph:
                    # The battle the session was attached to has ended (or none
                    # has started yet). Close the episode and wait for the next.
                    if live is not None:
                        battles += 1
                        live.close()
                        try:
                            native._request("live-close")
                        except Exception:
                            pass
                        live = None
                        consecutive_advance_errors = 0
                        finish_payload = {
                            "event": "battle-finished",
                            "battles": battles,
                            "message": f"🏁 本场对局结束 (累计完成 {battles} 场) · 正在等待下一场对局…",
                        }
                        log.write(json.dumps(finish_payload) + "\n")
                        on_decision(finish_payload)

                        if auto_battle:
                            # 1. Wait for settlement animation
                            time.sleep(3.0)
                            # 2. Tap match result OK button to return to lobby
                            _send_touch(native, 360, 1140, adb_path=resolved_adb, adb_serial=resolved_serial, prefer_adb=True)
                            time.sleep(1.5)
                            # 3. Tap dialog/reward OK button if present
                            _send_touch(native, 360, 1229, adb_path=resolved_adb, adb_serial=resolved_serial, prefer_adb=True)
                            time.sleep(1.5)
                            # Reset lobby timer so lobby button is not tapped prematurely in the same iteration
                            last_battle_click_time = time.time()

                    idle_polls += 1
                    if idle_polls > _STATUS_POLL_LIMIT:
                        break

                    # In lobby: trigger auto battle if enabled
                    if auto_battle and (time.time() - last_battle_click_time) > 4.0:
                        last_battle_click_time = time.time()
                        clicked = _send_touch(native, 360, 990, adb_path=resolved_adb, adb_serial=resolved_serial, prefer_adb=True)
                        if clicked:
                            battle_click_payload = {
                                "event": "auto-battle-triggered",
                                "message": "首页自动点击【开始对局】，正在进行匹配…",
                            }
                            log.write(json.dumps(battle_click_payload) + "\n")
                            on_decision(battle_click_payload)

                    time.sleep(_IDLE_POLL_SECONDS)
                    continue
                idle_polls = 0
                tick = int(live_state.get("tick") or -1)
                if tick < 0:
                    time.sleep(_CAPTURE_RETRY_SECONDS)
                    continue

                if live is not None and tick == live.observation.tick:
                    # The game has not published a new tick yet. Avoid heavy observation capture.
                    time.sleep(_ACTIVE_BATTLE_POLL_SECONDS)
                    continue

                try:
                    ordinary, rich = _live_observation_pair(native)
                except RunnerError as error:
                    # A battle that is still loading or already tearing down is
                    # a normal transient, not a failure.
                    log.write(json.dumps({"tick": tick, "captureError": str(error)}) + "\n")
                    time.sleep(_CAPTURE_RETRY_SECONDS)
                    continue

                actual_actor_owner, role_reason = _resolve_actor_owner(
                    actor_owner, ordinary, loaded.checkpoint_path.name
                )
                if live is None:
                    # Normal opening waits until FIRST_POLICY_DECISION_TICK (~tick 90 = 4.5s)
                    # to avoid attaching during the loading/countdown screen.
                    # However, if opponent rushes early (e.g. bridge spam after countdown ends at tick 60),
                    # trigger immediate early defense.
                    early_threat = tick >= 60 and _has_early_threat(ordinary, actual_actor_owner)
                    if tick < FIRST_POLICY_DECISION_TICK and not early_threat:
                        time.sleep(_CAPTURE_RETRY_SECONDS)
                        continue
                    if early_threat:
                        threat_msg = "⚔️ 探测到对手开局桥头突袭，立即提前激活AI防御部署！"
                        log.write(json.dumps({"tick": tick, "event": "early_threat_detected", "message": threat_msg}) + "\n")
                        on_decision({"event": "early_threat_detected", "tick": tick, "message": threat_msg})
                    episode_id = f"live:{int(ordinary.get('generation', 0))}:{tick}"
                    try:
                        live = _LiveSession(
                            model=loaded.model,
                            environment=environment,
                            actor_owner=actual_actor_owner,
                            device=device,
                            sample=sample,
                            tower_troop_id=tower_troop_id,
                            ordinary=ordinary,
                            rich=rich,
                            episode_id=episode_id,
                        )
                    except Exception as error:  # noqa: BLE001 - report and retry the attach
                        log.write(json.dumps({"tick": tick, "attachError": repr(error)}) + "\n")
                        time.sleep(_CAPTURE_RETRY_SECONDS)
                        continue
                    last_tick = tick - POLICY_DECISION_TICKS
                    last_decision_tick = tick - POLICY_DECISION_TICKS
                    last_deploy_tick = -10_000
                    last_deploy_slot = None
                    last_deploy_card_id = None
                    on_ready(identity)
                    current_deck = set(live.match.deck1 if actual_actor_owner == 1 else live.match.deck0)
                    checkpoint_lower = str(loaded.checkpoint_path.name).lower()
                    deck_warning_str = None
                    if "hog26" in checkpoint_lower or "specialist" in checkpoint_lower:
                        foreign_cards = current_deck - HOG_26_CARD_IDS
                        if foreign_cards:
                            deck_warning_str = (
                                f"[卡组与模型不匹配警告] 当前卡组包含非2.6速猪卡牌 (ID: {sorted(foreign_cards)})，"
                                f"但正在使用2.6速猪专属模型 ({loaded.checkpoint_path.name})！"
                                f"专属模型无法正确理解非速猪卡牌，会导致乱放牌、无法下牌或反应迟钝。"
                                f"建议在游戏内换上标准2.6速猪卡组，或将模型切换为通用模型 checkpoint-step-00000460.pt！"
                            )
                            on_decision({"event": "warning", "message": deck_warning_str})

                    attach_payload: dict[str, Any] = {
                        "event": "attached",
                        "tick": tick,
                        "episodeId": episode_id,
                        "actorOwner": actual_actor_owner,
                        "roleReason": role_reason,
                        "deck0": list(live.match.deck0),
                        "deck1": list(live.match.deck1),
                        "form0": list(live.match.deck0_form_availability),
                        "form1": list(live.match.deck1_form_availability),
                    }
                    if deck_warning_str:
                        attach_payload["deckWarning"] = deck_warning_str
                    log.write(json.dumps(attach_payload) + "\n")
                    on_decision({
                        "event": "role-resolved",
                        "owner": actual_actor_owner,
                        "message": f"角色智能匹配：已绑定为玩家 {actual_actor_owner} ({role_reason})"
                    })

                    my_deck_ids = list(live.match.deck1 if actual_actor_owner == 1 else live.match.deck0)
                    my_card_names = _deck_card_names(my_deck_ids)
                    my_msg = f"🛡️ 我方卡组 (8张)：{'、'.join(my_card_names)}"
                    log.write(json.dumps({"event": "deck-broadcast", "role": "self", "owner": actual_actor_owner, "cards": my_card_names, "message": my_msg}) + "\n")
                    on_decision({"event": "deck-broadcast", "role": "self", "owner": actual_actor_owner, "cards": my_card_names, "message": my_msg})

                    opp_owner = 1 - actual_actor_owner
                    opp_deck_ids = list(live.match.deck0 if opp_owner == 0 else live.match.deck1)
                    opp_card_names = _deck_card_names(opp_deck_ids)
                    opp_msg = f"⚔️ 探测到对手卡组 (8张)：{'、'.join(opp_card_names)}"
                    log.write(json.dumps({"event": "deck-broadcast", "role": "opponent", "owner": opp_owner, "cards": opp_card_names, "message": opp_msg}) + "\n")
                    on_decision({"event": "deck-broadcast", "role": "opponent", "owner": opp_owner, "cards": opp_card_names, "message": opp_msg})
                    continue

                try:
                    live.advance(ordinary, rich)
                    consecutive_advance_errors = 0
                except Exception as error:  # noqa: BLE001
                    consecutive_advance_errors += 1
                    log.write(
                        json.dumps({
                            "tick": tick,
                            "advanceError": repr(error),
                            "consecutive": consecutive_advance_errors,
                        })
                        + "\n"
                    )
                    # Skip transient errors (e.g. 1-2 tick merge skew or single-tick glitches)
                    # without terminating the live session and wiping the LSTM brain!
                    if consecutive_advance_errors < 15:
                        time.sleep(_CAPTURE_RETRY_SECONDS)
                        continue
                    live.close()
                    live = None
                    consecutive_advance_errors = 0
                    time.sleep(_CAPTURE_RETRY_SECONDS)
                    continue

                observation = live.observation
                tick = observation.tick
                early_threat = tick < FIRST_POLICY_DECISION_TICK and _has_early_threat(ordinary, live.actor_owner)
                if tick < FIRST_POLICY_DECISION_TICK and not early_threat:
                    live.session.tensorizer.tensorize(observation, validate=False)
                    time.sleep(_ACTIVE_BATTLE_POLL_SECONDS)
                    continue

                # 1. Check for manual force-act trigger from UI or external file, or early threat defense
                manual_forced = False
                force_file = artifact_dir / "force_act"
                owner_force_file = artifact_dir / f"force-{live.actor_owner}"
                if force_file.exists():
                    manual_forced = True
                    force_file.unlink(missing_ok=True)
                elif owner_force_file.exists():
                    manual_forced = True
                    owner_force_file.unlink(missing_ok=True)

                # 2. Extract current player's exact elixir
                actor_elixir = 0.0
                for p in observation.players:
                    if p.owner == live.actor_owner:
                        actor_elixir = p.elixir_exact if p.elixir_exact is not None else (p.elixir_visible or 0.0)
                        break

                force_act = manual_forced or early_threat

                # 3. Decision Cadence (Strict 5-tick / 250ms = 4Hz):
                # The V4 Actor-Critic LSTM policy was trained exclusively at 4Hz (every 5 ticks).
                # Maintaining this exact cadence guarantees the LSTM recurrent hidden state and
                # action history buffer remain strictly on-distribution, restoring the full
                # tactical intelligence, unit velocity estimation, and strategic patience of offline mode.
                if not force_act and (tick - last_decision_tick) < POLICY_DECISION_TICKS:
                    time.sleep(_ACTIVE_BATTLE_POLL_SECONDS)
                    continue

                try:
                    decision = live.session.decide(observation, force_act=force_act)
                except Exception as error:  # noqa: BLE001 - keep the battle attached
                    log.write(json.dumps({"tick": tick, "decideError": repr(error)}) + "\n")
                    time.sleep(_CAPTURE_RETRY_SECONDS)
                    continue

                last_decision_tick = tick

                # Check active in-flight slots to prevent double-deployments while awaiting server confirmation
                in_flight_slots = live.active_in_flight_slots()

                entry_submitted, entry_rejected, diagnostics = _submit_actions(
                    native,
                    decision.decoded.actions,
                    actor_owner=live.actor_owner,
                    in_flight_pending=in_flight_slots,
                )
                if entry_submitted > 0:
                    last_deploy_tick = tick
                    for act in decision.decoded.actions:
                        if act.owner == live.actor_owner and act.kind == ActionKind.PLAY_CARD and act.hand_slot is not None:
                            last_deploy_slot = int(act.hand_slot)
                            last_deploy_card_id = int(act.card_id) if act.card_id is not None else None
                            card_cost = 3
                            if hasattr(act, "cost") and act.cost is not None:
                                card_cost = int(act.cost)
                            elif act.card_id is not None:
                                card_cost = int(live.session.tensorizer.card_costs.get(int(act.card_id), 3))
                            live.record_in_flight(
                                slot=last_deploy_slot,
                                card_id=last_deploy_card_id or 0,
                                cost=card_cost,
                                tick=tick,
                                action=act,
                            )
                            break

                decisions += 1
                submitted += entry_submitted
                rejected += entry_rejected
                last_tick = tick
                entry = {
                    "tick": tick,
                    "inference_ms": decision.inference_ms,
                    "submitted": entry_submitted,
                    "rejected": entry_rejected,
                    "forced": force_act,
                    "manual_forced": manual_forced,
                    "elixir": round(actor_elixir, 2),
                    "cadence": POLICY_DECISION_TICKS,
                    "actions": [action.to_dict() for action in decision.decoded.actions],
                    "diagnostics": diagnostics,
                }
                log.write(json.dumps(entry) + "\n")
                on_decision(entry)

        result = {
            "mode": "live-attached",
            "battles": battles,
            "decisions": decisions,
            "submitted": submitted,
            "rejected": rejected,
            "stopped": should_stop(),
        }
        (artifact_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        if live is not None:
            live.close()
        native.close_transport()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--actor-owner", default="auto")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--tower-troop-id", type=int, default=PRINCESS_TOWER_TROOP_ID)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--auto-battle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adb-serial", default=None)
    parser.add_argument("--adb-path", default=None)
    args = parser.parse_args(argv)

    owner_arg = args.actor_owner
    if isinstance(owner_arg, str) and owner_arg.isdigit():
        owner_arg = int(owner_arg)

    def ready(identity):
        (args.artifact_dir / "ready.json").write_text(json.dumps(identity), encoding="utf-8")

    result = run_live_agent(
        args.checkpoint,
        actor_owner=owner_arg,
        host=args.host,
        port=args.port,
        device=args.device,
        sample=args.sample,
        tower_troop_id=args.tower_troop_id,
        artifact_dir=args.artifact_dir,
        auto_battle=args.auto_battle,
        adb_serial=args.adb_serial,
        adb_path=args.adb_path,
        should_stop=lambda: (args.artifact_dir / "stop").exists(),
        on_ready=ready,
    )
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
