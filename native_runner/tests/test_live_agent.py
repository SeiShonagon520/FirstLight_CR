"""Live (server-driven) attach path: episode construction and action submission.

These cover the parts of the online agent that used to be unreachable: the
tensorizer could not even be constructed, the converter had no episode identity
or bound towers, and every decoded action was dropped because the hand slot and
placement were read from metadata keys that no producer ever writes.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from native_runner.arena import GRID_HEIGHT, GRID_WIDTH, cell_to_world
from native_runner.battle_env import TOWER_LAYOUT
from native_runner.contracts import ActionKind, ActionV1, TargetKind
from native_runner.cr_native_env import RunnerError
from native_runner.training.v4 import live_agent
from native_runner.training.v4.live_agent import (
    _live_deck,
    _live_form_availability,
    _live_initial_elixir,
    _live_match_config,
    _submit_actions,
    _world_target,
)
from native_runner.training.v4.policy_session import load_policy_v4
from test_battle_env_rich_telemetry import _ordinary, _rich

CHECKPOINT = "./checkpoints/2_6hog_expert/hog26-specialist1.pt"


def _play(hand_slot: int, grid: tuple[int, int], *, owner: int = 0, offset: tuple[float, float] | None = None) -> ActionV1:
    """A card play carrying the fields the model actually emits."""

    return ActionV1(
        owner=owner,
        kind=ActionKind.PLAY_CARD,
        hand_slot=hand_slot,
        target_kind=TargetKind.GRID,
        target_grid=grid,
        subcell_offset=offset,
    )


def test_live_deck_and_elixir_come_from_the_live_envelope() -> None:
    ordinary = _ordinary()

    assert len(_live_deck(ordinary, 0)) == 8
    assert len(_live_deck(ordinary, 1)) == 8
    # The probe reports elixir in ten-thousandths, not as a float field.
    assert _live_initial_elixir(ordinary) == {0: 6.0, 1: 6.0}


def test_live_form_availability_reads_the_live_form_code() -> None:
    ordinary = _ordinary()
    # cardParameter & 0xF names EvoForm as 1 and HeroForm as 2.
    ordinary["players"][0]["deck"][3]["cardParameter"] = 1
    ordinary["players"][0]["deck"][5]["cardParameter"] = 2
    ordinary["players"][1]["deck"][2]["cardParameter"] = 2
    ordinary["players"][1]["deck"][7]["cardParameter"] = 1

    assert _live_form_availability(ordinary, 0) == (0, 0, 0, 0b01, 0, 0b10, 0, 0)
    assert _live_form_availability(ordinary, 1) == (0, 0, 0b10, 0, 0, 0, 0, 0b01)


def test_live_match_config_is_a_valid_offline_contract() -> None:
    from native_runner.card_specs import build_card_catalog
    from native_runner.semantic_subset import build_semantic_supported_subset

    ordinary = _ordinary()
    # 26000005 (Minions) carries no supported form, so use a slot whose card
    # does: slot 4 is Goblins (26000002), which the subset admits as a Hero.
    ordinary["players"][0]["deck"][4]["cardParameter"] = 2
    match = _live_match_config(ordinary, tower_troop_id=159_000_000)

    assert match.deck0 == _live_deck(ordinary, 0)
    assert match.tower_troop0_id == match.tower_troop1_id == 159_000_000
    # The live match must satisfy exactly the checks reset() applies.
    contract = build_semantic_supported_subset(build_card_catalog())
    contract.assert_deck(match.deck0, form_availability=match.deck0_form_availability, label="deck0")
    contract.assert_deck(match.deck1, form_availability=match.deck1_form_availability, label="deck1")


def test_world_target_matches_the_offline_grid_conversion() -> None:
    assert _world_target(_play(2, (9, 20))) == cell_to_world((9, 20))


def test_world_target_applies_the_subcell_offset() -> None:
    base_x, base_y = cell_to_world((9, 20))

    assert _world_target(_play(0, (9, 20), offset=(0.25, -0.5))) == (base_x + 250, base_y - 500)


def test_world_target_clamps_to_the_arena() -> None:
    corner = _play(0, (GRID_WIDTH - 1, GRID_HEIGHT - 1), offset=(0.5, 0.5))
    low = _play(0, (0, 0), offset=(-0.5, -0.5))

    assert _world_target(corner) == (17_999, 31_999)
    assert _world_target(low) == (0, 0)


class _RecordingNative:
    """Minimal stand-in for the probe transport."""

    def __init__(self, replies: dict[str, Any] | None = None) -> None:
        self.commands: list[str] = []
        self.replies = replies or {}

    def _request(self, command: str) -> Any:
        self.commands.append(command)
        head = command.split(" ")[0]
        reply = self.replies.get(head, {"ok": True})
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_submit_actions_sends_the_hand_slot_and_world_coordinates() -> None:
    native = _RecordingNative()
    actions = (_play(3, (4, 12)), _play(1, (9, 22)))

    submitted, rejected, _ = _submit_actions(native, actions, actor_owner=0)

    assert (submitted, rejected) == (2, 0)
    first_x, first_y = cell_to_world((4, 12))
    second_x, second_y = cell_to_world((9, 22))
    assert native.commands == [
        f"live-deploy 0 3 {first_x} {first_y}",
        f"live-deploy 0 1 {second_x} {second_y}",
    ]


def test_submit_actions_ignores_the_human_side_and_waits() -> None:
    native = _RecordingNative()
    actions = (_play(0, (4, 12), owner=1), ActionV1.wait(1, ticks=5))

    submitted, rejected, _ = _submit_actions(native, actions, actor_owner=0)

    assert (submitted, rejected) == (0, 0)
    assert native.commands == []


def test_submit_actions_activates_hero_ability() -> None:
    native = _RecordingNative()
    ability_action = ActionV1(
        owner=1,
        kind=ActionKind.ACTIVATE_ABILITY,
        source_entity=42,
        ability_id="Musketeer_hero_Ability",
    )

    submitted, rejected, diagnostics = _submit_actions(native, (ability_action,), actor_owner=1)

    assert (submitted, rejected) == (1, 0)
    assert native.commands == ["live-tap-ability"]
    assert diagnostics[0]["kind"] == "activate_ability"


def test_submit_actions_reports_a_rejected_deploy() -> None:
    native = _RecordingNative({"live-deploy": {"ok": False, "error": "no elixir"}})

    submitted, rejected, diagnostics = _submit_actions(native, (_play(0, (4, 12)),), actor_owner=0)

    assert (submitted, rejected) == (0, 1)
    assert diagnostics[0]["accepted"] is False


def test_submit_actions_reports_a_transport_failure() -> None:
    native = _RecordingNative({"live-deploy": RunnerError("socket closed")})

    submitted, rejected, diagnostics = _submit_actions(native, (_play(0, (4, 12)),), actor_owner=0)

    assert (submitted, rejected) == (0, 1)
    assert "socket closed" in diagnostics[0]["reason"]


def test_submit_actions_reports_a_play_without_a_placement() -> None:
    native = _RecordingNative()
    unstargeted = ActionV1(owner=0, kind=ActionKind.PLAY_CARD, hand_slot=0)

    submitted, rejected, diagnostics = _submit_actions(native, (unstargeted,), actor_owner=0)

    assert (submitted, rejected) == (0, 1)
    assert native.commands == []
    assert "hand slot" in diagnostics[0]["reason"]


def test_attach_live_battle_binds_towers_and_keeps_one_episode_identity() -> None:
    from native_runner.training.v4.factory import production_semantic_bundle
    from native_runner.rich_telemetry_adapter import RuntimeEffectCatalog, bind_rich_telemetry

    bundle = production_semantic_bundle()
    environment = live_agent._live_environment()
    ordinary = _ordinary()
    rich = _rich(ordinary)
    match = _live_match_config(ordinary, tower_troop_id=159_000_000)

    environment.attach_live_battle(
        match_config=match,
        ordinary=ordinary,
        rich_snapshot=bind_rich_telemetry(ordinary, rich),
        episode_id="live:1:20",
        decision_hz=4.0,
    )

    assert environment.episode_id == "live:1:20"
    assert set(environment._tower_object_keys) == {item[0] for item in TOWER_LAYOUT}
    observation = environment.observe("fair", owner=0, policy_features_only=True)
    assert observation.episode_id == "live:1:20"
    assert len(observation.towers) == len(TOWER_LAYOUT)
    assert observation.owner == 0
    # The tensorizer requires this identity; the old path never set it.
    assert observation.episode_id


def test_advance_live_battle_advances_the_episode_clock() -> None:
    from native_runner.rich_telemetry_adapter import bind_rich_telemetry

    environment = live_agent._live_environment()
    ordinary = _ordinary()
    rich = _rich(ordinary)
    match = _live_match_config(ordinary, tower_troop_id=159_000_000)
    environment.attach_live_battle(
        match_config=match,
        ordinary=ordinary,
        rich_snapshot=bind_rich_telemetry(ordinary, rich),
        episode_id="live:1:20",
        decision_hz=4.0,
    )

    later = _ordinary()
    later["tick"] = 25
    later["players"][0]["elixirRaw"] = 50_000
    later_rich = _rich(later)
    environment.advance_live_battle(
        ordinary=later, rich_snapshot=bind_rich_telemetry(later, later_rich)
    )

    observation = environment.observe("fair", owner=0, policy_features_only=True)
    assert observation.tick == 25
    assert observation.episode_id == "live:1:20"


@pytest.mark.slow
def test_live_session_drives_the_real_checkpoint() -> None:
    """End-to-end: the shipped Hog specialist produces a decision from a live frame."""

    from native_runner.rich_telemetry_adapter import bind_rich_telemetry

    loaded = load_policy_v4(CHECKPOINT, device="cpu")
    ordinary = _ordinary()
    rich = _rich(ordinary)
    environment = live_agent._live_environment()

    session = live_agent._LiveSession(
        model=loaded.model,
        environment=environment,
        actor_owner=0,
        device="cpu",
        sample=False,
        tower_troop_id=159_000_000,
        ordinary=ordinary,
        rich=rich,
        episode_id="live:3:20",
    )
    try:
        assert session.observation.episode_id == "live:3:20"
        batch = session.session.tensorizer.tensorize(session.observation, validate=True)
        assert batch.candidates.mask.shape[0] == 1
        decision = session.session.decide(session.observation)
        assert decision.decoded.actions
        native = _RecordingNative()
        _submit_actions(native, decision.decoded.actions, actor_owner=0)
        # Every decoded card play must be submittable, i.e. carry the fields
        # the old code looked for in metadata and never found.
        for action in decision.decoded.actions:
            if action.kind == ActionKind.PLAY_CARD and action.owner == 0:
                assert action.hand_slot is not None
                assert action.target_grid is not None
    finally:
        session.close()


@pytest.mark.slow
def test_live_session_supports_force_act() -> None:
    """When force_act=True is passed, the model is forced to deploy a card."""
    loaded = load_policy_v4(CHECKPOINT, device="cpu")
    ordinary = _ordinary()
    ordinary["players"][0]["elixirRaw"] = 100_000
    rich = _rich(ordinary)
    environment = live_agent._live_environment()

    session = live_agent._LiveSession(
        model=loaded.model,
        environment=environment,
        actor_owner=0,
        device="cpu",
        sample=False,
        tower_troop_id=159_000_000,
        ordinary=ordinary,
        rich=rich,
        episode_id="live:4:20",
    )
    try:
        decision = session.session.decide(session.observation, force_act=True)
        assert any(action.kind == ActionKind.PLAY_CARD for action in decision.decoded.actions)
    finally:
        session.close()


def test_run_live_agent_owner_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="actor owner must be 0, 1, or auto"):
        live_agent.run_live_agent(
            CHECKPOINT,
            actor_owner="invalid",
            artifact_dir=tmp_path,
        )


def test_send_touch_prefers_probe_and_falls_back_to_adb(monkeypatch: pytest.MonkeyPatch) -> None:
    native = _RecordingNative({"live-tap": {"ok": True}})
    adb_commands: list[list[str]] = []

    def mock_run(cmd, **kwargs):
        adb_commands.append(cmd)
        import subprocess
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("subprocess.run", mock_run)

    # When probe succeeds, adb is not called
    assert live_agent._send_touch(native, 360, 990, adb_path="adb", adb_serial="127.0.0.1:16384") is True
    assert native.commands == ["live-tap 360 990"]
    assert adb_commands == []

    # When probe fails, adb fallback is used
    failing_native = _RecordingNative({"live-tap": RunnerError("no probe")})
    assert live_agent._send_touch(failing_native, 360, 990, adb_path="adb", adb_serial="127.0.0.1:16384") is True
    assert len(adb_commands) == 1
    assert adb_commands[0] == ["adb", "-s", "127.0.0.1:16384", "shell", "input", "tap", "360", "990"]


def test_recover_client_relaunches_when_process_is_dead(monkeypatch: pytest.MonkeyPatch) -> None:
    native = _RecordingNative()
    notifications: list[dict[str, Any]] = []
    executed_commands: list[list[str]] = []

    def mock_run(cmd, **kwargs):
        executed_commands.append(cmd)
        import subprocess
        # First pidof returns empty (crashed)
        if "pidof" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="".encode(), stderr="".encode())
        return subprocess.CompletedProcess(cmd, 0, stdout="".encode(), stderr="".encode())

    monkeypatch.setattr("subprocess.run", mock_run)
    monkeypatch.setattr("time.sleep", lambda s: None)

    recovered = live_agent._recover_client(
        native,
        adb_path="adb",
        adb_serial="127.0.0.1:16384",
        port=26789,
        notify=lambda evt: notifications.append(evt),
    )

    assert recovered is True
    assert any("am" in cmd and "start" in cmd for cmd in executed_commands)
    assert any("forward" in cmd for cmd in executed_commands)
    assert any(n["event"] == "watchdog-recovering" for n in notifications)
    assert any(n["event"] == "watchdog-recovered" for n in notifications)


def test_submit_actions_suppresses_in_flight_pending() -> None:
    native = _RecordingNative({"live-deploy": {"ok": True}})
    act = ActionV1(
        owner=0,
        kind=ActionKind.PLAY_CARD,
        hand_slot=2,
        card_id=26000000,
        target_kind=TargetKind.GRID,
        target_grid=(5, 10),
    )

    # Suppressed when slot and card_id match pending
    submitted, rejected, diag = _submit_actions(
        native,
        (act,),
        actor_owner=0,
        in_flight_pending=(2, 26000000),
    )
    assert (submitted, rejected) == (0, 0)
    assert len(native.commands) == 0
    assert diag[0]["reason"] == "in_flight_pending_suppressed"

    # Permitted when slot differs or pending is None
    submitted, rejected, diag = _submit_actions(
        native,
        (act,),
        actor_owner=0,
        in_flight_pending=(1, 26000000),
    )
    assert (submitted, rejected) == (1, 0)
    assert len(native.commands) == 1


def test_warmup_policy_session_executes_dummy_forward() -> None:
    ckpt = Path(CHECKPOINT)
    if not ckpt.is_file():
        pytest.skip(f"Checkpoint {CHECKPOINT} not present on disk")

    loaded = load_policy_v4(ckpt, device="cpu")
    assert live_agent._warmup_policy_session(loaded.model, device="cpu") is True


def test_live_observation_pair_success_and_failures() -> None:
    # 1. Success
    native = _RecordingNative({
        "live-observe-atomic": {
            "ok": True,
            "ordinary": {"generation": 1, "tick": 10},
            "rich": {"generation": 1, "tick": 10},
        }
    })
    ord_env, rich_env = live_agent._live_observation_pair(native)
    assert ord_env["tick"] == 10
    assert rich_env["tick"] == 10

    # 2. Failed request (ok = False)
    failing_native = _RecordingNative({
        "live-observe-atomic": {"ok": False, "error": "not ready"}
    })
    with pytest.raises(RunnerError, match="live atomic observation failed"):
        live_agent._live_observation_pair(failing_native)

    # 3. Missing envelope
    incomplete_native = _RecordingNative({
        "live-observe-atomic": {"ok": True, "error": "graph changed"}
    })
    with pytest.raises(RunnerError, match="live atomic observation lacks ordinary/rich envelopes"):
        live_agent._live_observation_pair(incomplete_native)


def test_has_early_threat_detection() -> None:
    # Player 0 defending bottom (threat if enemy y < 17000)
    ord_safe = {
        "objects": [
            {"null": False, "owner": 1, "x": 9000, "y": 25000},  # enemy in enemy half
        ]
    }
    assert live_agent._has_early_threat(ord_safe, actor_owner=0) is False

    ord_threat = {
        "objects": [
            {"null": False, "owner": 1, "x": 3500, "y": 14000},  # enemy crossed bridge
        ]
    }
    assert live_agent._has_early_threat(ord_threat, actor_owner=0) is True

    # Towers themselves must not be counted as rushing units
    ord_tower = {
        "objects": [
            {"null": False, "owner": 1, "x": 3500, "y": 25500},
        ]
    }
    assert live_agent._has_early_threat(ord_tower, actor_owner=0) is False


def test_in_flight_tracking_and_prune() -> None:
    ckpt = Path(CHECKPOINT)
    if not ckpt.is_file():
        pytest.skip(f"Checkpoint {CHECKPOINT} not present on disk")
    loaded = load_policy_v4(ckpt, device="cpu")
    ordinary = _ordinary()
    rich = _rich(ordinary)
    live_env = live_agent._live_environment()
    session = live_agent._LiveSession(
        model=loaded.model,
        environment=live_env,
        actor_owner=0,
        device="cpu",
        sample=False,
        tower_troop_id=159_000_000,
        ordinary=ordinary,
        rich=rich,
        episode_id="live:test_in_flight",
    )

    act = _play(1, (9, 20), owner=0)
    session.record_in_flight(slot=1, card_id=26000001, cost=3, tick=100, action=act)
    assert session.active_in_flight_slots() == {1}

    # Still in-flight if ordinary has the same card at slot 1 and tick is 105
    ordinary_pending = _ordinary()
    ordinary_pending["tick"] = 105
    ordinary_pending["players"][0]["hand"][1]["cardId"] = 26000001
    active = session._prune_in_flight(ordinary_pending)
    assert len(active) == 1
    assert session.active_in_flight_slots() == {1}

    # Pruned once hand rotates to a different card
    ordinary_rotated = _ordinary()
    ordinary_rotated["tick"] = 106
    ordinary_rotated["players"][0]["hand"][1]["cardId"] = 26000002
    active = session._prune_in_flight(ordinary_rotated)
    assert len(active) == 0
    assert session.active_in_flight_slots() == set()


def test_partial_tower_mid_game_attach() -> None:
    ckpt = Path(CHECKPOINT)
    if not ckpt.is_file():
        pytest.skip(f"Checkpoint {CHECKPOINT} not present on disk")
    loaded = load_policy_v4(ckpt, device="cpu")
    ordinary = _ordinary()
    # In the native engine, destroyed tower slot has null: True
    # Enemy princess right is at (14500, 25500)
    for obj in ordinary["objects"]:
        if int(obj.get("x", -1)) == 14500 and int(obj.get("y", -1)) == 25500:
            slot = obj["slot"]
            obj.clear()
            obj.update({"slot": slot, "null": True})
    rich = _rich(ordinary)
    live_env = live_agent._live_environment()
    session = live_agent._LiveSession(
        model=loaded.model,
        environment=live_env,
        actor_owner=0,
        device="cpu",
        sample=False,
        tower_troop_id=159_000_000,
        ordinary=ordinary,
        rich=rich,
        episode_id="live:test_partial_tower",
    )
    assert session.observation is not None
    # Destroyed tower should have 0 hp
    towers = {t.entity_id: t for t in session.observation.towers}
    assert towers[5].hitpoints == 0.0


def test_soft_truncation_alignment() -> None:
    ckpt = Path(CHECKPOINT)
    if not ckpt.is_file():
        pytest.skip(f"Checkpoint {CHECKPOINT} not present on disk")
    loaded = load_policy_v4(ckpt, device="cpu")
    ordinary = _ordinary()
    rich = _rich(ordinary)

    # Simulate C++ probe returning truncated=True when object count exceeds limit
    ordinary["truncated"] = True
    ordinary["count"] = 300
    ordinary["returned"] = len(ordinary["objects"])
    rich["truncated"] = True
    rich["count"] = 300
    rich["returned"] = len(rich["objects"])

    live_env = live_agent._live_environment()
    session = live_agent._LiveSession(
        model=loaded.model,
        environment=live_env,
        actor_owner=0,
        device="cpu",
        sample=False,
        tower_troop_id=159_000_000,
        ordinary=ordinary,
        rich=rich,
        episode_id="live:test_truncated",
    )
    assert session.observation is not None

    # Advance with truncated envelopes should also succeed smoothly
    ordinary["tick"] = 15
    rich["tick"] = 15
    session.advance(ordinary, rich)
    assert session.observation.tick == 15





