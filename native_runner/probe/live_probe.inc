// Live (server-driven) battle reconnaissance for cr_replay_probe.cpp.
//
// Every battle the offline harness owns is created by the probe itself: either
// the headless GameStateManager or a replay submitted through the stock replay
// singleton.  A match the client starts from its own network path is owned by
// the game instead, so these commands inspect the client's own battle rather
// than the probe's:
//   live-status  : the stock replay singleton, the controller it holds, that
//                  controller's manager, and the manager's world
//   live-scan N  : structurally valid GameStateManager instances in the heap
//
// Include this file after engine_instance.inc so every GameState helper used
// below (read_object_field, read_battle_result, read_player_state,
// process_range_is_readable, g_game_state_tick) is already declared.

struct LiveProbeWriter {
  char *buffer = nullptr;
  std::size_t size = 0;
  std::size_t used = 0;
  bool first = true;
  bool truncated = false;

  void raw(const char *text) {
    if (buffer == nullptr || text == nullptr || used >= size) {
      truncated = true;
      return;
    }
    const int written = std::snprintf(buffer + used, size - used, "%s", text);
    if (written <= 0 || static_cast<std::size_t>(written) >= size - used) {
      truncated = true;
      return;
    }
    used += static_cast<std::size_t>(written);
  }

  void rawf(const char *format, ...) __attribute__((format(printf, 2, 3))) {
    if (buffer == nullptr || used >= size) {
      truncated = true;
      return;
    }
    va_list arguments;
    va_start(arguments, format);
    const int written = std::vsnprintf(buffer + used, size - used, format, arguments);
    va_end(arguments);
    if (written <= 0 || static_cast<std::size_t>(written) >= size - used) {
      truncated = true;
      return;
    }
    used += static_cast<std::size_t>(written);
  }

  // Comma-and-key helpers so fields can be appended positionally while the JSON
  // stays well formed.
  void separator() {
    if (!first) {
      raw(",");
    }
    first = false;
  }

  void key(const char *name) {
    separator();
    rawf("\"%s\":", name);
  }

  void number(const char *name, long long value) {
    key(name);
    rawf("%lld", value);
  }

  void boolean(const char *name, bool value) {
    key(name);
    raw(value ? "true" : "false");
  }

  void pointer(const char *name, const void *value) {
    key(name);
    rawf("\"0x%llx\"", static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(value)));
  }

  void words(const char *name, const void *base, std::size_t count) {
    if (base == nullptr || count == 0 || !process_range_is_readable(base, count * sizeof(std::uint64_t))) {
      return;
    }
    key(name);
    raw("[");
    for (std::size_t index = 0; index < count; ++index) {
      std::uint64_t value = 0;
      std::memcpy(&value, static_cast<const std::uint8_t *>(base) + index * sizeof(value), sizeof(value));
      rawf("%s\"0x%llx\"", index == 0 ? "" : ",", static_cast<unsigned long long>(value));
    }
    raw("]");
  }
};

constexpr std::size_t kLiveScanMaximumBytes = 1024ULL * 1024ULL * 1024ULL;
constexpr std::size_t kLiveScanMaximumCandidates = 32;
constexpr std::size_t kLiveManagerReadableBytes = 0x230;
constexpr std::size_t kLiveControllerVtableSlots = 48;
constexpr std::size_t kLiveControllerBytes = 0x340;

bool live_probe_manager_graph_readable(void *manager);

std::int32_t live_probe_manager_tick(void *manager) {
  if (manager == nullptr || g_game_state_tick == nullptr ||
      !process_range_is_readable(manager, kLiveManagerReadableBytes) ||
      !live_probe_manager_graph_readable(manager)) {
    return -1;
  }
  return g_game_state_tick(manager);
}

// Lifetimes of the stock controller/manager/world are owned by the game thread.
// Every world read below is optional-serverlike recon and can race battle
// teardown, so each pointer is gated by process_range_is_readable before it is
// dereferenced.  Nothing here may assume the offline harness's stable objects.
bool live_probe_read_battle_result(void *world, BattleResultView *result_out) {
  if (world == nullptr || result_out == nullptr) {
    return false;
  }
  constexpr std::size_t kWorldBytes = 0x1e8;
  if (!process_range_is_readable(world, kWorldBytes)) {
    return false;
  }
  BattleResultView value;
  value.finalized = read_object_field<std::uint8_t>(world, 0x1e0) != 0;
  value.world_result_raw = read_object_field<std::int32_t>(world, 0x1b8);
  // Crown score is derived from the native tower graph (read_native_crowns),
  // which walks unguarded object_manager/object pointers that can be freed
  // during teardown.  Recon should never crash over crowns, so they are left
  // unread here; the model reads tower hitpoints directly from entities.
  value.crowns_raw[0] = -1;
  value.crowns_raw[1] = -1;
  value.readable = true;
  *result_out = value;
  return true;
}

bool live_probe_read_player_state(void *world, std::int32_t owner, PlayerStateView *result) {
  if (world == nullptr || result == nullptr || owner < 0 || owner >= kPlayerCount) {
    return false;
  }
  constexpr std::size_t kWorldBytes = 0x1e8;
  if (!process_range_is_readable(world, kWorldBytes)) {
    return false;
  }
  result->player = read_object_field<void *>(world, 0xe0 + owner * sizeof(void *));
  result->identity = read_object_field<void *>(world, 0x30 + owner * sizeof(void *));
  result->deck = read_object_field<void *>(world, 0x88 + owner * sizeof(void *));
  constexpr std::size_t kPlayerBytes = 0x2f8 + sizeof(std::int32_t);
  if (result->player == nullptr || result->identity == nullptr || result->deck == nullptr ||
      !process_range_is_readable(result->player, kPlayerBytes) ||
      !process_range_is_readable(result->identity, 0x08) ||
      !process_range_is_readable(result->deck, 0x28)) {
    return false;
  }
  result->account_id_high = read_object_field<std::uint32_t>(result->identity, 0x00);
  result->account_id_low = read_object_field<std::uint32_t>(result->identity, 0x04);
  result->elixir_raw = read_object_field<std::int32_t>(result->player, 0x2f8);
  result->crowns_raw = 0;
  return read_native_vector(result->deck, 0x20, kMaxDeckCards, &result->deck_slots) &&
         read_native_vector(result->player, 0x220, kMaxHandCards, &result->hand) &&
         read_native_vector(result->player, 0x230, kMaxCycleCards, &result->cycle);
}

// Persistent live (server-driven) battle state and inject mailbox.  The
// control thread records the live manager from the stock replay singleton on
// demand and queues deploy requests here; the game thread drains the mailbox
// from game_state_step_hook (see live_probe_flush_pending below) so injects
// execute on the engine's own step, avoiding cross-thread queue mutation.
std::atomic<void *> g_live_manager{nullptr};
std::atomic<std::int64_t> g_last_live_close_ns{0};
// Monotonic stand-in for the offline generation counter.  Rich observation
// only needs a stable non-zero identity per live battle; the game thread does
// not publish an epoch for server-driven matches.
std::atomic<std::uint64_t> g_live_generation{1};
char g_live_pending_json[kMaxPendingCommandBytes] = {};
std::uint64_t g_live_pending_sequence = 0;
std::uint64_t g_live_processed_sequence = 0;
pthread_mutex_t g_live_mailbox_mutex = PTHREAD_MUTEX_INITIALIZER;

void *live_probe_current_controller();

void *live_probe_current_manager() {
  const std::int64_t last_close = g_last_live_close_ns.load(std::memory_order_acquire);
  if (last_close != 0 && (process_monotonic_now_ns() - last_close) < 2000000000LL) {
    return nullptr;
  }
  if (g_replay_manager_getter == nullptr) {
    return nullptr;
  }
  void *const singleton = g_replay_manager_getter();
  if (singleton == nullptr || !process_range_is_readable(singleton, 0x40)) {
    return nullptr;
  }
  void *controller = nullptr;
  std::memcpy(&controller, static_cast<const std::uint8_t *>(singleton) + 0x20, sizeof(controller));
  if (controller == nullptr || !process_range_is_readable(controller, kLiveControllerBytes)) {
    return nullptr;
  }
  // Validate that controller is an active ReplayBattleController (state 8).
  // When a battle ends, state transitions away from 8 (e.g. teardown or lobby object).
  const std::int32_t state = read_object_field<std::int32_t>(controller, kReplayBattleControllerStateOffset);
  if (state != 8) {
    return nullptr;
  }
  void *manager = nullptr;
  std::memcpy(&manager, static_cast<const std::uint8_t *>(controller) + 0x90, sizeof(manager));
  if (manager == nullptr || !live_probe_manager_graph_readable(manager)) {
    return nullptr;
  }
  return manager;
}

// The rich observation builder dereferences manager->world->kingTower->
// objectManager->objects without its own guard, which is safe only for the
// probe-owned harness objects.  For a server-driven battle the same graph can
// be torn down by the game thread at any moment, so prove every level is
// readable before handing it to the builder.  A false result means "try again
// later", never a crash.
bool live_probe_manager_graph_readable(void *manager) {
  if (manager == nullptr || !process_range_is_readable(manager, kLiveManagerReadableBytes)) {
    return false;
  }
  void *const world = read_object_field<void *>(manager, 0xa8);
  if (world == nullptr || !process_range_is_readable(world, 0x1e8)) {
    return false;
  }
  void *const king_tower = read_object_field<void *>(world, 0xe0);
  if (king_tower == nullptr || !process_range_is_readable(king_tower, 0x18)) {
    return false;
  }
  void *const object_manager = read_object_field<void *>(king_tower, 0x10);
  if (object_manager == nullptr || !process_range_is_readable(object_manager, 0x18)) {
    return false;
  }
  void **const objects = read_object_field<void **>(object_manager, 0x08);
  const std::int32_t capacity = read_object_field<std::int32_t>(object_manager, 0x10);
  const std::int32_t count = read_object_field<std::int32_t>(object_manager, 0x14);
  if (count < 0 || capacity < count || capacity > 100000) {
    return false;
  }
  return count == 0 || (objects != nullptr &&
                        process_range_is_readable(objects, static_cast<std::size_t>(count) * sizeof(void *)));
}

// Runs on the game thread (from game_state_step_hook) for every native logic
// tick while a live battle manager is recorded.  Drains one queued deploy and
// appends it to the engine command queue through the stock command path.
void live_probe_flush_pending(void *manager) {
  if (manager == nullptr) {
    manager = live_probe_current_manager();
  }
  if (manager == nullptr) {
    return;
  }
  char command_json[kMaxPendingCommandBytes] = {};
  std::uint64_t sequence = 0;
  pthread_mutex_lock(&g_live_mailbox_mutex);
  if (g_live_pending_sequence != g_live_processed_sequence) {
    sequence = g_live_pending_sequence;
    std::memcpy(command_json, g_live_pending_json, sizeof(command_json));
    command_json[sizeof(command_json) - 1] = '\0';
  }
  pthread_mutex_unlock(&g_live_mailbox_mutex);
  if (sequence == 0 || command_json[0] == '\0') {
    return;
  }
  bool succeeded = false;
  void *logic_command = nullptr;
  if (parse_logic_command(manager, command_json, &logic_command) && logic_command != nullptr) {
    void *controller = nullptr;
    if (g_get_active_battle_controller != nullptr) {
      controller = g_get_active_battle_controller();
    }
    if (controller == nullptr) {
      controller = live_probe_current_controller();
    }
    if (controller != nullptr && g_finish_player_command != nullptr) {
      g_finish_player_command(controller, &logic_command, 1);
      succeeded = true;
      LOGI("live deploy flushed via finish_player_command controller=%p manager=%p seq=%llu",
           controller, manager, static_cast<unsigned long long>(sequence));
    } else if (g_logic_command_append != nullptr) {
      void *command_queue = nullptr;
      std::memcpy(&command_queue, static_cast<std::uint8_t *>(manager) + 0x38, sizeof(command_queue));
      if (command_queue != nullptr) {
        g_logic_command_append(command_queue, &logic_command);
        succeeded = true;
        LOGI("live deploy fallback dispatched via logic_command_append queue=%p manager=%p seq=%llu",
             command_queue, manager, static_cast<unsigned long long>(sequence));
      }
    }
  } else {
    LOGE("live deploy failed to parse logic command: %s", command_json);
  }
  pthread_mutex_lock(&g_live_mailbox_mutex);
  if (g_live_processed_sequence < sequence) {
    g_live_processed_sequence = sequence;
    if (succeeded) {
      g_live_pending_json[0] = '\0';
    }
  }
  pthread_mutex_unlock(&g_live_mailbox_mutex);
  LOGI("live deploy flushed manager=%p seq=%llu success=%d", manager,
       static_cast<unsigned long long>(sequence), succeeded ? 1 : 0);
}

// Directly resolves and dispatches card deployment via BattleController::finishPlayerCommand.
bool live_probe_resolve_and_queue(void *manager, std::int32_t owner, std::int32_t hand_index, std::int32_t world_x,
                                  std::int32_t world_y) {
  if (manager == nullptr || !process_range_is_readable(manager, kLiveManagerReadableBytes)) {
    return false;
  }
  void *const world = read_object_field<void *>(manager, 0xa8);
  constexpr std::size_t kWorldBytes = 0x1e8;
  if (world == nullptr || !process_range_is_readable(world, kWorldBytes)) {
    return false;
  }
  DeployRequest request;
  if (!resolve_hand_deploy_request(manager, owner, hand_index, world_x, world_y, &request)) {
    LOGE("live deploy could not resolve hand deploy owner=%d hand=%d world=(%d,%d)", owner, hand_index, world_x, world_y);
    return false;
  }
  const std::int32_t current_tick = live_probe_manager_tick(manager);
  if (current_tick < 0 || current_tick > INT32_MAX - 20) {
    LOGE("live deploy invalid tick %d", current_tick);
    return false;
  }
  char command_json[512] = {};
  if (!format_deploy_command(request, current_tick, command_json, sizeof(command_json))) {
    LOGE("live deploy could not format deploy command");
    return false;
  }
  void *logic_command = nullptr;
  if (!parse_logic_command(manager, command_json, &logic_command) || logic_command == nullptr) {
    LOGE("live deploy failed to parse logic command: %s", command_json);
    return false;
  }
  void *controller = nullptr;
  if (g_get_active_battle_controller != nullptr) {
    controller = g_get_active_battle_controller();
  }
  if (controller == nullptr) {
    controller = live_probe_current_controller();
  }
  if (controller != nullptr && g_finish_player_command != nullptr) {
    g_finish_player_command(controller, &logic_command, 1);
    LOGI("live deploy executed directly via finish_player_command controller=%p manager=%p owner=%d hand=%d world=(%d,%d) tick=%d",
         controller, manager, owner, hand_index, world_x, world_y, current_tick);
    return true;
  }
  if (g_logic_command_append != nullptr) {
    void *command_queue = nullptr;
    std::memcpy(&command_queue, static_cast<std::uint8_t *>(manager) + 0x38, sizeof(command_queue));
    if (command_queue != nullptr) {
      g_logic_command_append(command_queue, &logic_command);
      LOGI("live deploy fallback dispatched via logic_command_append queue=%p manager=%p owner=%d",
           command_queue, manager, owner);
      return true;
    }
  }
  LOGE("live deploy failed: no controller or command queue available");
  return false;
}
bool live_probe_manager_candidate(void *manager, BattleResultView *result_out, std::int32_t *tick_out) {
  if (manager == nullptr || reinterpret_cast<std::uintptr_t>(manager) % sizeof(void *) != 0 ||
      !process_range_is_readable(manager, kLiveManagerReadableBytes)) {
    return false;
  }
  void *const command_queue = read_object_field<void *>(manager, 0x38);
  if (command_queue == nullptr || !process_range_is_readable(command_queue, 0x40)) {
    return false;
  }
  void *const world = read_object_field<void *>(manager, 0xa8);
  BattleResultView result;
  if (world == nullptr || !live_probe_read_battle_result(world, &result)) {
    return false;
  }
  const std::int32_t tick = live_probe_manager_tick(manager);
  if (tick < 0) {
    return false;
  }
  if (result_out != nullptr) {
    *result_out = result;
  }
  if (tick_out != nullptr) {
    *tick_out = tick;
  }
  return true;
}

// The stock UI resolves the live ReplayBattleController through c85c28: it
// fetches the replay singleton, checks the controller state, then reads the
// controller pointer at singleton+0x20.  That controller is the client's own
// polymorphic surface, and the method that finishes a player command (and so
// reaches the network) is one of its vtable slots.  The vtable is filled in at
// runtime, so the slot addresses can only be read from a live process; this
// command exports them so the host can identify the real outbound entry point
// from evidence instead of from a possibly stale source comment.
void *live_probe_current_controller() {
  const std::int64_t last_close = g_last_live_close_ns.load(std::memory_order_acquire);
  if (last_close != 0 && (process_monotonic_now_ns() - last_close) < 2000000000LL) {
    return nullptr;
  }
  if (g_replay_manager_getter == nullptr) {
    return nullptr;
  }
  void *const singleton = g_replay_manager_getter();
  if (singleton == nullptr || !process_range_is_readable(singleton, 0x40)) {
    return nullptr;
  }
  void *controller = nullptr;
  std::memcpy(&controller, static_cast<const std::uint8_t *>(singleton) + 0x20, sizeof(controller));
  if (controller == nullptr || !process_range_is_readable(controller, kLiveControllerBytes)) {
    return nullptr;
  }
  const std::int32_t state = read_object_field<std::int32_t>(controller, kReplayBattleControllerStateOffset);
  if (state != 8) {
    return nullptr;
  }
  return controller;
}

bool format_live_probe_outbound(char *response, std::size_t response_size) {
  if (response == nullptr || response_size < 4096) {
    return false;
  }
  void *const controller = live_probe_current_controller();
  LiveProbeWriter writer;
  writer.buffer = response;
  writer.size = response_size;
  writer.raw("{\"ok\":true,\"mode\":\"live-outbound\",");
  writer.pointer("replaySingleton", g_replay_manager_getter == nullptr ? nullptr : g_replay_manager_getter());
  writer.pointer("controller", controller);
  if (controller == nullptr) {
    writer.number("controllerState", -1);
    writer.boolean("vtableReadable", false);
    writer.key("vtable");
    writer.raw("[]");
    writer.raw("}");
    return true;
  }
  writer.number("controllerState", read_object_field<std::int32_t>(controller, kReplayBattleControllerStateOffset));
  void *const vtable = read_object_field<void **>(controller, 0x00);
  const bool readable =
      vtable != nullptr && process_range_is_readable(vtable, kLiveControllerVtableSlots * sizeof(void *));
  writer.boolean("vtableReadable", readable);
  writer.pointer("vtable", vtable);
  writer.key("vtable");
  writer.raw("[");
  if (readable) {
    for (std::size_t slot = 0; slot < kLiveControllerVtableSlots; ++slot) {
      void *entry = nullptr;
      std::memcpy(&entry, static_cast<const std::uint8_t *>(vtable) + slot * sizeof(void *), sizeof(entry));
      if (entry == nullptr) {
        continue;
      }
      // Report the module-relative offset so the host can reason about it
      // without knowing the ASLR base.
      const auto address = reinterpret_cast<std::uintptr_t>(entry);
      const bool in_module = address >= g_libg_base && address - g_libg_base < 0x2000000ULL;
      writer.rawf("%s{\"slot\":%zu,\"offset\":%lld}", slot == 0 ? "" : ",", slot,
                  in_module ? static_cast<long long>(address - g_libg_base) : -1LL);
    }
  }
  writer.raw("]");
  writer.raw("}");
  return !writer.truncated;
}

void append_live_probe_manager(LiveProbeWriter &writer, const char *label, void *manager, bool include_words) {
  writer.key(label);
  writer.raw("{");
  const bool saved_first = writer.first;
  writer.first = true;

  const bool manager_readable = manager != nullptr && process_range_is_readable(manager, kLiveManagerReadableBytes);
  void *const world = manager_readable ? read_object_field<void *>(manager, 0xa8) : nullptr;
  void *const command_queue = manager_readable ? read_object_field<void *>(manager, 0x38) : nullptr;
  BattleResultView result;
  const bool result_readable = world != nullptr && live_probe_read_battle_result(world, &result);
  const std::int32_t tick = manager_readable ? live_probe_manager_tick(manager) : -1;
  writer.pointer("manager", manager);
  writer.pointer("world", world);
  writer.pointer("commandQueue", command_queue);
  writer.number("tick", tick);
  writer.boolean("resultReadable", result_readable);
  writer.boolean("finalized", result_readable && result.finalized);
  writer.number("worldResult", result_readable ? result.world_result_raw : -1);
  writer.key("crowns");
  writer.rawf("[%d,%d]", result_readable ? result.crowns_raw[0] : -1, result_readable ? result.crowns_raw[1] : -1);
  for (std::int32_t owner = 0; owner < kPlayerCount; ++owner) {
    PlayerStateView player;
    char label_buffer[32] = {};
    std::snprintf(label_buffer, sizeof(label_buffer), "player%d", owner);
    writer.key(label_buffer);
    writer.raw("{");
    const bool saved_player_first = writer.first;
    writer.first = true;
    const bool readable = world != nullptr && live_probe_read_player_state(world, owner, &player);
    writer.boolean("readable", readable);
    if (readable) {
      writer.pointer("object", player.player);
      writer.number("elixirRaw", player.elixir_raw);
      writer.number("handCount", player.hand.count);
      writer.number("deckCount", player.deck_slots.count);
      writer.number("cycleCount", player.cycle.count);
    }
    writer.first = saved_player_first;
    writer.raw("}");
  }
  if (include_words) {
    writer.words("managerWords", manager, 24);
    writer.words("worldWords", world, 24);
  }

  writer.first = saved_first;
  writer.raw("}");
}

bool format_live_probe_status(char *response, std::size_t response_size) {
  if (response == nullptr || response_size < 4096 || g_replay_manager_getter == nullptr) {
    return false;
  }
  void *const singleton = g_replay_manager_getter();
  void *controller = nullptr;
  if (singleton != nullptr && process_range_is_readable(singleton, 0x40)) {
    std::memcpy(&controller, static_cast<const std::uint8_t *>(singleton) + 0x20, sizeof(controller));
  }
  void *const manager = live_probe_current_manager();

  LiveProbeWriter writer;
  writer.buffer = response;
  writer.size = response_size;
  writer.raw("{\"ok\":true,\"mode\":\"live-probe\",");
  writer.number("runnerMode",
                g_runner_mode.load(std::memory_order_acquire) == static_cast<std::int32_t>(RunnerMode::NativeRender)
                    ? 1
                    : 0);
  writer.boolean("offlineControlSession", g_offline_control_session_active.load(std::memory_order_acquire));
  writer.pointer("replaySingleton", singleton);
  writer.pointer("controller", controller);
  if (controller != nullptr && process_range_is_readable(controller, 0x190)) {
    writer.number("controllerState", read_object_field<std::int32_t>(controller, kReplayBattleControllerStateOffset));
    writer.number("controllerSuspended",
                  read_object_field<std::uint8_t>(controller, kReplayBattleControllerSuspendedOffset));
    float speed = 0.0f;
    std::memcpy(&speed, static_cast<const std::uint8_t *>(controller) + kReplayBattleControllerSpeedOffset,
                sizeof(speed));
    writer.pointer("controllerManagerSlot", manager);
    writer.key("controllerSpeed");
    writer.rawf("%.4f", static_cast<double>(speed));
  }
  append_live_probe_manager(writer, "live", manager, true);
  writer.words("singletonWords", singleton, 12);
  writer.words("controllerWords", controller, 24);
  writer.boolean("truncated", writer.truncated);
  writer.raw("}");
  return !writer.truncated;
}

// The V4 model consumes two envelopes captured at one native state: the
// ordinary observation (players/hand/entities) and the rich telemetry
// (combat/phase/runtime events).  Offline this is observe-atomic.  For a
// server-driven battle the same pair is built here from the live manager with
// the graph readability pre-check, so a torn-down battle fails closed.
bool live_probe_build_atomic_json(char *response, std::size_t response_size) {
  constexpr const char *kSchema = "native-observation-capture.v1";
  if (response == nullptr || response_size < 4096) {
    return false;
  }
  response[0] = '\0';
  void *const manager = live_probe_current_manager();
  if (manager == nullptr || !live_probe_manager_graph_readable(manager)) {
    std::snprintf(response, response_size,
                  "{\"ok\":false,\"schema\":\"%s\",\"error\":\"live battle graph is not safely readable\"}", kSchema);
    return false;
  }
  const std::uint64_t generation = g_live_generation.load(std::memory_order_acquire);
  if (g_combat_game_manager.load(std::memory_order_acquire) != manager ||
      !g_combat_epoch_active.load(std::memory_order_acquire)) {
    begin_runtime_telemetry_epoch(manager, generation, generation);
  }
  const std::int32_t tick = live_probe_manager_tick(manager);
  std::size_t used = 0;
  JsonWriter json(response, response_size, used);
  if (!json.append("{\"ok\":true,\"schema\":\"%s\",\"atomic\":true,\"live\":true,"
                   "\"generation\":%llu,\"stateEpoch\":%llu,\"tick\":%d,\"ordinary\":",
                   kSchema, static_cast<unsigned long long>(generation),
                   static_cast<unsigned long long>(generation), tick)) {
    return false;
  }
  bool ordinary_complete = false;
  if (!build_observation_json(manager, generation, response + used, response_size - used, &ordinary_complete)) {
    std::snprintf(response, response_size,
                  "{\"ok\":false,\"schema\":\"%s\",\"error\":\"failed to encode live ordinary observation\"}", kSchema);
    return false;
  }
  used += std::strlen(response + used);
  if (!live_probe_manager_graph_readable(manager) || !json.append(",\"rich\":")) {
    std::snprintf(response, response_size,
                  "{\"ok\":false,\"schema\":\"%s\",\"error\":\"live battle identity changed during capture\"}", kSchema);
    return false;
  }
  bool rich_complete = false;
  if (!build_rich_observation_json(manager, generation, response + used, response_size - used, &rich_complete)) {
    std::snprintf(response, response_size,
                  "{\"ok\":false,\"schema\":\"%s\",\"error\":\"failed to encode live rich observation\"}", kSchema);
    return false;
  }
  used += std::strlen(response + used);
  if (!live_probe_manager_graph_readable(manager) || !json.append("}")) {
    std::snprintf(response, response_size,
                  "{\"ok\":false,\"schema\":\"%s\",\"error\":\"live battle identity changed during capture\"}", kSchema);
    return false;
  }
  return true;
}

bool format_live_probe_scan(char *response, std::size_t response_size, std::size_t maximum_candidates) {
  if (response == nullptr || response_size < 4096) {
    return false;
  }
  const std::size_t candidate_cap =
      maximum_candidates == 0
          ? kLiveScanMaximumCandidates
          : (maximum_candidates < kLiveScanMaximumCandidates ? maximum_candidates : kLiveScanMaximumCandidates);
  LiveProbeWriter writer;
  writer.buffer = response;
  writer.size = response_size;
  writer.raw("{\"ok\":true,\"mode\":\"live-scan\"");
  writer.key("candidates");
  writer.raw("[");

  std::size_t scanned_bytes = 0;
  std::size_t scanned_regions = 0;
  std::size_t candidates = 0;
  bool first_candidate = true;
  FILE *maps = std::fopen("/proc/self/maps", "r");
  if (maps != nullptr) {
    char line[4096] = {};
    while (std::fgets(line, sizeof(line), maps) != nullptr) {
      unsigned long long raw_begin = 0;
      unsigned long long raw_end = 0;
      char permissions[5] = {};
      if (std::sscanf(line, "%llx-%llx %4s", &raw_begin, &raw_end, permissions) != 3) {
        continue;
      }
      // Only writable, private, non-file-backed mappings can hold a heap manager.
      if (permissions[0] != 'r' || permissions[1] != 'w' || permissions[3] != 'p') {
        continue;
      }
      if (std::strchr(line, '/') != nullptr) {
        continue;
      }
      const char *const bracket = std::strchr(line, '[');
      if (bracket != nullptr && std::strstr(bracket, "[heap]") == nullptr) {
        continue;
      }
      if (raw_begin >= raw_end || raw_begin % sizeof(void *) != 0) {
        continue;
      }
      const std::uintptr_t begin = static_cast<std::uintptr_t>(raw_begin);
      const std::uintptr_t end = static_cast<std::uintptr_t>(raw_end);
      if (end - begin > kLiveScanMaximumBytes || scanned_bytes >= kLiveScanMaximumBytes) {
        continue;
      }
      ++scanned_regions;
      // /proc/self/maps already proved this range is mapped, so a structural
      // check per aligned word is safe here.
      for (std::uintptr_t cursor = begin; cursor + sizeof(void *) <= end; cursor += sizeof(void *)) {
        scanned_bytes += sizeof(void *);
        void *candidate = nullptr;
        std::memcpy(&candidate, reinterpret_cast<const void *>(cursor), sizeof(candidate));
        BattleResultView result;
        std::int32_t tick = -1;
        if (!live_probe_manager_candidate(candidate, &result, &tick)) {
          continue;
        }
        if (!first_candidate) {
          writer.raw(",");
        }
        first_candidate = false;
        writer.rawf("{\"manager\":\"0x%llx\",\"slot\":\"0x%llx\",\"tick\":%d,\"finalized\":%s,\"crowns\":[%d,%d]}",
                    static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(candidate)),
                    static_cast<unsigned long long>(cursor), tick, result.finalized ? "true" : "false",
                    result.crowns_raw[0], result.crowns_raw[1]);
        if (++candidates >= candidate_cap) {
          break;
        }
      }
      if (candidates >= candidate_cap) {
        break;
      }
    }
    std::fclose(maps);
  }

  writer.raw("]");
  writer.number("candidateCount", static_cast<long long>(candidates));
  writer.number("scannedRegions", static_cast<long long>(scanned_regions));
  writer.number("scannedBytes", static_cast<long long>(scanned_bytes));
  writer.pointer("singleton", g_replay_manager_getter == nullptr ? nullptr : g_replay_manager_getter());
  writer.boolean("truncated", writer.truncated);
  writer.raw("}");
  return true;
}
