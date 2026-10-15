#include "native_touch_interceptor.h"

#include "native_touch_layout.h"

#include <android/log.h>
#include <jni.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <pthread.h>

namespace {

constexpr const char *kLogTag = "CRNativeTouch";
constexpr std::uintptr_t kGameAppTouchOffset = 0x013d0be0;
constexpr std::uint8_t kGameAppTouchPrologue[16] = {
    0xfd, 0x7b, 0xbc, 0xa9, 0xf8, 0x5f, 0x01, 0xa9, 0xf6, 0x57, 0x02, 0xa9, 0xf4, 0x4f, 0x03, 0xa9,
};

constexpr std::int32_t kActionDown = 0;
constexpr std::int32_t kActionUp = 1;
constexpr std::int32_t kActionMove = 2;
constexpr std::int32_t kActionCancel = 3;

using GameAppTouch = void (*)(JNIEnv *, jclass, jint, jint, jint, jint);

enum class CaptureKind : std::int32_t {
  None = 0,
  Hand = 1,
  Arena = 2,
};

struct GestureCapture {
  CaptureKind kind = CaptureKind::None;
  std::int32_t pointer_id = -1;
  std::int32_t owner = -1;
  std::int32_t hand_index = -1;
};

NativeTouchBindings g_bindings;
GameAppTouch g_original_game_app_touch = nullptr;
pthread_mutex_t g_touch_mutex = PTHREAD_MUTEX_INITIALIZER;
GestureCapture g_capture;
NativeTouchSelectionSnapshot g_selection;
std::atomic<bool> g_installed{false};
JavaVM *g_touch_java_vm = nullptr;
jclass g_cached_game_app_class = nullptr;
std::atomic<std::int32_t> g_touch_layout_width{720};
std::atomic<std::int32_t> g_touch_layout_height{1280};
std::atomic<std::int32_t> g_touch_controlled_owner{-1};

#define TOUCH_LOGI(...) __android_log_print(ANDROID_LOG_INFO, kLogTag, __VA_ARGS__)
#define TOUCH_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, kLogTag, __VA_ARGS__)

void detect_screen_resolution_once(JNIEnv *env) {
  static std::atomic<bool> detected{false};
  if (detected.load(std::memory_order_acquire) || env == nullptr) {
    return;
  }
  detected.store(true, std::memory_order_release);

  jclass resources_class = env->FindClass("android/content/res/Resources");
  if (resources_class == nullptr) {
    if (env->ExceptionCheck()) env->ExceptionClear();
    return;
  }
  jmethodID get_system = env->GetStaticMethodID(resources_class, "getSystem", "()Landroid/content/res/Resources;");
  if (get_system == nullptr) {
    if (env->ExceptionCheck()) env->ExceptionClear();
    return;
  }
  jobject resources = env->CallStaticObjectMethod(resources_class, get_system);
  if (resources == nullptr) {
    if (env->ExceptionCheck()) env->ExceptionClear();
    return;
  }
  jmethodID get_metrics = env->GetMethodID(resources_class, "getDisplayMetrics", "()Landroid/util/DisplayMetrics;");
  if (get_metrics == nullptr) {
    if (env->ExceptionCheck()) env->ExceptionClear();
    return;
  }
  jobject metrics = env->CallObjectMethod(resources, get_metrics);
  if (metrics == nullptr) {
    if (env->ExceptionCheck()) env->ExceptionClear();
    return;
  }
  jclass metrics_class = env->FindClass("android/util/DisplayMetrics");
  if (metrics_class == nullptr) {
    if (env->ExceptionCheck()) env->ExceptionClear();
    return;
  }
  jfieldID width_field = env->GetFieldID(metrics_class, "widthPixels", "I");
  jfieldID height_field = env->GetFieldID(metrics_class, "heightPixels", "I");
  if (width_field != nullptr && height_field != nullptr) {
    const jint w = env->GetIntField(metrics, width_field);
    const jint h = env->GetIntField(metrics, height_field);
    if (w > 0 && h > 0) {
      g_touch_layout_width.store(w, std::memory_order_release);
      g_touch_layout_height.store(h, std::memory_order_release);
      TOUCH_LOGI("auto-detected display resolution: %dx%d", w, h);
    }
  }
  if (env->ExceptionCheck()) env->ExceptionClear();
}

bool native_render_ready() {
  return g_bindings.is_native_render_ready != nullptr && g_bindings.is_native_render_ready(g_bindings.context);
}

std::uint64_t native_render_generation() {
  return g_bindings.native_render_generation == nullptr ? 0 : g_bindings.native_render_generation(g_bindings.context);
}

void clear_capture_locked() { g_capture = {}; }

void clear_selection_locked() {
  if (g_selection.owner != -1 || g_selection.hand_index != -1) {
    g_selection.owner = -1;
    g_selection.hand_index = -1;
    ++g_selection.revision;
  }
}

void forward_original(JNIEnv *env, jclass owner_class, jint action, jint x, jint y, jint pointer_id) {
  GameAppTouch original = g_original_game_app_touch;
  if (original != nullptr) {
    original(env, owner_class, action, x, y, pointer_id);
  }
}

extern "C" void native_touch_game_app_hook(JNIEnv *env, jclass owner_class, jint action, jint x, jint y,
                                           jint pointer_id) {
  if (owner_class != nullptr && g_cached_game_app_class == nullptr && env != nullptr) {
    g_cached_game_app_class = static_cast<jclass>(env->NewGlobalRef(owner_class));
  }
  detect_screen_resolution_once(env);
  if (x > g_touch_layout_width.load(std::memory_order_relaxed) ||
      y > g_touch_layout_height.load(std::memory_order_relaxed)) {
    if (x > 720 || y > 1280) {
      g_touch_layout_width.store(1080, std::memory_order_release);
      g_touch_layout_height.store(1920, std::memory_order_release);
      TOUCH_LOGI("promoted touch layout to 1080x1920 based on coordinate (%d, %d)", x, y);
    }
  }
  const cr_native_touch::Layout layout = cr_native_touch::compute_touch_layout(
      g_touch_layout_width.load(std::memory_order_acquire),
      g_touch_layout_height.load(std::memory_order_acquire));
  const std::int32_t controlled = g_touch_controlled_owner.load(std::memory_order_relaxed);

  const bool ready = native_render_ready();
  const std::uint64_t generation = ready ? native_render_generation() : 0;
  bool consume = false;
  bool selected = false;
  bool submit = false;
  std::int32_t submit_owner = -1;
  std::int32_t submit_hand_index = -1;
  std::int32_t selected_owner = -1;
  std::int32_t selected_hand_index = -1;
  cr_native_touch::WorldPoint submit_point;

  pthread_mutex_lock(&g_touch_mutex);
  bool invalidated_captured_pointer = false;
  if (g_selection.generation != generation) {
    invalidated_captured_pointer = g_capture.kind != CaptureKind::None && g_capture.pointer_id == pointer_id;
    clear_capture_locked();
    clear_selection_locked();
    g_selection.generation = generation;
  }
  const bool captured_pointer = g_capture.kind != CaptureKind::None && g_capture.pointer_id == pointer_id;

  if (invalidated_captured_pointer) {
    // The matching DOWN belonged to the previous manager generation.
    consume = true;
  } else if (!ready) {
    // If native-render disappeared during a captured gesture, consume its
    // tail so the stock UI never receives an orphan UP/CANCEL.  Otherwise
    // clear stale selection and leave the event completely untouched.
    if (captured_pointer) {
      consume = true;
      if (action == kActionUp || action == kActionCancel) {
        clear_capture_locked();
      }
    } else {
      clear_capture_locked();
    }
    clear_selection_locked();
  } else if (action == kActionDown && g_capture.kind == CaptureKind::None) {
    const cr_native_touch::HandHit hand = cr_native_touch::hand_at_pixel(layout, x, y);
    if (hand.hit && (controlled < 0 || hand.owner == controlled)) {
      g_capture = {
          CaptureKind::Hand,
          pointer_id,
          hand.owner,
          hand.hand_index,
      };
      consume = true;
    } else if (g_selection.owner >= 0 && g_selection.hand_index >= 0 &&
               cr_native_touch::arena_contains_pixel(layout, x, y)) {
      g_capture = {
          CaptureKind::Arena,
          pointer_id,
          g_selection.owner,
          g_selection.hand_index,
      };
      consume = true;
    }
  } else if (captured_pointer) {
    consume = true;
    if (action == kActionUp) {
      if (g_capture.kind == CaptureKind::Hand) {
        const cr_native_touch::HandHit hand =
            cr_native_touch::hand_at_pixel(layout, x, y);
        if (hand.hit && hand.owner == g_capture.owner && hand.hand_index == g_capture.hand_index) {
          if (g_selection.owner == hand.owner && g_selection.hand_index == hand.hand_index) {
            // Tapped again on the currently selected card -> deselect (toggle off)
            clear_selection_locked();
            TOUCH_LOGI("deselected native-render hand owner=%d slot=%d", hand.owner, hand.hand_index);
          } else {
            // Tap to select
            g_selection.owner = hand.owner;
            g_selection.hand_index = hand.hand_index;
            ++g_selection.revision;
            selected = true;
            selected_owner = hand.owner;
            selected_hand_index = hand.hand_index;
          }
        } else if (cr_native_touch::arena_contains_pixel(layout, x, y)) {
          // DRAG AND DROP: Pressed down on hand card, dragged onto arena, and released!
          submit = true;
          submit_owner = g_capture.owner;
          submit_hand_index = g_capture.hand_index;
          submit_point = cr_native_touch::arena_pixel_to_world(layout, x, y);
          clear_selection_locked();
        }
      } else if (g_capture.kind == CaptureKind::Arena && g_selection.owner == g_capture.owner &&
                 g_selection.hand_index == g_capture.hand_index &&
                 cr_native_touch::arena_contains_pixel(layout, x, y)) {
        // TAP TO PLACE: Card was already selected, tapped arena to deploy
        submit = true;
        submit_owner = g_capture.owner;
        submit_hand_index = g_capture.hand_index;
        submit_point = cr_native_touch::arena_pixel_to_world(layout, x, y);
      }
      clear_capture_locked();
    } else if (action == kActionCancel) {
      clear_capture_locked();
    } else if (action != kActionMove) {
      // Pointer DOWN/UP are normalized to 0/1 by the Java SurfaceView.
      // Unknown terminal-like actions are swallowed for this captured
      // pointer but do not mutate the selection.
    }
  }
  g_selection.gesture_captured = g_capture.kind != CaptureKind::None;
  const std::uint64_t selection_revision = g_selection.revision;
  pthread_mutex_unlock(&g_touch_mutex);

  if (selected) {
    TOUCH_LOGI("selected native-render hand owner=%d slot=%d revision=%llu", selected_owner, selected_hand_index,
               static_cast<unsigned long long>(selection_revision));
  }

  if (submit) {
    const bool queued = g_bindings.submit_hand_action != nullptr &&
                        g_bindings.submit_hand_action(g_bindings.context, submit_owner, submit_hand_index,
                                                      submit_point.x, submit_point.y);
    if (queued) {
      pthread_mutex_lock(&g_touch_mutex);
      if (g_selection.owner == submit_owner && g_selection.hand_index == submit_hand_index) {
        clear_selection_locked();
      }
      const std::uint64_t revision = g_selection.revision;
      pthread_mutex_unlock(&g_touch_mutex);
      TOUCH_LOGI("queued native-render placement owner=%d slot=%d world=(%d,%d) revision=%llu", submit_owner,
                 submit_hand_index, submit_point.x, submit_point.y, static_cast<unsigned long long>(revision));
    } else {
      TOUCH_LOGE("native-render placement queue rejected owner=%d slot=%d world=(%d,%d)", submit_owner,
                 submit_hand_index, submit_point.x, submit_point.y);
    }
  }

  if (!consume) {
    forward_original(env, owner_class, action, x, y, pointer_id);
  }
}

void emit_absolute_jump(void *destination, const void *target) {
  // ldr x17, #8; br x17; .quad target
  const std::uint32_t instructions[2] = {0x58000051u, 0xd61f0220u};
  std::memcpy(destination, instructions, sizeof(instructions));
  std::memcpy(static_cast<std::uint8_t *>(destination) + 8, &target, sizeof(target));
}

bool make_page_writable(void *address, int protection) {
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return false;
  }
  const auto raw = reinterpret_cast<std::uintptr_t>(address);
  const auto page = raw & ~static_cast<std::uintptr_t>(page_size - 1);
  return mprotect(reinterpret_cast<void *>(page), static_cast<std::size_t>(page_size), protection) == 0;
}

} // namespace

bool install_native_touch_interceptor(std::uintptr_t libg_base, const NativeTouchBindings &bindings) {
#if !defined(__aarch64__)
  (void)libg_base;
  (void)bindings;
  return false;
#else
  if (libg_base == 0 || bindings.is_native_render_ready == nullptr || bindings.native_render_generation == nullptr ||
      bindings.submit_hand_action == nullptr) {
    TOUCH_LOGE("native touch bindings are incomplete");
    return false;
  }
  if (g_installed.load(std::memory_order_acquire)) {
    return true;
  }

  auto *target = reinterpret_cast<std::uint8_t *>(libg_base + kGameAppTouchOffset);
  if (std::memcmp(target, kGameAppTouchPrologue, sizeof(kGameAppTouchPrologue)) != 0) {
    TOUCH_LOGE("GameApp.nOnTouchEvent prologue mismatch at +0x%zx", static_cast<std::size_t>(kGameAppTouchOffset));
    return false;
  }

  void *trampoline = mmap(nullptr, 32, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (trampoline == MAP_FAILED) {
    TOUCH_LOGE("mmap touch trampoline failed");
    return false;
  }
  std::memcpy(trampoline, target, sizeof(kGameAppTouchPrologue));
  emit_absolute_jump(static_cast<std::uint8_t *>(trampoline) + 16, target + 16);
  __builtin___clear_cache(static_cast<char *>(trampoline), static_cast<char *>(trampoline) + 32);

  if (!make_page_writable(target, PROT_READ | PROT_WRITE | PROT_EXEC)) {
    TOUCH_LOGE("mprotect touch target RWX failed");
    munmap(trampoline, 32);
    return false;
  }
  g_bindings = bindings;
  g_original_game_app_touch = reinterpret_cast<GameAppTouch>(trampoline);
  emit_absolute_jump(target, reinterpret_cast<const void *>(&native_touch_game_app_hook));
  __builtin___clear_cache(reinterpret_cast<char *>(target), reinterpret_cast<char *>(target + 16));
  make_page_writable(target, PROT_READ | PROT_EXEC);
  g_installed.store(true, std::memory_order_release);
  TOUCH_LOGI("GameApp.nOnTouchEvent hook installed target=%p trampoline=%p layout=1080x1920", target, trampoline);
  return true;
#endif
}

NativeTouchSelectionSnapshot native_touch_selection_snapshot() {
  pthread_mutex_lock(&g_touch_mutex);
  NativeTouchSelectionSnapshot snapshot = g_selection;
  snapshot.gesture_captured = g_capture.kind != CaptureKind::None;
  pthread_mutex_unlock(&g_touch_mutex);
  return snapshot;
}

void native_touch_clear_selection() {
  pthread_mutex_lock(&g_touch_mutex);
  clear_capture_locked();
  clear_selection_locked();
  g_selection.gesture_captured = false;
  pthread_mutex_unlock(&g_touch_mutex);
}

bool native_touch_inject(std::int32_t action, std::int32_t x, std::int32_t y, std::int32_t pointer_id) {
  GameAppTouch original = g_original_game_app_touch;
  if (original == nullptr) {
    TOUCH_LOGE("native_touch_inject: original GameApp.nOnTouchEvent trampoline is not installed");
    return false;
  }
  JNIEnv *env = nullptr;
  bool attached = false;
  if (g_touch_java_vm != nullptr) {
    const jint status = g_touch_java_vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6);
    if (status == JNI_EDETACHED) {
      if (g_touch_java_vm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
        attached = true;
      }
    }
  }
  if (env != nullptr && g_cached_game_app_class == nullptr) {
    jclass local_class = env->FindClass("com/supercell/clashroyale/GameApp");
    if (local_class != nullptr) {
      g_cached_game_app_class = static_cast<jclass>(env->NewGlobalRef(local_class));
      env->DeleteLocalRef(local_class);
    } else if (env->ExceptionCheck()) {
      env->ExceptionClear();
    }
  }
  if (env != nullptr && g_cached_game_app_class != nullptr) {
    original(env, g_cached_game_app_class, action, x, y, pointer_id);
  }
  if (attached && g_touch_java_vm != nullptr) {
    g_touch_java_vm->DetachCurrentThread();
  }
  return true;
}

bool is_native_touch_installed() {
  return g_installed.load(std::memory_order_acquire);
}

void set_native_touch_java_vm(void *vm) {
  g_touch_java_vm = static_cast<JavaVM *>(vm);
}

void set_native_touch_layout_dimensions(std::int32_t width, std::int32_t height) {
  if (width > 0 && height > 0) {
    g_touch_layout_width.store(width, std::memory_order_release);
    g_touch_layout_height.store(height, std::memory_order_release);
    TOUCH_LOGI("set_native_touch_layout_dimensions: %dx%d", width, height);
  }
}

void set_native_touch_controlled_owner(std::int32_t owner) {
  g_touch_controlled_owner.store(owner, std::memory_order_release);
  TOUCH_LOGI("set_native_touch_controlled_owner: %d", owner);
}
