#pragma once

#include <cstdint>

// Pure, platform-independent geometry for the stock v15 replay HUD used by
// native-render.  Keeping this separate from the Android hook makes every hit
// box and the pixel-to-world transform host-testable.
namespace cr_native_touch {

struct Layout {
  std::int32_t screen_width;
  std::int32_t screen_height;
  std::int32_t arena_left;
  std::int32_t arena_top;
  std::int32_t arena_right;
  std::int32_t arena_bottom;
  std::int32_t world_left;
  std::int32_t world_top;
  std::int32_t world_right;
  std::int32_t world_bottom;
  std::int32_t top_hand_y;
  std::int32_t bottom_hand_y;
  std::int32_t hand_half_width;
  std::int32_t hand_half_height;
  std::int32_t hand_centers_x[4];
  std::int32_t ability_x;
  std::int32_t ability_y;
};

// Measured from captures/native_render_offline_final.png (1080x1920).
// The Android Surface itself is 1080x1920 even when the MuMu host window is
// resized, so host-window scaling does not change these input coordinates.
inline constexpr Layout kNativeRender1080x1920 = {
    1080, 1920, 54, 282, 1026, 1666, 111, 436, 967, 1666, 171, 1749, 67, 86, {211, 354, 497, 640}, 0, 0,
};

// Scaled for 720x1280 portrait (standard for MuMu/emulator instances).
inline constexpr Layout kNativeRender720x1280 = {
    720, 1280, 36, 188, 684, 1111, 74, 291, 645, 1111, 114, 1166, 45, 57, {141, 236, 331, 427}, 0, 0,
};

inline constexpr Layout compute_touch_layout(std::int32_t width, std::int32_t height) {
  if (width == 1080 && height == 1920) {
    return kNativeRender1080x1920;
  }
  if (width == 720 && height == 1280) {
    return kNativeRender720x1280;
  }
  const double sx = static_cast<double>(width) / 1080.0;
  const double sy = static_cast<double>(height) / 1920.0;
  Layout l = {};
  l.screen_width = width;
  l.screen_height = height;
  l.arena_left = static_cast<std::int32_t>(54 * sx);
  l.arena_top = static_cast<std::int32_t>(282 * sy);
  l.arena_right = static_cast<std::int32_t>(1026 * sx);
  l.arena_bottom = static_cast<std::int32_t>(1666 * sy);
  l.world_left = static_cast<std::int32_t>(111 * sx);
  l.world_top = static_cast<std::int32_t>(436 * sy);
  l.world_right = static_cast<std::int32_t>(967 * sx);
  l.world_bottom = static_cast<std::int32_t>(1666 * sy);
  l.top_hand_y = static_cast<std::int32_t>(171 * sy);
  l.bottom_hand_y = static_cast<std::int32_t>(1749 * sy);
  l.hand_half_width = static_cast<std::int32_t>(67 * sx);
  l.hand_half_height = static_cast<std::int32_t>(86 * sy);
  for (std::int32_t i = 0; i < 4; ++i) {
    l.hand_centers_x[i] = static_cast<std::int32_t>(kNativeRender1080x1920.hand_centers_x[i] * sx);
  }
  l.ability_x = 0;
  l.ability_y = 0;
  return l;
}

// Measured directly from live battle HUD (720x1280 portrait).
// In active combat, hand cards are evenly distributed across the bottom center/right
// with "Next Card" on the left, unlike replay HUD which squishes cards to the left.
// Ability button is on the right combat edge: (649, 967) on 720x1280, (974, 1450) on 1080x1920.
inline constexpr Layout kLiveBattle720x1280 = {
    720, 1280, 50, 170, 670, 970, 65, 190, 655, 940, 0, 1120, 55, 55, {224, 360, 494, 630},
    649, 967,
};

inline constexpr Layout kLiveBattle1080x1920 = {
    1080, 1920, 75, 255, 1005, 1455, 98, 285, 983, 1410, 0, 1680, 82, 82, {336, 540, 741, 945},
    974, 1450,
};

inline constexpr Layout compute_live_touch_layout(std::int32_t width, std::int32_t height) {
  if (width == 720 && height == 1280) {
    return kLiveBattle720x1280;
  }
  if (width == 1080 && height == 1920) {
    return kLiveBattle1080x1920;
  }
  const double sx = static_cast<double>(width) / 720.0;
  const double sy = static_cast<double>(height) / 1280.0;
  Layout l = {};
  l.screen_width = width;
  l.screen_height = height;
  l.arena_left = static_cast<std::int32_t>(kLiveBattle720x1280.arena_left * sx);
  l.arena_top = static_cast<std::int32_t>(kLiveBattle720x1280.arena_top * sy);
  l.arena_right = static_cast<std::int32_t>(kLiveBattle720x1280.arena_right * sx);
  l.arena_bottom = static_cast<std::int32_t>(kLiveBattle720x1280.arena_bottom * sy);
  l.world_left = static_cast<std::int32_t>(kLiveBattle720x1280.world_left * sx);
  l.world_top = static_cast<std::int32_t>(kLiveBattle720x1280.world_top * sy);
  l.world_right = static_cast<std::int32_t>(kLiveBattle720x1280.world_right * sx);
  l.world_bottom = static_cast<std::int32_t>(kLiveBattle720x1280.world_bottom * sy);
  l.top_hand_y = 0;
  l.bottom_hand_y = static_cast<std::int32_t>(kLiveBattle720x1280.bottom_hand_y * sy);
  l.hand_half_width = static_cast<std::int32_t>(kLiveBattle720x1280.hand_half_width * sx);
  l.hand_half_height = static_cast<std::int32_t>(kLiveBattle720x1280.hand_half_height * sy);
  for (std::int32_t i = 0; i < 4; ++i) {
    l.hand_centers_x[i] = static_cast<std::int32_t>(kLiveBattle720x1280.hand_centers_x[i] * sx);
  }
  l.ability_x = static_cast<std::int32_t>(kLiveBattle720x1280.ability_x * sx);
  l.ability_y = static_cast<std::int32_t>(kLiveBattle720x1280.ability_y * sy);
  return l;
}

struct HandHit {
  bool hit = false;
  std::int32_t owner = -1;
  std::int32_t hand_index = -1;
};

struct WorldPoint {
  std::int32_t x = 0;
  std::int32_t y = 0;
};

constexpr std::int32_t absolute(std::int32_t value) { return value < 0 ? -value : value; }

constexpr bool screen_point_valid(const Layout &layout, std::int32_t x, std::int32_t y) {
  return x >= 0 && y >= 0 && x < layout.screen_width && y < layout.screen_height;
}

constexpr HandHit hand_at_pixel(const Layout &layout, std::int32_t x, std::int32_t y) {
  if (!screen_point_valid(layout, x, y)) {
    return {};
  }

  std::int32_t owner = -1;
  if (absolute(y - layout.top_hand_y) <= layout.hand_half_height) {
    owner = 0;
  } else if (absolute(y - layout.bottom_hand_y) <= layout.hand_half_height) {
    owner = 1;
  } else {
    return {};
  }

  for (std::int32_t index = 0; index < 4; ++index) {
    if (absolute(x - layout.hand_centers_x[index]) <= layout.hand_half_width) {
      // The spectator HUD mirrors owner 0's hand at the top but keeps
      // owner 1's bottom hand in native order.
      const std::int32_t hand_index = owner == 0 ? 3 - index : index;
      return {true, owner, hand_index};
    }
  }
  return {};
}

constexpr bool arena_contains_pixel(const Layout &layout, std::int32_t x, std::int32_t y) {
  return x >= layout.arena_left && x <= layout.arena_right && y >= layout.arena_top && y <= layout.arena_bottom;
}

constexpr std::int32_t clamp_deploy(std::int64_t value, std::int32_t minimum, std::int32_t maximum) {
  return value <= minimum ? minimum : value >= maximum ? maximum : static_cast<std::int32_t>(value);
}

// Inverse of the measured logical 18x32 replay-HUD grid.  The visible arena
// catchment includes decorative padding, which clamps to the nearest cell.
// Returning cell centers mirrors the stock command stream and keeps the outer
// columns and the final rows behind the king towers reachable.
constexpr WorldPoint arena_pixel_to_world(const Layout &layout, std::int32_t x, std::int32_t y) {
  const std::int64_t pixel_width = layout.world_right - layout.world_left;
  const std::int64_t pixel_height = layout.world_bottom - layout.world_top;
  if (pixel_width <= 0 || pixel_height <= 0) {
    return {};
  }

  const auto cell_center = [](std::int32_t pixel, std::int32_t minimum, std::int32_t maximum,
                              std::int32_t cell_count) constexpr -> std::int32_t {
    if (pixel <= minimum) {
      return 500;
    }
    if (pixel >= maximum) {
      return (cell_count - 1) * 1000 + 500;
    }
    const std::int64_t raw_world = static_cast<std::int64_t>(pixel - minimum) * cell_count * 1000 / (maximum - minimum);
    const std::int32_t cell = clamp_deploy(raw_world / 1000, 0, cell_count - 1);
    return cell * 1000 + 500;
  };

  // Native world Y grows in the same direction as the stock replay surface:
  // owner 0 is near Y=0 at the top and owner 1 near Y=32000 at the bottom.
  return {
      cell_center(x, layout.world_left, layout.world_right, 18),
      cell_center(y, layout.world_top, layout.world_bottom, 32),
  };
}

// Map native world coordinates (0..18000, 0..32000) to screen pixel coordinates.
constexpr void world_to_arena_pixel(const Layout &layout, std::int32_t world_x, std::int32_t world_y,
                                    std::int32_t &out_x, std::int32_t &out_y) {
  const std::int64_t span_x = layout.world_right - layout.world_left;
  const std::int64_t span_y = layout.world_bottom - layout.world_top;
  const std::int64_t clamped_x = clamp_deploy(world_x, 0, 18000);
  const std::int64_t clamped_y = clamp_deploy(world_y, 0, 32000);
  out_x = static_cast<std::int32_t>(layout.world_left + (clamped_x * span_x) / 18000);
  out_y = static_cast<std::int32_t>(layout.world_top + (clamped_y * span_y) / 32000);
}

// Map native world coordinates (0..18000, 0..32000) to live combat screen pixels.
// In active live combat on the phone, the local player is ALWAYS rendered at the bottom
// of the screen with their left tower on the left and right tower on the right, regardless
// of whether Supercell's matchmaking assigned the local client as Player 0 or Player 1.
constexpr void live_world_to_arena_pixel(const Layout &layout, std::int32_t owner,
                                         std::int32_t world_x, std::int32_t world_y,
                                         std::int32_t &out_x, std::int32_t &out_y) {
  const std::int64_t span_x = layout.world_right - layout.world_left;
  const std::int64_t span_y = layout.world_bottom - layout.world_top;
  const std::int64_t clamped_x = clamp_deploy(world_x, 0, 18000);
  const std::int64_t clamped_y = clamp_deploy(world_y, 0, 32000);

  if (owner == 1) {
    // Owner 1 (True Blue / live client standard):
    // In Clashaiaa: view_x = world_x / 1000.0, view_y = 32.0 - world_y / 1000.0
    // Left is world_x = 0..3500 -> screen left (world_left)
    // Right is world_x = 14500..18000 -> screen right (world_right)
    // Own base is world_y = 32000 -> screen bottom (world_bottom)
    // Enemy base is world_y = 0 -> screen top (world_top)
    out_x = static_cast<std::int32_t>(layout.world_left + (clamped_x * span_x) / 18000);
    out_y = static_cast<std::int32_t>(layout.world_top + (clamped_y * span_y) / 32000);
  } else {
    // Owner 0 (True Red):
    // In Clashaiaa: view_x = 18.0 - world_x / 1000.0, view_y = world_y / 1000.0
    // Left is world_x = 14500..18000 -> screen left (world_left)
    // Right is world_x = 0..3500 -> screen right (world_right)
    // Own base is world_y = 0 -> screen bottom (world_bottom)
    // Enemy base is world_y = 32000 -> screen top (world_top)
    out_x = static_cast<std::int32_t>(layout.world_right - (clamped_x * span_x) / 18000);
    out_y = static_cast<std::int32_t>(layout.world_bottom - (clamped_y * span_y) / 32000);
  }
}

} // namespace cr_native_touch
