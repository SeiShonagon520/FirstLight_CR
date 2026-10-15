#include "native_touch_layout.h"

#include <cassert>

int main() {
    using namespace cr_native_touch;
    constexpr Layout layout = kNativeRender1080x1920;

    for (std::int32_t slot = 0; slot < 4; ++slot) {
        const HandHit top = hand_at_pixel(
            layout,
            layout.hand_centers_x[slot],
            layout.top_hand_y);
        assert(top.hit);
        assert(top.owner == 0);
        assert(top.hand_index == 3 - slot);

        const HandHit bottom = hand_at_pixel(
            layout,
            layout.hand_centers_x[slot],
            layout.bottom_hand_y);
        assert(bottom.hit);
        assert(bottom.owner == 1);
        assert(bottom.hand_index == slot);
    }

    assert(!hand_at_pixel(layout, 282, layout.top_hand_y).hit);
    assert(!hand_at_pixel(layout, 900, layout.top_hand_y).hit);
    assert(!hand_at_pixel(layout, 211, 970).hit);
    assert(!hand_at_pixel(layout, -1, layout.top_hand_y).hit);

    assert(arena_contains_pixel(layout, 54, 282));
    assert(arena_contains_pixel(layout, 1026, 1666));
    assert(arena_contains_pixel(layout, 540, 970));
    assert(!arena_contains_pixel(layout, 53, 970));
    assert(!arena_contains_pixel(layout, 540, 1667));

    // Decorative padding clamps to the nearest logical outer cell.
    constexpr WorldPoint top_left = arena_pixel_to_world(layout, 54, 282);
    static_assert(top_left.x == 500 && top_left.y == 500);
    constexpr WorldPoint center = arena_pixel_to_world(layout, 551, 1070);
    static_assert(center.x == 9500 && center.y == 16500);
    constexpr WorldPoint bottom_right =
        arena_pixel_to_world(layout, 1026, 1666);
    static_assert(bottom_right.x == 17500 && bottom_right.y == 31500);

    // Measured visual centers of the first/last columns and king-tower back row.
    constexpr WorldPoint far_left = arena_pixel_to_world(layout, 135, 1350);
    static_assert(far_left.x == 500 && far_left.y == 23500);
    constexpr WorldPoint far_right = arena_pixel_to_world(layout, 943, 1350);
    static_assert(far_right.x == 17500 && far_right.y == 23500);
    constexpr WorldPoint king_back = arena_pixel_to_world(layout, 540, 1647);
    static_assert(king_back.x == 9500 && king_back.y == 31500);

    // Conversion is fail-closed at the outermost legal deployment centers.
    constexpr WorldPoint beyond = arena_pixel_to_world(layout, 2000, 3000);
    static_assert(beyond.x == 17500 && beyond.y == 31500);

    // Live combat touch projection must map to the same physical phone screen pixels
    // whether the local player was assigned Player 0 or Player 1 by Supercell.
    constexpr Layout live_720 = kLiveBattle720x1280;
    std::int32_t px = 0, py = 0;

    // Player 1 own left tower (world_x = 3500, world_y = 25500) -> screen bottom-left
    live_world_to_arena_pixel(live_720, 1, 3500, 25500, px, py);
    assert(px == 179 && py == 787);

    // Player 1 own right tower (world_x = 14500, world_y = 25500) -> screen bottom-right
    live_world_to_arena_pixel(live_720, 1, 14500, 25500, px, py);
    assert(px == 540 && py == 787);

    // Player 1 opponent left tower (world_x = 3500, world_y = 6500) -> screen top-left
    live_world_to_arena_pixel(live_720, 1, 3500, 6500, px, py);
    assert(px == 179 && py == 342);

    // Player 0 own left tower (world_x = 14500, world_y = 6500) -> screen bottom-left
    live_world_to_arena_pixel(live_720, 0, 14500, 6500, px, py);
    assert(px == 180 && py == 788);

    // Player 0 own right tower (world_x = 3500, world_y = 6500) -> screen bottom-right
    live_world_to_arena_pixel(live_720, 0, 3500, 6500, px, py);
    assert(px == 541 && py == 788);

    // Player 0 opponent left tower (world_x = 14500, world_y = 25500) -> screen top-left
    live_world_to_arena_pixel(live_720, 0, 14500, 25500, px, py);
    assert(px == 180 && py == 343);

    // Verify 720x1280 native-render layout (MuMu standard resolution)
    constexpr Layout layout_720 = kNativeRender720x1280;
    for (std::int32_t slot = 0; slot < 4; ++slot) {
        const HandHit top = hand_at_pixel(layout_720, layout_720.hand_centers_x[slot], layout_720.top_hand_y);
        assert(top.hit);
        assert(top.owner == 0);
        assert(top.hand_index == 3 - slot);

        const HandHit bottom = hand_at_pixel(layout_720, layout_720.hand_centers_x[slot], layout_720.bottom_hand_y);
        assert(bottom.hit);
        assert(bottom.owner == 1);
        assert(bottom.hand_index == slot);
    }
    assert(arena_contains_pixel(layout_720, 36, 188));
    assert(arena_contains_pixel(layout_720, 684, 1111));
    assert(arena_contains_pixel(layout_720, 360, 650));
    assert(!arena_contains_pixel(layout_720, 35, 650));
    assert(!arena_contains_pixel(layout_720, 360, 1112));

    const WorldPoint top_left_720 = arena_pixel_to_world(layout_720, 36, 188);
    assert(top_left_720.x == 500 && top_left_720.y == 500);
    const WorldPoint bottom_right_720 = arena_pixel_to_world(layout_720, 684, 1111);
    assert(bottom_right_720.x == 17500 && bottom_right_720.y == 31500);

    return 0;
}
