#include "color_pipeline.h"
#include "dinov3_preprocess.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

int main() {
    // Known half-way tie cases where round() (half away from zero) and nearbyint() / Python round() (half to even)
    // previously disagreed (finding 3.8):
    // For (w=33, h=64, max_size=512, patch_size=16):
    //   scale = 512 / 64 = 8.0; w * scale / 16 = 33 * 8 / 16 = 16.5 -> nearbyint -> 16 (grid_w=16, target_w=256).
    {
        const negicc::AspectPreservedShape s = negicc::compute_aspect_preserved_shape(33, 64, 512, 16);
        assert(s.grid_w == 16 && s.grid_h == 32);
        assert(s.target_w == 256 && s.target_h == 512);

        int tw = 0, th = 0, gw = 0, gh = 0;
        compute_aspect_preserved_shape(33, 64, 512, 16, tw, th, gw, gh);
        assert(tw == s.target_w && th == s.target_h && gw == s.grid_w && gh == s.grid_h);
    }
    {
        const negicc::AspectPreservedShape s = negicc::compute_aspect_preserved_shape(2112, 4096, 512, 16);
        assert(s.grid_w == 16 && s.grid_h == 32);
        assert(s.target_w == 256 && s.target_h == 512);

        int tw = 0, th = 0, gw = 0, gh = 0;
        compute_aspect_preserved_shape(2112, 4096, 512, 16, tw, th, gw, gh);
        assert(tw == s.target_w && th == s.target_h && gw == s.grid_w && gh == s.grid_h);
    }

    // Odd half-way case (35, 64): 35 * 8 / 16 = 17.5 -> nearbyint -> 18 (even)
    {
        const negicc::AspectPreservedShape s = negicc::compute_aspect_preserved_shape(35, 64, 512, 16);
        assert(s.grid_w == 18 && s.grid_h == 32);
        assert(s.target_w == 288 && s.target_h == 512);
    }

    // Exhaustive cross-check across a grid of aspect ratios and sizes
    for (int w = 1; w <= 512; w += 3) {
        for (int h = 1; h <= 512; h += 5) {
            const negicc::AspectPreservedShape s = negicc::compute_aspect_preserved_shape(w, h, 512, 16);
            int tw = 0, th = 0, gw = 0, gh = 0;
            compute_aspect_preserved_shape(w, h, 512, 16, tw, th, gw, gh);
            assert(tw == s.target_w && th == s.target_h && gw == s.grid_w && gh == s.grid_h);
        }
    }

    std::printf("dinov3_shape_test: all assertions passed.\n");
    return 0;
}
