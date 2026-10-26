#include "cfa_pipeline.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace negicc;

static void test_bayer_color_negative_coords() {
    // 0x94949494 is standard RGGB: (0,0)->0(R), (0,1)->1(G1), (1,0)->3(G2), (1,1)->2(B)
    const unsigned int filters = 0x94949494u;
    // Negative rows/cols occur in cpu_pixel_shift_merge at r=0 (r-1 = -1).
    // Must not trigger signed left-shift UB under UBSan.
    assert(bayer_color_at(-1, 0, filters) == bayer_color_at(1, 0, filters));
    assert(bayer_color_at(-1, -1, filters) == bayer_color_at(1, 1, filters));
    assert(bayer_color_at(-2, -3, filters) == bayer_color_at(0, 1, filters));
}

static void test_from_cfa_half_validation_and_g1() {
    const unsigned int rggb = 0x94949494u;
    std::vector<uint16_t> cfa(16 * 12, 0);
    for (int r = 0; r < 12; ++r) {
        for (int c = 0; c < 16; ++c) {
            cfa[r * 16 + c] = (uint16_t)(1000 + r * 100 + c);
        }
    }

    // Invalid filters (0 and 1 are non-Bayer sentinels in LibRaw)
    assert(!RgbView::from_cfa_half(cfa.data(), 16, 12, 0u).valid());
    assert(!RgbView::from_cfa_half(cfa.data(), 16, 12, 1u).valid());
    // X-Trans without xtrans_col36 must fail
    assert(!RgbView::from_cfa_half(cfa.data(), 16, 16, 12, 9u, nullptr).valid());

    // Valid RGGB view: G1 only (no g2_ptr averaging), matching film_image_from_cfa
    RgbView v = RgbView::from_cfa_half(cfa.data(), 16, 12, rggb);
    assert(v.valid());
    assert(v.width == 8 && v.height == 6);
    assert(v.g2_ptr == nullptr);
    assert(v.cfa_step == 2);

    // Compare against film_image_from_cfa green channel
    const float black[4] = {0, 0, 0, 0};
    FilmImage fimg;
    const bool ok = film_image_from_cfa(
        cfa.data(), 16, 12, 16, 2,
        [rggb](int row, int col) { return bayer_color_at(row, col, rggb); },
        black, 65535.0f, fimg);
    assert(ok && fimg.step == 2);
    for (int y = 0; y < v.height; ++y) {
        for (int x = 0; x < v.width; ++x) {
            uint16_t px[3];
            v.sample(x, y, px);
            assert(px[0] == cfa[(2 * y + 0) * 16 + (2 * x + 0)]); // R at (0,0)
            assert(px[1] == cfa[(2 * y + 0) * 16 + (2 * x + 1)]); // G1 at (0,1)
            assert(px[2] == cfa[(2 * y + 1) * 16 + (2 * x + 1)]); // B at (1,1)
            const uint16_t fg = fimg.ch[1].data[y * fimg.ch[1].stride + x * fimg.ch[1].pitch];
            assert(px[1] == fg);
        }
    }
}

static void test_crop_and_orient_validation_and_d4() {
    const int W = 12, H = 8;
    std::vector<uint16_t> rgb(W * H * 3);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            rgb[(y * W + x) * 3 + 0] = (uint16_t)(100 + y * 20 + x);
            rgb[(y * W + x) * 3 + 1] = (uint16_t)(2000 + y * 20 + x);
            rgb[(y * W + x) * 3 + 2] = (uint16_t)(4000 + y * 20 + x);
        }
    }

    std::vector<float> gains(4 * 4 * 3, 1.0f);
    for (size_t i = 0; i < gains.size(); ++i) {
        gains[i] = 0.85f + 0.01f * (float)(i % 11);
    }
    FlatField ff(gains, 4, 4, 2, W, H, "test_ff");
    auto tables = ff.tables(W, H, true);
    RgbView base = RgbView::from_interleaved_rgb(rgb.data(), W, H, 3, W * 3, tables->host(), tables);
    assert(base.valid());

    // Zero or negative ROI dimensions must return invalid view
    assert(!base.crop_and_orient(0, 0, 0, 4, 0, false, false).valid());
    assert(!base.crop_and_orient(0, 0, 4, -1, 0, false, false).valid());
    // Completely outside ROI must return invalid view (not clamp to 1x1)
    assert(!base.crop_and_orient(W + 2, 0, 4, 4, 0, false, false).valid());
    assert(!base.crop_and_orient(0, H + 2, 4, 4, 0, false, false).valid());
    assert(!base.crop_and_orient(-10, 0, 4, 4, 0, false, false).valid());
    assert(!base.crop_and_orient(0, -10, 4, 4, 0, false, false).valid());

    // Non-multiple-of-90 rotation must throw std::invalid_argument
    bool threw = false;
    try {
        (void)base.crop_and_orient(0, 0, 4, 4, 45, false, false);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    // Materialize full unoriented view as reference (already flat-divided)
    std::vector<uint16_t> full_mat(W * H * 3);
    base.materialize(full_mat.data());

    // Verify all 16 D4 orientations on a sub-ROI (rx=2, ry=1, rw=7, rh=5)
    const int rx = 2, ry = 1, rw = 7, rh = 5;
    for (int rot : {0, 90, 180, 270, -90, 450}) {
        for (int hf = 0; hf < 2; ++hf) {
            for (int vf = 0; vf < 2; ++vf) {
                RgbView sub = base.crop_and_orient(rx, ry, rw, rh, rot, hf != 0, vf != 0);
                assert(sub.valid());
                const int norm_rot = ((rot % 360) + 360) % 360;
                const int exp_w = (norm_rot == 90 || norm_rot == 270) ? rh : rw;
                const int exp_h = (norm_rot == 90 || norm_rot == 270) ? rw : rh;
                assert(sub.width == exp_w && sub.height == exp_h);

                for (int y = 0; y < exp_h; ++y) {
                    for (int x = 0; x < exp_w; ++x) {
                        uint16_t px[3];
                        sub.sample(x, y, px);
                        int x1 = hf ? (exp_w - 1 - x) : x;
                        int y1 = vf ? (exp_h - 1 - y) : y;
                        int x_rel = 0, y_rel = 0;
                        if (norm_rot == 0) { x_rel = x1; y_rel = y1; }
                        else if (norm_rot == 90) { x_rel = y1; y_rel = rh - 1 - x1; }
                        else if (norm_rot == 180) { x_rel = rw - 1 - x1; y_rel = rh - 1 - y1; }
                        else { x_rel = rw - 1 - y1; y_rel = x1; }
                        const uint16_t* ref = &full_mat[((ry + y_rel) * W + (rx + x_rel)) * 3];
                        assert(px[0] == ref[0] && px[1] == ref[1] && px[2] == ref[2]);
                    }
                }
            }
        }
    }
}

static void test_xtrans_6x6_view_and_to_film_image() {
    // Standard Fujifilm X-Trans 6x6 pattern
    const uint8_t xtrans[36] = {
        1, 1, 0, 1, 1, 2,
        1, 1, 2, 1, 1, 0,
        2, 0, 1, 0, 2, 1,
        1, 1, 2, 1, 1, 0,
        1, 1, 0, 1, 1, 2,
        0, 2, 1, 2, 0, 1
    };
    const int W = 24, H = 18, PITCH = 28;
    std::vector<uint16_t> cfa(PITCH * H, 0);
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            cfa[r * PITCH + c] = (uint16_t)(500 + r * 50 + c);
        }
    }
    RgbView xv = RgbView::from_cfa_half(cfa.data(), PITCH, W, H, 9u, xtrans);
    assert(xv.valid());
    assert(xv.width == 4 && xv.height == 3);
    assert(xv.cfa_step == 6);
    assert(xv.pitch == 6);
    assert(xv.stride == 6 * PITCH);

    FilmImage fi1 = xv.to_film_image(65535.0f, 1);
    assert(fi1.step == 1 && fi1.width == 4 && fi1.height == 3);
    FilmImage fi6 = xv.to_film_image(65535.0f, 2);
    assert(fi6.step == 6 && fi6.width == 24 && fi6.height == 18);

    // Oriented view must be refused by to_film_image
    RgbView xv_rot = xv.crop_and_orient(0, 0, 4, 3, 90, false, false);
    FilmImage fi_bad = xv_rot.to_film_image(65535.0f, 2);
    assert(fi_bad.step == 0 && fi_bad.width == 0);
}

int main() {
    test_bayer_color_negative_coords();
    test_from_cfa_half_validation_and_g1();
    test_crop_and_orient_validation_and_d4();
    test_xtrans_6x6_view_and_to_film_image();
    std::printf("rgbview_test: all assertions passed.\n");
    return 0;
}
