#include "cfa_pipeline.h"
#include "flat_field.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using namespace negicc;

static void test_preview_debayer_matches_rgbview_g1() {
    const unsigned int filters = 0xB4B4B4B4u; // RGGB with G1=1, G2=3
    const int W = 24, H = 16, PITCH = 28;
    std::vector<uint16_t> cfa(PITCH * H, 0);
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            // Make G1 and G2 distinctly different so G1-only vs (G1+G2)>>1 would differ
            const int col = bayer_color_at(r, c, filters);
            cfa[r * PITCH + c] = (uint16_t)(col * 5000 + r * 37 + c * 13 + 100);
        }
    }

    std::vector<float> gains(4 * 4 * 3, 1.0f);
    for (size_t i = 0; i < gains.size(); ++i) gains[i] = 0.9f + 0.01f * (float)(i % 9);
    FlatField ff(gains, 4, 4, 2, W / 2, H / 2, "test_ff");
    auto tables = ff.tables(W / 2, H / 2, true);

    RgbView full_view = RgbView::from_cfa_half(cfa.data(), PITCH, W, H, filters, nullptr, tables->host(), tables);
    assert(full_view.valid());

    // Sub-ROI at even sensor coordinates (rx=4, ry=2, rw=12, rh=8 -> half-size 2, 1, 6, 4)
    const int rx = 4, ry = 2, rw = 12, rh = 8;
    for (int rot : {0, 90, 180, 270}) {
        for (int hf = 0; hf < 2; ++hf) {
            for (int vf = 0; vf < 2; ++vf) {
                RgbView sub = full_view.crop_and_orient(rx / 2, ry / 2, rw / 2, rh / 2, rot, hf != 0, vf != 0);
                assert(sub.valid());
                std::vector<uint16_t> from_view(sub.width * sub.height * 3);
                sub.materialize(from_view.data());

                std::vector<uint16_t> from_debayer(sub.width * sub.height * 3);
                PreviewDebayerArgs pa{};
                pa.cfa_bayer = cfa.data();
                pa.src_w = W;
                pa.src_h = H;
                pa.rx = rx;
                pa.ry = ry;
                pa.rw = rw;
                pa.rh = rh;
                pa.rot_cw = rot;
                pa.hflip = (hf != 0);
                pa.vflip = (vf != 0);
                pa.dst_w = sub.width;
                pa.dst_h = sub.height;
                pa.filters = filters;
                pa.rgb_out = from_debayer.data();
                pa.flat = tables->host();
                pa.flat_x0 = rx / 2;
                pa.flat_y0 = ry / 2;
                pa.cfa_pitch = PITCH;
                cpu_preview_debayer(pa);

                assert(from_view == from_debayer);
            }
        }
    }
}

static void test_grain_aware_debayer_and_pixel_shift() {
    // Both 0x94949494 (G2=1) and 0xB4B4B4B4 (G2=3) must be accepted as RGGB
    const int W = 8, H = 8;
    std::vector<uint16_t> cfa(W * H, 65500); // near-clip (> 65535 - 64)
    // Set diagonal neighbors around (1,1) (B pixel in RGGB) so 0.25*(r00+r01+r10+r11) = 65500.5 -> rounds to 65501 (away from zero, not 65500 ties-to-even)
    cfa[0 * W + 0] = 65500;
    cfa[0 * W + 2] = 65501;
    cfa[2 * W + 0] = 65500;
    cfa[2 * W + 2] = 65501;

    for (unsigned int f : {0x94949494u, 0xB4B4B4B4u}) {
        std::vector<uint16_t> out(W * H * 3, 0);
        GrainAwareDebayerArgs ga{};
        ga.cfa_bayer = cfa.data();
        ga.src_w = W;
        ga.src_h = H;
        ga.rx = 0;
        ga.ry = 0;
        ga.rw = W;
        ga.rh = H;
        ga.dst_w = W;
        ga.dst_h = H;
        ga.filters = f;
        ga.white_level = 65535.0f;
        ga.rgb_out = out.data();
        cpu_grain_aware_debayer(ga);
        // At (r=1, c=1), R is 0.25 * (65500 + 65501 + 65500 + 65501) = 65500.5 -> std::round -> 65501
        assert(out[(1 * W + 1) * 3 + 0] == 65501);
    }

    // Invalid rotation must throw
    {
        std::vector<uint16_t> out(W * H * 3, 0);
        GrainAwareDebayerArgs ga{};
        ga.cfa_bayer = cfa.data();
        ga.src_w = W;
        ga.src_h = H;
        ga.rw = W;
        ga.rh = H;
        ga.rot_cw = 30;
        ga.dst_w = W;
        ga.dst_h = H;
        ga.filters = 0xB4B4B4B4u;
        ga.rgb_out = out.data();
        bool threw = false;
        try {
            cpu_grain_aware_debayer(ga);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    // cpu_pixel_shift_merge at ry=0 accesses r-1 = -1 in bayer_color_at
    {
        std::vector<uint16_t> out(W * H * 3, 0);
        PixelShiftMergeArgs psa{};
        psa.bayer_f0 = cfa.data();
        psa.bayer_f1 = cfa.data();
        psa.bayer_f2 = cfa.data();
        psa.bayer_f3 = cfa.data();
        psa.src_w = W;
        psa.src_h = H;
        psa.rx = 0;
        psa.ry = 0;
        psa.rw = W;
        psa.rh = H;
        psa.dst_w = W;
        psa.dst_h = H;
        psa.filters = 0xB4B4B4B4u;
        psa.rgb_out = out.data();
        cpu_pixel_shift_merge(psa);
        assert(out[0] > 0);
    }
}

static void test_extract_and_transform_roi_throws_on_mismatch() {
    std::vector<uint16_t> src(8 * 6 * 3, 100);
    std::vector<uint16_t> dst(4 * 4 * 3, 0);
    bool threw = false;
    try {
        // Clamped ROI is 2x4 (since rx=6 on src_w=8), which != dst 4x4
        extract_and_transform_roi(src.data(), dst.data(), 8, 6, 6, 0, 4, 4, 0, false, false, 4, 4);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

static void test_corrupt_npz_overflow_guards() {
    char tmpl[] = "/tmp/negicc_npz_test_XXXXXX";
    int fd = mkstemp(tmpl);
    assert(fd >= 0);
    close(fd);
    std::string path(tmpl);

    // Craft minimal ZIP with 1 central directory entry whose lhoff = 0xFFFFFFF0u (overflows 32-bit lhoff + 30)
    std::vector<uint8_t> zip(46 + 4 + 22, 0);
    // Central directory header at offset 0
    zip[0] = 0x50; zip[1] = 0x4b; zip[2] = 0x01; zip[3] = 0x02;
    zip[20] = 4; // comp_sz
    zip[24] = 4; // uncomp_sz
    zip[28] = 4; // name_len ("a.py")
    // lhoff at pos + 42 = 0xFFFFFFF0
    zip[42] = 0xF0; zip[43] = 0xFF; zip[44] = 0xFF; zip[45] = 0xFF;
    std::memcpy(&zip[46], "a.py", 4);
    // EOCD at offset 50
    const size_t eocd = 50;
    zip[eocd + 0] = 0x50; zip[eocd + 1] = 0x4b; zip[eocd + 2] = 0x05; zip[eocd + 3] = 0x06;
    zip[eocd + 10] = 1; // num_entries = 1
    zip[eocd + 16] = 0; // cd_offset = 0

    {
        std::ofstream ofs(path, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(zip.data()), (std::streamsize)zip.size());
    }
    bool threw = false;
    try {
        (void)load_flat_field_file(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    // Also test oversized uncomp_sz = 0xFFFFFFFFu
    zip[24] = 0xFF; zip[25] = 0xFF; zip[26] = 0xFF; zip[27] = 0xFF;
    {
        std::ofstream ofs(path, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(zip.data()), (std::streamsize)zip.size());
    }
    threw = false;
    try {
        (void)load_flat_field_file(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    std::remove(path.c_str());
}

int main() {
    test_preview_debayer_matches_rgbview_g1();
    test_grain_aware_debayer_and_pixel_shift();
    test_extract_and_transform_roi_throws_on_mismatch();
    test_corrupt_npz_overflow_guards();
    std::printf("kernels_test: all assertions passed.\n");
    return 0;
}
