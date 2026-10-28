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

static void test_preview_debayer_matches_rgbview_averaged_green() {
    const unsigned int filters = 0xB4B4B4B4u; // RGGB with G1=1, G2=3
    const int W = 24, H = 16, PITCH = 28;
    std::vector<uint16_t> cfa(PITCH * H, 0);
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            // Make G1 and G2 distinctly different so G1-only and (G1+G2)>>1 differ
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

// ---- Highlight shoulder ------------------------------------------------------------------------------------

static PipelineConfig make_test_cfg(float x_hi, int encoding = ENCODE_SRGB) {
    PipelineConfig cfg{};
    cfg.knee = 1.0f;  // no soft knee: the test controls the pre-shoulder value through the matrix
    cfg.encoding = encoding;
    set_shoulder(cfg, x_hi);
    return cfg;
}

static void test_shoulder_off_is_bit_identical() {
    const PipelineConfig cfg = make_test_cfg(1.0f);
    const float m[9] = {1.9f, 0.1f, 0.0f, 0.05f, 1.7f, 0.02f, 0.0f, 0.1f, 1.5f};
    int n = 0;
    for (int r = 0; r < 65536; r += 4099) {
        for (int g = 0; g < 65536; g += 5003) {
            for (int b = 0; b < 65536; b += 6007) {
                const uint16_t px[3] = {(uint16_t)r, (uint16_t)g, (uint16_t)b};
                // The pipeline before the shoulder existed: clamp in the PCS to linear sRGB step, then encode.
                Rgb v = load_u16(px);
                v = apply_matrix(m, v);
                v = apply_knee(v, cfg.knee);
                v = apply_profile(cfg.profile, v);
                const Rgb want = encode_output(pcs_to_linear_srgb(v), ENCODE_SRGB);
                const Rgb got = convert_pixel(cfg, m, px);
                assert(std::memcmp(&want, &got, sizeof(Rgb)) == 0);
                // The pre-shoulder value is the same, clamped at 0 only.
                const Rgb pre = convert_pixel_linear(cfg, m, px);
                const Rgb clamped = pcs_to_linear_srgb(v);
                assert(clamp01(pre.r) == clamped.r && clamp01(pre.g) == clamped.g && clamp01(pre.b) == clamped.b);
                ++n;
            }
        }
    }
    assert(n > 100);
}

static void test_shoulder_curve() {
    for (float x_hi : {1.01f, 1.1f, 1.333f, 1.5f}) {
        for (float n : {2.0f, 8.0f}) {
            PipelineConfig cfg{};
            cfg.encoding = ENCODE_SRGB;
            set_shoulder(cfg, x_hi, n);
            assert(cfg.shoulder_x_k > 0.5f && cfg.shoulder_x_k < 1.0f);
            // The identity up to the knee, continuous and with a continuous slope there.
            const float k = cfg.shoulder_x_k;
            assert(apply_shoulder_channel(cfg, 0.5f * k) == 0.5f * k);
            assert(std::fabs(apply_shoulder_channel(cfg, k * 1.0001f) - k * 1.0001f) < 2e-5f);
            const float h = 2e-4f;
            const float slope_below = (apply_shoulder_channel(cfg, k) - apply_shoulder_channel(cfg, k * (1 - h))) / (k * h);
            const float slope_above =
                (apply_shoulder_channel(cfg, k * (1 + h)) - apply_shoulder_channel(cfg, k)) / (k * h);
            assert(std::fabs(slope_below - 1.0f) < 0.01f);
            assert(std::fabs(slope_above - 1.0f) < 0.1f);  // C1 join; the tightest spans curve fastest
            // The ceiling is reached at x_hi, holds above it, and the curve never decreases.
            const float ceiling = srgb_decode(cfg.shoulder_c);
            assert(std::fabs(apply_shoulder_channel(cfg, x_hi * 0.99999f) - ceiling) < 5e-3f);
            assert(std::fabs(apply_shoulder_channel(cfg, x_hi) - ceiling) < 1e-5f);
            assert(apply_shoulder_channel(cfg, 3.0f) == cfg.shoulder_y_c && std::fabs(cfg.shoulder_y_c - ceiling) < 1e-5f);
            float prev = 0.0f;
            for (float x = 0.0f; x < 2.0f; x += 0.001f) {
                const float y = apply_shoulder_channel(cfg, x);
                assert(y >= prev - 1e-6f);
                prev = y;
            }
        }
    }
    // Measured values (double precision bisection): x_hi 1.333, n = 2 -> knee 0.735, the chart white lands at code 245.
    PipelineConfig cfg{};
    cfg.encoding = ENCODE_SRGB;
    set_shoulder(cfg, 1.333f);
    assert(std::fabs(cfg.shoulder_x_k - 0.735f) < 0.005f);
    const float white_code = 255.0f * srgb_encode(apply_shoulder_channel(cfg, 1.0f));
    assert(std::fabs(white_code - 245.1f) < 0.3f);
    // Off, and out of range.
    PipelineConfig off{};
    set_shoulder(off, 1.0f);
    assert(off.shoulder_x_hi == 1.0f);
    set_shoulder(off, 9.0f);
    assert(off.shoulder_x_hi == kShoulderXHiLimit);
    // The ceiling is ramped in: continuous at the on/off boundary, the full 0.988 from 1 + kShoulderRamp.
    {
        PipelineConfig a{}, b{}, c{};
        a.encoding = b.encoding = c.encoding = ENCODE_SRGB;
        set_shoulder(a, 1.0002f);
        set_shoulder(b, 1.0f + kShoulderRamp);
        set_shoulder(c, 1.4f);
        assert(a.shoulder_x_hi > 1.0f && std::fabs(a.shoulder_c - 1.0f) < 1e-3f);
        assert(std::fabs(b.shoulder_c - kShoulderCeiling) < 1e-5f && std::fabs(c.shoulder_c - kShoulderCeiling) < 1e-5f);
        const float white_off = 255.0f * srgb_encode(1.0f);
        assert(std::fabs(255.0f * srgb_encode(apply_shoulder_channel(a, 1.0f)) - white_off) < 0.5f);
        // Neighbouring frames either side of the boundary differ by well under a code of white.
        PipelineConfig lo{}, hi{};
        lo.encoding = hi.encoding = ENCODE_SRGB;
        set_shoulder(lo, 1.0f);
        set_shoulder(hi, 1.005f);
        assert(std::fabs(255.0f * srgb_encode(apply_shoulder_channel(hi, 1.0f)) -
                         255.0f * srgb_encode(apply_shoulder_channel(lo, 1.0f))) < 0.7f);
        // Bad exponents are rejected; every in-range x_hi has a knee.
        bool threw = false;
        try { set_shoulder(a, 1.2f, 0.5f); } catch (const std::invalid_argument&) { threw = true; }
        assert(threw);
        for (float x = 1.0002f; x <= kShoulderXHiLimit; x += 0.0007f) {
            PipelineConfig k{};
            k.encoding = ENCODE_SRGB;
            set_shoulder(k, x);
            assert(k.shoulder_x_hi > 1.0f && k.shoulder_x_k > 0.0f && k.shoulder_x_k < 1.0001f);
        }
    }
    // The anchored x_hi maps the frame's 99.5% point q to sRGB 242: off at or below it, the value at q falls with x_hi,
    // q lands on the anchor code within a hundredth of a code, and past the limit the limit is used.
    {
        const float anchor = kShoulderAnchorCode / 255.0f;
        assert(shoulder_x_hi_anchored(0.5f) == 1.0f);
        assert(shoulder_x_hi_anchored(srgb_decode(anchor) * 0.999f) == 1.0f);
        assert(shoulder_x_hi_anchored(0.0f) == 1.0f && shoulder_x_hi_anchored(std::nanf("")) == 1.0f);
        float prev_x = 1.0f;
        for (float q = srgb_decode(anchor) * 1.002f; q <= 1.2f; q += 0.002f) {
            const float x_hi = shoulder_x_hi_anchored(q);
            PipelineConfig a{};
            a.encoding = ENCODE_SRGB;
            set_shoulder(a, x_hi);
            assert(x_hi > kShoulderOff && x_hi <= kShoulderXHiLimit && x_hi >= prev_x);
            assert(std::fabs(255.0f * srgb_encode(apply_shoulder_channel(a, q)) - kShoulderAnchorCode) < 0.01f);
            prev_x = x_hi;
        }
        for (float q = 0.9f; q <= 1.2f; q += 0.1f) {
            float prev = 2.0f;
            for (float x = 1.0002f; x <= kShoulderXHiLimit; x += 0.01f) {
                PipelineConfig a{};
                a.encoding = ENCODE_SRGB;
                set_shoulder(a, x);
                const float v = apply_shoulder_channel(a, q);
                assert(v <= prev + 1e-6f);
                prev = v;
            }
        }
        // Measured: q = 0.95 (code 249) -> x_hi 1.358, knee at code 221.1.
        PipelineConfig m{};
        m.encoding = ENCODE_SRGB;
        set_shoulder(m, shoulder_x_hi_anchored(0.95f));
        assert(std::fabs(m.shoulder_x_hi - 1.358f) < 0.002f && std::fabs(255.0f * m.shoulder_s_k - 221.1f) < 0.1f);
        assert(shoulder_x_hi_anchored(5.0f) == kShoulderXHiLimit);
    }
    // Only the sRGB encoding applies it.
    const PipelineConfig lin = make_test_cfg(1.3f, ENCODE_SRGB_LINEAR);
    const Rgb v = {1.2f, 0.5f, 0.9f};
    const Rgb same = apply_shoulder(lin, v);
    assert(same.r == v.r && same.g == v.g && same.b == v.b);
}

// The statistics read the pre-shoulder value and the render and the Lab the post-shoulder one.
static void test_shoulder_kernels_agree() {
    const int W = 64, H = 48;
    std::vector<uint16_t> rgb((size_t)W * H * 3);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const float t = (x + y * W) / (float)(W * H);  // 0..1 ramp to well above the chart white
            for (int c = 0; c < 3; ++c) rgb[((size_t)y * W + x) * 3 + c] = (uint16_t)(t * (0.8f + 0.1f * c) * 65535.0f);
        }
    }
    RgbView view = RgbView::from_interleaved_rgb(rgb.data(), W, H, 3);
    const float m[9] = {1.6f, 0, 0, 0, 1.6f, 0, 0, 0, 1.6f};

    PipelineConfig off = make_test_cfg(1.0f), on = make_test_cfg(1.0f);
    set_shoulder(on, 1.25f);

    auto clip = [&](const PipelineConfig& cfg, std::vector<uint64_t>* hist, float* out5) {
        HighlightClipArgs ha{};
        ha.in_view = &view;
        ha.width = W;
        ha.height = H;
        ha.matrices = m;
        ha.n = 1;
        ha.cfg = cfg;
        ha.ceil_lin[0] = ha.ceil_lin[1] = ha.ceil_lin[2] = 0.9f;
        ha.out = out5;
        if (hist) {
            hist->assign(kMaxHistBins, 0);
            ha.max_hist = hist->data();
        }
        cpu_highlight_clip(ha);
    };
    float c_off[5], c_on[5];
    std::vector<uint64_t> h_off, h_on;
    clip(off, &h_off, c_off);
    clip(on, &h_on, c_on);
    // Exceedance and white clip: pre-shoulder, the same either way. The median L* is post-shoulder: lower.
    for (int i = 0; i < 4; ++i) assert(c_off[i] == c_on[i]);
    assert(c_on[4] <= c_off[4]);
    assert(h_off == h_on);
    uint64_t total = 0;
    for (uint64_t v : h_on) total += v;
    assert(total == (uint64_t)W * H);
    const double q = max_hist_quantile(h_on.data(), 0.995);
    assert(q > 1.0 && q < kMaxHistRange);
    // Without the histogram the statistics are unchanged.
    float c_nohist[5];
    clip(on, nullptr, c_nohist);
    for (int i = 0; i < 5; ++i) assert(c_nohist[i] == c_on[i]);

    // Self-consistency: the Lab the intent loss scores is the Lab of the rendered (encoded) output. With the render's
    // own Lab as the target (sigma 1) the loss is zero to float noise; a render without the shoulder is not.
    std::vector<float> rendered((size_t)W * H * 3);
    RenderArgs ra{};
    ra.in_view = &view;
    ra.width = W;
    ra.height = H;
    ra.matrix = m;
    ra.cfg = on;
    ra.out_type = 2;
    ra.out = rendered.data();
    cpu_render(ra);
    std::vector<float> mu((size_t)W * H * 3), sigma((size_t)W * H * 3, 1.0f);
    for (int i = 0; i < W * H; ++i) {
        const Lab lab = srgb_to_lab({rendered[i * 3], rendered[i * 3 + 1], rendered[i * 3 + 2]});
        mu[i * 3] = lab.L;
        mu[i * 3 + 1] = lab.a;
        mu[i * 3 + 2] = lab.b;
    }
    auto loss = [&](const PipelineConfig& cfg) {
        IntentLossArgs ia{};
        ia.in_view = &view;
        ia.width = W;
        ia.height = H;
        ia.matrices = m;
        ia.n = 1;
        ia.cfg = cfg;
        ia.mu = mu.data();
        ia.sigma = sigma.data();
        float out[4];
        ia.out = out;
        cpu_intent_loss(ia);
        return out[0] + out[1] + out[2];
    };
    assert(loss(on) < 1e-2f);
    assert(loss(off) > 1.0f);

    // lab_stats agrees with the render too: its L* histogram counts the post-shoulder values.
    LabStatsRow rows[2];
    for (int k = 0; k < 2; ++k) {
        LabStatsArgs la{};
        la.in_view = &view;
        la.width = W;
        la.height = H;
        la.matrices = m;
        la.n = 1;
        la.cfg = k ? on : off;
        la.out = &rows[k];
        cpu_lab_stats(la);
    }
    assert(rows[0].white_clip == rows[1].white_clip);  // pre-shoulder
    float mean_on = 0, mean_off = 0;
    for (int b = 0; b < kLabBins; ++b) {
        mean_on += rows[1].hist[b] * b;
        mean_off += rows[0].hist[b] * b;
    }
    assert(mean_on < mean_off);
}

int main() {
    test_preview_debayer_matches_rgbview_averaged_green();
    test_grain_aware_debayer_and_pixel_shift();
    test_extract_and_transform_roi_throws_on_mismatch();
    test_corrupt_npz_overflow_guards();
    test_shoulder_off_is_bit_identical();
    test_shoulder_curve();
    test_shoulder_kernels_agree();
    std::printf("kernels_test: all assertions passed.\n");
    return 0;
}
