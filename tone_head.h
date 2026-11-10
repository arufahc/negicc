// The white-light tone heads: from the frozen DINOv3 features of a preview, the tone a lab scan of the frame would
// have, as 11 descriptors of its pixel L* and chroma at network size (mean and standard deviation of each):
//   L2 L5 L10 L50 L90 L95 L98 (L* percentiles), IQR (L75 - L25), R9010 (L90 - L10), Cmean (mean C*ab), C90.
// Input: the 1920 pooled features [CLS of layer 12, mean patch token of layer 9, mean patch token of layer 12 (the
// backbone's "pooled" output, 1152), the patch tokens' standard deviation (DinoV3InferenceResult::token_sd, 768)],
// standardised with the stored xm and xs. Model: Linear(1920, 512), GELU (exact, erf), Linear(512, 256), GELU,
// Linear(256, 22); mean = out[j] * ys[j] + ym[j], sd = exp(clamp(out[11 + j], -6, 4) / 2) * ys[j], as the PyTorch
// model it was trained as. Weights: models/tone_head/tone_head.bin (models/tone_head/export_tone_head.py writes it
// from tone_head.pt; README.md there has the training and validation). Plain C++, about 1.1 M multiply-adds per run.
#ifndef TONE_HEAD_H
#define TONE_HEAD_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

class ToneHead {
public:
    enum Target { kL2, kL5, kL10, kL50, kL90, kL95, kL98, kIQR, kR9010, kCmean, kC90, kTargets };
    static constexpr int kInDim = 1920;
    static const char* target_name(int t) {
        static const char* names[kTargets] = {"L2", "L5", "L10", "L50", "L90", "L95", "L98", "IQR", "R9010",
                                              "Cmean", "C90"};
        return t >= 0 && t < kTargets ? names[t] : "";
    }

    // Parses tone_head.bin (magic "TNH1", uint32 in_dim, hidden1, hidden2, targets, then float32 xm, xs, w1, b1, w2, b2,
    // w3, b3, ym, ys). False, with the reason in error, when the data is not such a file.
    bool load(const void* data, size_t size, std::string& error) {
        in_dim_ = 0;
        const uint8_t* p = static_cast<const uint8_t*>(data);
        if (!p || size < 20 || std::memcmp(p, "TNH1", 4) != 0) {
            error = "not a tone head file (magic TNH1)";
            return false;
        }
        uint32_t d = 0, h1 = 0, h2 = 0, n = 0;
        std::memcpy(&d, p + 4, 4);
        std::memcpy(&h1, p + 8, 4);
        std::memcpy(&h2, p + 12, 4);
        std::memcpy(&n, p + 16, 4);
        if (d != (uint32_t)kInDim || n != (uint32_t)kTargets || h1 == 0 || h2 == 0 || h1 > 65536 || h2 > 65536) {
            error = "tone head file is not 1920 features to 11 targets";
            return false;
        }
        const size_t floats = 2 * (size_t)d + (size_t)h1 * d + h1 + (size_t)h2 * h1 + h2 + 2 * (size_t)n * h2 +
                              2 * (size_t)n + 2 * (size_t)n;
        if (size != 20 + 4 * floats) {
            error = "tone head file has the wrong size";
            return false;
        }
        std::vector<float> v(floats);
        std::memcpy(v.data(), p + 20, 4 * floats);
        const float* f = v.data();
        auto take = [&f](std::vector<float>& dst, size_t k) {
            dst.assign(f, f + k);
            f += k;
        };
        take(xm_, d);
        take(xs_, d);
        take(w1_, (size_t)h1 * d);
        take(b1_, h1);
        take(w2_, (size_t)h2 * h1);
        take(b2_, h2);
        take(w3_, 2 * (size_t)n * h2);
        take(b3_, 2 * (size_t)n);
        take(ym_, n);
        take(ys_, n);
        in_dim_ = (int)d;
        h1_ = (int)h1;
        h2_ = (int)h2;
        return true;
    }

    bool loaded() const { return in_dim_ > 0; }

    // mean[kTargets] and sd[kTargets] from the 1920 features. False when the head is not loaded.
    bool run(const float* features, double* mean, double* sd) const {
        if (!loaded()) return false;
        auto gelu = [](double a) { return 0.5 * a * (1.0 + std::erf(a * 0.70710678118654752440)); };
        std::vector<double> x(in_dim_), a1(h1_), a2(h2_);
        for (int k = 0; k < in_dim_; ++k) x[k] = (double)((features[k] - xm_[k]) / xs_[k]);  // float, as in training
        for (int j = 0; j < h1_; ++j) {
            const float* w = w1_.data() + (size_t)j * in_dim_;
            double s = b1_[j];
            for (int k = 0; k < in_dim_; ++k) s += (double)w[k] * x[k];
            a1[j] = gelu(s);
        }
        for (int j = 0; j < h2_; ++j) {
            const float* w = w2_.data() + (size_t)j * h1_;
            double s = b2_[j];
            for (int k = 0; k < h1_; ++k) s += (double)w[k] * a1[k];
            a2[j] = gelu(s);
        }
        for (int t = 0; t < kTargets; ++t) {
            double o[2];
            for (int half = 0; half < 2; ++half) {
                const int j = half * kTargets + t;
                const float* w = w3_.data() + (size_t)j * h2_;
                double s = b3_[j];
                for (int k = 0; k < h2_; ++k) s += (double)w[k] * a2[k];
                o[half] = s;
            }
            mean[t] = o[0] * ys_[t] + ym_[t];
            sd[t] = std::exp(0.5 * std::clamp(o[1], -6.0, 4.0)) * ys_[t];
        }
        return true;
    }

private:
    int in_dim_ = 0, h1_ = 0, h2_ = 0;
    std::vector<float> xm_, xs_, w1_, b1_, w2_, b2_, w3_, b3_, ym_, ys_;
};

#endif  // TONE_HEAD_H
