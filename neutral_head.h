// The neutral-patch head: per DINOv3 patch, the probability that the patch is neutral grey in a good rendering of
// the frame, from the backbone's layer 9 + 12 tokens [N, 768] (DinoV3Runner "tokens", the adapters' input).
// p = sigmoid(w2 . gelu(W1 ((t - mean) / std) + b1) + b2), W1 [256][768], exact (erf) GELU, as the PyTorch model it
// was trained as. Weights: models/neutral_head/neutral_head.bin (models/neutral_head/export_neutral_head.py writes
// it from neutral_head.pt; README.md there has the training and validation).
// A plain C++ implementation (about 0.2 GFLOP per 672-patch preview), so the head needs no runtime of its own and
// runs wherever the tokens are on the host.
#ifndef NEUTRAL_HEAD_H
#define NEUTRAL_HEAD_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

class NeutralHead {
public:
    // Parses neutral_head.bin (magic "NPH1", uint32 in_dim and hidden, then float32 mean, std, w1, b1, w2, b2).
    // False, with the reason in error, when the data is not such a file.
    bool load(const void* data, size_t size, std::string& error) {
        in_dim_ = hidden_ = 0;
        const uint8_t* p = static_cast<const uint8_t*>(data);
        if (!p || size < 12 || std::memcmp(p, "NPH1", 4) != 0) {
            error = "not a neutral head file (magic NPH1)";
            return false;
        }
        uint32_t d = 0, h = 0;
        std::memcpy(&d, p + 4, 4);
        std::memcpy(&h, p + 8, 4);
        const size_t floats = 2 * (size_t)d + (size_t)h * d + 2 * (size_t)h + 1;
        if (d == 0 || h == 0 || d > 65536 || h > 65536 || size != 12 + 4 * floats) {
            error = "neutral head file has the wrong size for in_dim " + std::to_string(d) + ", hidden " +
                    std::to_string(h);
            return false;
        }
        std::vector<float> v(floats);
        std::memcpy(v.data(), p + 12, 4 * floats);
        const float* f = v.data();
        mean_.assign(f, f + d);
        f += d;
        std_.assign(f, f + d);  // divided by, as in PyTorch (not multiplied by a reciprocal)
        f += d;
        // w1 transposed to [in][hidden], so the inner loop runs over contiguous hidden units (vectorises without
        // reassociating the sums, which -ffp-contract=off and the absence of -ffast-math keep in order).
        w1t_.resize((size_t)d * h);
        for (uint32_t j = 0; j < h; ++j)
            for (uint32_t k = 0; k < d; ++k) w1t_[(size_t)k * h + j] = f[(size_t)j * d + k];
        f += (size_t)h * d;
        b1_.assign(f, f + h);
        f += h;
        w2_.assign(f, f + h);
        f += h;
        b2_ = *f;
        in_dim_ = (int)d;
        hidden_ = (int)h;
        return true;
    }

    bool loaded() const { return in_dim_ > 0; }
    int in_dim() const { return in_dim_; }

    // p[i] for the n patches of tokens (n x in_dim, row-major).
    void run(const float* tokens, int n, float* p) const {
        const int d = in_dim_, h = hidden_;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            std::vector<float> x(d), acc(b1_);
            const float* t = tokens + (size_t)i * d;
            for (int k = 0; k < d; ++k) x[k] = (t[k] - mean_[k]) / std_[k];
            for (int k = 0; k < d; ++k) {
                const float xk = x[k];
                const float* w = w1t_.data() + (size_t)k * h;
                for (int j = 0; j < h; ++j) acc[j] += xk * w[j];
            }
            double z = b2_;
            for (int j = 0; j < h; ++j) {
                const double a = acc[j];
                z += (double)w2_[j] * (0.5 * a * (1.0 + std::erf(a * 0.70710678118654752440)));
            }
            p[i] = (float)(1.0 / (1.0 + std::exp(-z)));
        }
    }

private:
    int in_dim_ = 0, hidden_ = 0;
    std::vector<float> mean_, std_, w1t_, b1_, w2_;
    float b2_ = 0.0f;
};

#endif  // NEUTRAL_HEAD_H
