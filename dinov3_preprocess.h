#ifndef NEGICC_DINOV3_PREPROCESS_H
#define NEGICC_DINOV3_PREPROCESS_H

#include <cstdint>
#include <cstddef>
#include <vector>

namespace negicc {

// Antialiased 2-pass separable bilinear resize on 8-bit RGB with kernel support widening;
// bit-exact with Pillow's Image.Resampling.BILINEAR (fixed point, 8-bit intermediate).
// py-ref: negicc_station/src/tensorrt_intent_model.py:189 @ 2f7ee4a (PIL Image.Resampling.BILINEAR)
void antialiased_bilinear_resize_rgb(const uint8_t* src, int src_w, int src_h, int src_stride,
                                     uint8_t* dst, int dst_w, int dst_h);

// cv2.resize(..., interpolation=cv2.INTER_LINEAR) on 8-bit RGB (the fixed-point SIMD path).
// py-ref: negicc_station/src/per_target_conversion.py:217-222 @ 2f7ee4a (self_consistency_residual)
void cv_linear_resize_rgb(const uint8_t* src, int src_w, int src_h, int src_stride,
                          uint8_t* dst, int dst_w, int dst_h);

// intent_model.compute_tone_weights on a (grid_w * patch) x (grid_h * patch) RGB image: [N, 3] weights.
// py-ref: negicc_station/src/dino_geometry.py:43-74 @ 2f7ee4a (compute_tone_weights)
void compute_tone_weights(const uint8_t* img_u8, int grid_w, int grid_h, int patch_size, float* out_weights);

// Computes patch-aligned target dimensions preserving aspect ratio (intent_model.compute_aspect_preserved_shape)
// py-ref: negicc_station/src/dino_geometry.py:30-41 @ 2f7ee4a (compute_aspect_preserved_shape)
void compute_aspect_preserved_shape(int w, int h, int max_size, int patch_size,
                                    int& target_w, int& target_h, int& grid_w, int& grid_h);

// Compatibility alias for callers expecting dinov3:: namespace
namespace dinov3 {
    using negicc::antialiased_bilinear_resize_rgb;
    using negicc::cv_linear_resize_rgb;
    using negicc::compute_tone_weights;
    using negicc::compute_aspect_preserved_shape;
}

}  // namespace negicc

#endif  // NEGICC_DINOV3_PREPROCESS_H
