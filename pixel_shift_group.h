// Validation of the four files of a Sony 4-shot pixel shift capture by the group tags the camera writes into their
// makernotes (LibRaw: PixelShiftGroupID, nShotsInPixelShiftGroup, numInPixelShiftGroup), with no LibRaw dependency
// so tests can feed it tags directly.
#pragma once
#include <string>

struct PixelShiftTags {
  unsigned group_id = 0;
  int shots_in_group = 0;  // nShotsInPixelShiftGroup
  int num_in_group = 0;    // numInPixelShiftGroup, 1 for the first shot
  bool present() const { return shots_in_group > 0 && num_in_group > 0; }
};

enum class PixelShiftGroupCheck { kNoTags, kOk, kBad };

// kNoTags: none of the four files carries group tags (the caller falls back to the file counters). kOk: one group of
// four shots, given in shot order. kBad: err says why (some files tagged and some not, different groups, a group of
// another size, shots out of order or repeated).
inline PixelShiftGroupCheck check_pixel_shift_group(const PixelShiftTags tags[4], std::string& err) {
  int n_present = 0;
  for (int i = 0; i < 4; ++i) n_present += tags[i].present() ? 1 : 0;
  if (n_present == 0) return PixelShiftGroupCheck::kNoTags;
  if (n_present != 4) {
    err = "only some of the 4 files carry pixel shift group tags";
    return PixelShiftGroupCheck::kBad;
  }
  for (int i = 0; i < 4; ++i) {
    if (tags[i].group_id != tags[0].group_id) {
      err = "the 4 files belong to different pixel shift groups";
      return PixelShiftGroupCheck::kBad;
    }
    if (tags[i].shots_in_group != 4) {
      err = "the files belong to a pixel shift group of " + std::to_string(tags[i].shots_in_group) +
            " shots, not 4";
      return PixelShiftGroupCheck::kBad;
    }
    if (tags[i].num_in_group != i + 1) {
      err = "4-shot pixel shift files are out of frame order (file " + std::to_string(i + 1) + " is shot " +
            std::to_string(tags[i].num_in_group) + " of its group)";
      return PixelShiftGroupCheck::kBad;
    }
  }
  return PixelShiftGroupCheck::kOk;
}
