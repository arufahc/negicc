# Film frame detector evaluation

Scores the film frame detector (`film_frame_detector.cc`, which `neg_process --roi auto` uses to find the frame)
against a set of true crops, so a detector change can be checked for fixes and regressions instead of judged frame by
frame.

## Build and run

```bash
make neg_process frame_detect
python3 tools/frame_eval/score.py --bin bin_out/frame_detect \
  --root triled=<dir of the TriLED development RAWs> \
  --root web=<dir of the white-LED review corpus> \
  --root "film_scans=<dir holding 35mm BW, 35mm Negative, Negative Scans, 120 Targets, Ektar100 Research>" \
  --jobs 24 --out result.json [--truth other_truth.json] [--compare earlier_result.json]
```

`frame_detect` reads each input as neg_process does (a CFA RAW or DNG on its mosaic, a linear DNG on its demosaiced
pixels, a TIFF or JPEG as white-light RGB), runs the detector alone with neg_process's parameters and prints one JSON
line: the crop in sensor pixels, the format, confidence and reasons. The crop is the detector's even when its
confidence is under 0.5, where `neg_process --roi auto` takes the whole image instead. `score.py` runs it on every
frame of `truth.json` with the frame's true format (negicc-web passes `--film_format` with `--roi auto`) and counts a
crop as passing when its IoU with the true crop is at least 0.90. Frames whose root is not given are skipped and
counted. `--truth` scores another truth file of the same form; a frame may name its set in a `set` field, and the
summary lists the sets the file holds.

## The truth set (`truth.json`)

400 frames, each with its root, path relative to the root, true format, true crop (sensor pixels, x y w h) and how the
crop was obtained:

| Set | Frames | Truth |
|---|---|---|
| `triled` | 55 | negicc-station's export crops of the TriLED development frames (`stage26_setup_inventory.json`, half-size pixels x 2): 6x6 and 645 Portra 400 |
| `white` | 41 | the white-LED review set's sensor frames (negicc-station Iterations 226 and 229) |
| `bw` | 90 | 35mm B&W scans whose frame fills the sensor: the full sensor |
| `neg35` | 14 | 35mm colour negatives filling the sensor (the full sensor); the test-chart pair takes the chart's frame |
| `mf` | 200 | holder-framed 120 negatives, targets and Ektar research frames: the detector's crop where checked correct by eye on contact sheets (199 frames), else a hand label with each edge snapped to the strongest step within 150 px (1 frame); regression only (below) |

`excluded` lists 18 frames left out, with the reason: blank frames, and frames with no single frame to crop (two
half frames in the holder, a frame split by the holder).

The `mf` results are regression checks only: 199 of the 200 truths are the detector's own crops, checked by eye, so
a pass means the crop did not move from what an earlier detector found, a fail can be a fix as well as a break (look at
the frame), and a crop the eye check missed is never caught. The `mf` count is not an accuracy figure and should not be
quoted as one. The `triled`, `white`, `bw` and `neg35` truth come from outside the detector.
