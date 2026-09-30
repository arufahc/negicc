# Film Profiles

ICC profiles for Sony A7R IV captures with triband filter.

Portra 400 profiles were generated using 7 exposure brackets of IT8 target R190808 (from -1 to +3.5 EV). The multiple brackets allow patches above the nominal white patch to be measured rather than extrapolated, preventing highlight clipping and cyan casts in bright highlights.

## Brackets

- Portra400-1 (-1.2 EV): 1/8s, for underexposed frames
- Portra400-0.5 (-0.5 EV): 1/8s
- Portra400 (0 EV): 1/8s, box speed baseline
- Portra400+1 (+1.1 EV): 1/6s
- Portra400+2 (+1.8 EV): 1/5s
- Portra400+3 (+2.7 EV): 1/4s
- Portra400+3.5 (+3.5 EV): 0.4s, for high-key or overexposed frames

Each bracket has a cLUT ICC profile and a corresponding `Info.txt` with crosstalk matrix and film base RGB readings.

## Usage

```sh
# Convert RAW with Portra400 profile to sRGB
bin_out/neg_process -p "profiles/Sony A7RM4 Portra400 R190808 cLUT.icc" -o output.tif input.ARW
```
