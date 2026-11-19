# Film Profiles

Each directory is a profile set (a bundle) for one film stock, camera and scanning light: one ICC profile per
calibration bracket (`Target_<n>.icc`), one print ICC per print material (`print_<material>.icc`) and a
`bundle.json`. Its README says how it was built. The profile sets are built into `neg_process`, which picks one with
`--film_stock` and `--light_source` (`--list_profiles` lists them).
