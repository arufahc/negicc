# Film Profiles

Each directory is a profile bundle for one film stock, camera and scanning light, exported from negicc-station: one
ICC profile per calibration bracket and a `bundle.json`. Its README says how it was built. The bundles are built
into `neg_process`, which picks one with `--film_stock` and `--light_source` (`--list_profiles` lists them).
