INSTALL_DIR="--install_dir='/Users/alpha/Library/ColorSync/Profiles/NegICC Profiles/'"
FILMS=("portra160" "portra400" "ektar100")
EXPS=("u2" "u1" "o0" "o1" "o2")

for f in "${FILMS[@]}"; do
    for e in "${EXPS[@]}"; do
	make "nikon_z7_${f}_${e}" INSTALL_DIR="$INSTALL_DIR"
    done
done

# Make a profile with old data. Taken with 1/3s exposure instead of
# +0.7 center weighted exposure.
make all INSTALL_DIR="$INSTALL_DIR"
