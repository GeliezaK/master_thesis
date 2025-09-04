#!/bin/bash

# Input/output directories
INPUT_DIR="../data/claas-3_2018-2020"
OUTPUT_DIR="../data/claas-3_2018-2020/monthly_cropped"
TMP_DIR="../data/claas-3_2018-2020/tmp"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TMP_DIR"

# Bergen ROI ~100km x 100km
CENTER_LAT=60.39
CENTER_LON=5.33
DEG_LAT_TO_KM=111.412
DEG_LON_TO_KM=$(echo "111.317*c($CENTER_LAT*3.14159265/180)" | bc -l)
LAT_OFFSET=$(echo "12.5/$DEG_LAT_TO_KM" | bc -l)
LON_OFFSET=$(echo "12.5/$DEG_LON_TO_KM" | bc -l)
NORTH=$(echo "$CENTER_LAT + $LAT_OFFSET" | bc -l)
SOUTH=$(echo "$CENTER_LAT - $LAT_OFFSET" | bc -l)
WEST=$(echo "$CENTER_LON - $LON_OFFSET" | bc -l)
EAST=$(echo "$CENTER_LON + $LON_OFFSET" | bc -l)

echo "Bergen ROI: $WEST,$EAST,$SOUTH,$NORTH"

# Loop over years and months
for year in {2018..2020}; do
    for month in {1..12}; do
        # Skip months outside 2018-08 to 2020-07
        if [[ $year -eq 2018 && $month -lt 8 ]]; then continue; fi
        if [[ $year -eq 2020 && $month -gt 7 ]]; then continue; fi

        ym=$(printf "%04d%02d" "$year" "$month")
        out_file="$OUTPUT_DIR/CLAAS3_${ym}.nc"

        # Skip if already done
        if [[ -f "$out_file" ]]; then
            echo "Output for $ym exists, skipping"
            continue
        fi

        # Find input files
        files=($INPUT_DIR/CMAin${ym}*.nc)
        if [ ${#files[@]} -eq 0 ]; then
            echo "No files for $ym, skipping"
            continue
        fi

        echo "Processing $ym with ${#files[@]} files"

        tmp_files=()
        for f in "${files[@]}"; do
            fname=$(basename "$f")
            tmp_f="$TMP_DIR/${fname%.nc}_harm_crop.nc"

            # Extract timestamp from filename (assumes YYYYMMDDHHMM)
            ts=$(echo "$fname" | grep -oE '[0-9]{12}' || true)
            if [[ -n "$ts" ]]; then
                yyyy=${ts:0:4}; mm=${ts:4:2}; dd=${ts:6:2}; hh=${ts:8:2}; mi=${ts:10:2}
                # Harmonize time and crop safely
                cdo -s settaxis,"$yyyy-$mm-$dd","$hh:$mi:00",1hour \
                    -selname,cma -sellonlatbox,$WEST,$EAST,$SOUTH,$NORTH "$f" "$tmp_f" 2>/dev/null \
                    || { echo "Failed to process $f, skipping"; continue; }
            else
                # Fallback: just crop
                cdo -s selname,cma -sellonlatbox,$WEST,$EAST,$SOUTH,$NORTH "$f" "$tmp_f" 2>/dev/null \
                    || { echo "Failed to crop $f, skipping"; continue; }
            fi

            # Only add file if it exists (some may be empty after crop)
            if [[ -f "$tmp_f" ]]; then
                tmp_files+=("$tmp_f")
            fi
        done

        if [ ${#tmp_files[@]} -eq 0 ]; then
            echo "No valid files for $ym after cropping, skipping month"
            continue
        fi

        # Merge cropped files into monthly file
        echo "Merging ${#tmp_files[@]} cropped files -> $out_file"
        cdo -L mergetime "${tmp_files[@]}" "$out_file" 2>/dev/null \
            || { echo "Failed to merge $ym, skipping"; continue; }

        # Clean up tmp files
        rm -f "${tmp_files[@]}"
    done
done

echo "All done."
