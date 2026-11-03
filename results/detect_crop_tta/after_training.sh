#!/usr/bin/env bash
cd /home/lqtrung/code/python/ReDi-FM; R=results/detect_crop_tta
P=$(ps -eo pid,args | awk '/train_dinov3_detector.py/ && !/awk/ {print $1}' | head -1); [ -n "$P" ] && tail --pid=$P -f /dev/null
grep -E "^Saved" $R/train_detector.log || { echo "detector training failed"; exit 1; }
.venv/bin/python $R/box_accuracy.py 2>&1 | grep -vE "Warning|warn"
