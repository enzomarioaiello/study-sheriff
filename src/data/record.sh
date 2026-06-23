#!/usr/bin/env bash
# StudySheriff -- record a labelled training clip with correct naming.
#
# Usage:  ./record.sh <class_idx> <label> <subject> [seconds]
#   ./record.sh 0 deskwork alice 20
#   ./record.sh 2 phone    bob          # default 20 s
#   ./record.sh odd "" dave 15          # odd/unseen footage (Unknown threshold)
#
# Auto-increments the index, so just run it again for the next take.
# Frame the shot first in VNC with:  rpicam-hello -t 0
set -e

CLASS=$1; LABEL=$2; SUBJ=$3; SECS=${4:-20}
if [ -z "$CLASS" ] || [ -z "$SUBJ" ]; then
  echo "usage: $0 <class_idx|odd> <label> <subject> [seconds]"; exit 1
fi

DIR=~/study-sheriff/data/raw
mkdir -p "$DIR"

if [ "$CLASS" = "odd" ]; then
  PREFIX="odd_${SUBJ}"
else
  PREFIX="class${CLASS}_${LABEL}_${SUBJ}"
fi
N=$(ls "$DIR/${PREFIX}_"*.mp4 2>/dev/null | wc -l)
OUT="$DIR/${PREFIX}_$(printf '%03d' $((N + 1))).mp4"

echo "Recording ${SECS}s -> $(basename "$OUT")"
echo -n "Get ready... "; for i in 3 2 1; do echo -n "$i "; sleep 1; done; echo "GO"
rpicam-vid -t $((SECS * 1000)) --width 1280 --height 720 --framerate 15 --nopreview -o "$OUT"
echo "Saved $OUT"
