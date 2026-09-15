#!/usr/bin/env bash
# Copy one large file to a pod with runpodctl, ~25x faster than rsync.
#
# Measured on the same link with the same 1.16 GB .npz:
#   rsync -az      0.19 MB/s     (the -z alone costs a factor of 3 on
#                                 already-compressed data)
#   scp / raw ssh  0.54 MB/s     (one stream)
#   runpodctl      4.9-7.1 MB/s  (croc, several streams)
#
# Two things the handshake gets wrong if you do it by hand: `send` appends a
# digit to the code phrase you pass, and the *full* code has to reach the
# receiver or it reports "Malformed relay"; and a failed attempt leaves the relay
# room occupied ("room full"), so a retry needs a *new* phrase, not the same one.
# `receive` writes into the current directory and takes no destination argument.
#
# Usage:  bash tools/infra/transfer_to_pod.sh <ssh-alias> <local-file> [remote-dir]
set -euo pipefail

POD="${1:?usage: transfer_to_pod.sh <ssh-alias> <local-file> [remote-dir]}"
FILE="${2:?usage: transfer_to_pod.sh <ssh-alias> <local-file> [remote-dir]}"
DEST="${3:-/workspace/facetpy}"
[ -f "$FILE" ] || { echo "no such file: $FILE" >&2; exit 1; }

command -v runpodctl >/dev/null || {
  echo "runpodctl fehlt lokal: brew install runpod/runpodctl/runpodctl" >&2; exit 1; }

LOG=$(mktemp)
# A fresh phrase every run, so an earlier failure cannot leave the room occupied.
# Shape it like the codes runpodctl generates itself (digits-word-word-word);
# a fresh phrase every run, so an earlier failure cannot leave the room occupied.
CODE_BASE="$(date +%s | tail -c 5)-facet-$(basename "$FILE" | tr -cd 'a-z' | cut -c1-6)-send"
runpodctl send --code "$CODE_BASE" "$FILE" > "$LOG" 2>&1 &
SEND_PID=$!
trap 'kill $SEND_PID 2>/dev/null || true; rm -f "$LOG"' EXIT

CODE=""
for _ in $(seq 1 30); do
  # `|| true`: under `set -e` a grep that has not matched *yet* is not an error,
  # it is the normal state of the first second. Without it the loop aborts on its
  # first pass and the trap kills the sender before the handshake even starts.
  CODE=$(grep -o 'runpodctl receive [0-9a-z-]*' "$LOG" 2>/dev/null | tail -1 | awk '{print $3}' || true)
  [ -n "$CODE" ] && break
  sleep 1
done
[ -n "$CODE" ] || { echo "kein Code vom Sender:"; cat "$LOG"; exit 1; }

echo "$(basename "$FILE") -> $POD:$DEST  (code $CODE)"
START=$(date +%s)
ssh -o BatchMode=yes "$POD" "mkdir -p '$DEST' && cd '$DEST' && runpodctl receive $CODE" > /dev/null 2>&1
END=$(date +%s)

# Geprüft wird die Prüfsumme, nicht die Größe. croc legt die Zieldatei **vorab
# in voller Länge** an und füllt sie danach; eine Größenprüfung ist deshalb schon
# erfüllt, bevor ein Byte Nutzdaten angekommen ist. Am 12.09.2026 hat genau das
# zwei Pods mit einer 1,16-GB-Datei zurückgelassen, die die Größenprüfung bestand
# und beim Öffnen "BadZipFile" warf -- real waren 511 MB bzw. 1138 MB drin, der
# Rest waren Löcher. Ein stiller Datenfehler ist teurer als eine zweite Minute
# Prüfsumme.
SIZE=$(ssh -o BatchMode=yes "$POD" "stat -c %s '$DEST/$(basename "$FILE")' 2>/dev/null || echo 0")
LOCAL=$(stat -f %z "$FILE" 2>/dev/null || stat -c %s "$FILE")
LOCAL_MD5=$(md5 -q "$FILE" 2>/dev/null || md5sum "$FILE" | cut -d' ' -f1)
REMOTE_MD5=$(ssh -o BatchMode=yes "$POD" "md5sum '$DEST/$(basename "$FILE")' 2>/dev/null | cut -d' ' -f1")
if [ "$REMOTE_MD5" = "$LOCAL_MD5" ]; then
  echo "ok: $SIZE Bytes in $((END-START)) s ($(python3 -c "print(f'{$LOCAL/1e6/max($END-$START,1):.1f}')") MB/s), md5 $LOCAL_MD5"
else
  echo "FEHLER: md5 lokal $LOCAL_MD5, remote $REMOTE_MD5 (remote $SIZE von $LOCAL Bytes)" >&2
  # Die Ruine wegräumen: eine Datei richtiger Größe mit falschem Inhalt wird beim
  # nächsten Versuch sonst als "schon da" durchgewinkt.
  ssh -o BatchMode=yes "$POD" "rm -f '$DEST/$(basename "$FILE")'"
  exit 1
fi
