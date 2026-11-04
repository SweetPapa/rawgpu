#!/bin/sh
# Download the AMD firmware rawgpu loads, from a pinned linux-firmware commit,
# into local/firmware/amdgpu/ (gitignored). Every file is checked against the
# sha256 recorded here; a mismatch deletes the download and fails.
#
# The blobs are AMD's, under LICENSES/LICENSE.amdgpu (redistributable, no
# reverse engineering). rawgpu never commits them.
#
# usage: tools/fetch-firmware.sh [--verify]
set -eu

COMMIT=703a4d95f86606b14b8433e7197663874ef89eed
BASE="https://gitlab.com/kernel-firmware/linux-firmware/-/raw/$COMMIT"
DEST="$(cd "$(dirname "$0")/.." && pwd)/local/firmware"

# path  sha256  what loads it
FILES='
amdgpu/polaris10_sdma.bin    e27b3ec976f87e32883b9190e653b20d27f9e77da8dd2d77eeb10c35ee9f1e61  SDMA0 (sdma-test)
amdgpu/polaris10_sdma1.bin   23dd73eb1316d40a681841b4d3d116c5cae6d8317cebe8f5e1f9e5dec6de5c87  SDMA1
amdgpu/polaris10_rlc.bin     a5d4c2d3acc5c3c4988b95855fe21a5f682552b2b065f76439370835178ccbaf  RLC (compute queue)
amdgpu/polaris10_mec.bin     c19c1e2e36b47c4a8c25d601f5ae04a26458d98c3000f851d1be5c3a0a424b07  MEC (compute queue)
amdgpu/polaris10_mec_2.bin   d4914f1166e3b7af212837aaf8f393a155ce3f61f48bb08279c743a2ddea0b6a  MEC, newer variant
amdgpu/polaris10_mec2.bin    c19c1e2e36b47c4a8c25d601f5ae04a26458d98c3000f851d1be5c3a0a424b07  MEC2
amdgpu/polaris10_k2_smc.bin  b5d34a4b0ea0d0c9c53dddcac93e9e01fe759f0ad9a0b0d0b4905e5a6b254d03  SMC on Polaris 30 (power management)
amdgpu/polaris10_k_mc.bin    78a0e88d669cf91c09cd027ac3b5bd0c08ac126a8488092c567ecda21824e9ad  MC sequencer on Polaris 30
LICENSES/LICENSE.amdgpu      572872598565dc3513470de971a32bf9db301f47afeef3636345eadae33b2eee  license
'

verify_only=0
[ "${1:-}" = "--verify" ] && verify_only=1

fail=0
echo "$FILES" | while read -r path sum what; do
  [ -n "$path" ] || continue
  out="$DEST/$path"
  if [ "$verify_only" = 0 ] && [ ! -f "$out" ]; then
    mkdir -p "$(dirname "$out")"
    curl -sfL -o "$out.part" "$BASE/$path" || { echo "download failed: $path" >&2; rm -f "$out.part"; exit 1; }
    mv "$out.part" "$out"
  fi
  [ -f "$out" ] || { echo "missing:  $path" >&2; exit 1; }
  got=$(shasum -a 256 "$out" | cut -d' ' -f1)
  if [ "$got" != "$sum" ]; then
    echo "MISMATCH: $path (got $got)" >&2
    [ "$verify_only" = 0 ] && rm -f "$out"
    exit 1
  fi
  printf 'ok  %-28s %s\n' "$path" "$what"
done || fail=1

[ "$fail" = 0 ] || exit 1
echo "firmware in $DEST (linux-firmware $COMMIT)"
