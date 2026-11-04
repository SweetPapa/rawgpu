#!/bin/bash
# Install tinygrad's TinyGPU driver extension, verified, from a pinned release.
#
# This is the same release tinygrad's own installer fetches
# (extra/hcq1/remote.py: APLRemotePCIDevice.ensure_app), but pinned by sha256
# and checked for tinygrad's Developer ID, notarization, and the PCI
# entitlement before anything touches /Applications.
#
#   tools/install-tinygpu.sh            verify, install, request activation
#   tools/install-tinygpu.sh --verify   verify only
#   tools/install-tinygpu.sh --uninstall
#
# Activation needs a one-time approval: System Settings > General >
# Login Items & Extensions > Driver Extensions > TinyGPU.
set -euo pipefail

COMMIT="c0d024f9ff0e1dc8fdf217f255da7101d91e8323"
URL="https://github.com/tinygrad/tinygpu_releases/raw/${COMMIT}/TinyGPU.zip"
SHA256="0c47285e2232643210555cf30ce08289b9e55da261c300e0c82e8448a359a21f"
TEAM="9YG3G8543N" # tinygrad, Corp.
DEXT_ID="org.tinygrad.tinygpu.driver2"
APP="/Applications/TinyGPU.app"

if [[ "${1:-}" == "--uninstall" ]]; then
  [[ -x "$APP/Contents/MacOS/TinyGPU" ]] && "$APP/Contents/MacOS/TinyGPU" uninstall || true
  echo "After the extension is gone (systemextensionsctl list), remove $APP."
  exit 0
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
echo "==> downloading TinyGPU @ ${COMMIT:0:7}"
curl -fsSL -o "$work/TinyGPU.zip" "$URL"
got="$(shasum -a 256 "$work/TinyGPU.zip" | awk '{print $1}')"
[[ "$got" == "$SHA256" ]] || { echo "sha256 mismatch: $got" >&2; exit 1; }
echo "    sha256 OK"

mkdir "$work/x"
ditto -xk "$work/TinyGPU.zip" "$work/x"
cand="$work/x/TinyGPU.app"
dext="$cand/Contents/Library/SystemExtensions/$DEXT_ID.dext"

codesign --verify --deep --strict "$cand"
gk="$(spctl --assess --type execute -vv "$cand" 2>&1)" || { echo "Gatekeeper rejected the app: $gk" >&2; exit 1; }
grep -q "source=Notarized Developer ID" <<<"$gk" || { echo "not notarized: $gk" >&2; exit 1; }
for b in "$cand" "$dext"; do
  codesign -dvv "$b" 2>&1 | grep -q "TeamIdentifier=$TEAM" || { echo "unexpected signer on $b" >&2; exit 1; }
done
ents="$(codesign -d --entitlements :- "$dext" 2>/dev/null)"
grep -q "0x00001002&amp;0x0000FFFF\|0x00001002&0x0000FFFF" <<<"$ents" || { echo "dext lacks the AMD PCI entitlement" >&2; exit 1; }
echo "    signature, notarization, team $TEAM and AMD PCI entitlement OK"

[[ "${1:-}" == "--verify" ]] && exit 0

if [[ -e "$APP" ]]; then
  echo "==> $APP already exists; leaving it in place"
else
  echo "==> installing to $APP"
  ditto "$cand" "$APP"
fi
"$APP/Contents/MacOS/TinyGPU" install || true
"$APP/Contents/MacOS/TinyGPU" status
