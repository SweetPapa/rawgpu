#!/bin/bash
# Build, sign and notarize a release from a clean copy of the tracked files:
#
#   dist/rawgpu-<version>-macos-arm64.tar.gz      the CLI, librawgpu, the C header and the Python binding
#   dist/rawgpu-app-<version>-macos-arm64.zip     the menu bar app, with the CLI inside
#   dist/SHA256SUMS
#
#   tools/package-release.sh 0.1.0
#
# The release workflow (.github/workflows/release.yml) runs exactly this. Locally:
#   SIGN_IDENTITY   a "Developer ID Application: ..." identity in the keychain. Without it the build is
#                   ad-hoc signed: it runs here, but Gatekeeper blocks it on other Macs.
#   ASC_KEY_PATH, ASC_KEY_ID, ASC_ISSUER_ID
#                   an App Store Connect API key, for notarization. Without them notarization is skipped.
#
# Release binaries never look in the folder they were built in (make RELEASE=1). They keep firmware and
# device libraries in ~/Library/Application Support/rawgpu, which the bundled scripts fill.
set -euo pipefail

V="${1:?usage: tools/package-release.sh VERSION (e.g. 0.1.0)}"
V="${V#v}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/dist"
NAME="rawgpu-$V"
TAURI="npx -y @tauri-apps/cli@2.12.4"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT
mkdir -p "$OUT" "$W/src"
say() { printf '\033[36m==>\033[0m %s\n' "$*"; }

if [ -n "${SIGN_IDENTITY:-}" ]; then
  security find-identity -v -p codesigning | grep -qF "\"$SIGN_IDENTITY\"" ||
    { echo "signing identity not in the keychain: $SIGN_IDENTITY" >&2; exit 1; }
  say "signing as $SIGN_IDENTITY"
else
  say "no SIGN_IDENTITY: ad-hoc signing (not for distribution)"
fi
notarize=0
if [ -n "${ASC_KEY_PATH:-}" ] && [ -n "${ASC_KEY_ID:-}" ] && [ -n "${ASC_ISSUER_ID:-}" ] && [ -n "${SIGN_IDENTITY:-}" ]; then
  notarize=1
fi

say "clean tree"
git -C "$ROOT" ls-files -z | (cd "$ROOT" && tar --null -cf - -T -) | tar -xf - -C "$W/src"
if [ -d "$ROOT/local/device-libs" ]; then # saves a download; checked by sha256 either way
  mkdir -p "$W/src/local" && cp -R "$ROOT/local/device-libs" "$W/src/local/"
fi

say "build (RELEASE=1, version $V)"
make -C "$W/src" RELEASE=1 VERSION="$V" -j"$(sysctl -n hw.ncpu)" all >"$W/build.log" 2>&1 ||
  { tail -40 "$W/build.log"; exit 1; }
"$W/src/build/rawgpu" --version

sign() { # Mach-O files, with the hardened runtime and a secure timestamp
  if [ -n "${SIGN_IDENTITY:-}" ]; then
    codesign --force --options runtime --timestamp --sign "$SIGN_IDENTITY" "$@"
  else
    codesign --force --sign - "$@"
  fi
  for f in "$@"; do codesign --verify --strict "$f"; done
}

notarize_zip() { # $1: a zip of what to notarize
  [ "$notarize" = 1 ] || { say "notarization skipped (no App Store Connect key)"; return 0; }
  say "notarizing $(basename "$1")"
  local out id
  out="$(xcrun notarytool submit "$1" --key "$ASC_KEY_PATH" --key-id "$ASC_KEY_ID" --issuer "$ASC_ISSUER_ID" \
         --wait --output-format json)"
  echo "$out"
  id="$(printf '%s' "$out" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
  if ! printf '%s' "$out" | grep -q '"status" *: *"Accepted"'; then
    xcrun notarytool log "$id" --key "$ASC_KEY_PATH" --key-id "$ASC_KEY_ID" --issuer "$ASC_ISSUER_ID" || true
    echo "notarization failed" >&2
    exit 1
  fi
}

say "CLI and library: $NAME"
P="$W/pkg/$NAME"
mkdir -p "$P/bin" "$P/lib" "$P/include" "$P/python" "$P/share/rawgpu" "$P/examples"
cp "$W/src/build/rawgpu" "$P/bin/"
cp "$W/src/build/librawgpu.dylib" "$W/src/build/librawgpu.a" "$P/lib/"
cp "$W/src/include/rawgpu.h" "$P/include/"
cp -R "$W/src/python/rawgpu" "$P/python/"
cp "$W/src/python/test_rawgpu.py" "$P/python/"
cp "$W/src/kernels/sgemm.cl" "$P/python/rawgpu/nn/"   # rawgpu.nn compiles its kernels with it
find "$P/python" -name __pycache__ -prune -exec rm -rf {} +
cp "$W/src/tools/fetch-firmware.sh" "$W/src/tools/fetch-device-libs.sh" "$W/src/tools/install-tinygpu.sh" "$P/share/rawgpu/"
cp "$W/src/examples/saxpy.c" "$P/examples/"
cp "$W/src/LICENSE" "$W/src/THIRD_PARTY_NOTICES.md" "$P/"
cat >"$P/README.md" <<EOF
# rawgpu $V for Apple Silicon Macs

Drive an AMD Polaris GPU (tested: RX 590) in a Thunderbolt enclosure from userspace, for compute.
Documentation: https://github.com/SweetPapa/rawgpu

1. Install tinygrad's TinyGPU driver (verified download), then approve it in System Settings ›
   General › Login Items & Extensions › Driver Extensions:
       share/rawgpu/install-tinygpu.sh
2. Download AMD's firmware (pinned and checked) into ~/Library/Application Support/rawgpu:
       share/rawgpu/fetch-firmware.sh
3. To compile OpenCL kernels, also: brew install llvm, and share/rawgpu/fetch-device-libs.sh
4. Check everything, then bring the card up:
       bin/rawgpu doctor
       bin/rawgpu up

Python: put python/ on your path (import rawgpu); it finds lib/librawgpu.dylib by itself.
C: include/rawgpu.h, link lib/librawgpu.a. Before unplugging the enclosure: bin/rawgpu eject.

rawgpu writes to the GPU's registers and firmware-managed power state. It has been tested on one
card. Use it at your own risk; see LICENSE (MIT, no warranty) and THIRD_PARTY_NOTICES.md.
EOF
sign "$P/bin/rawgpu" "$P/lib/librawgpu.dylib"
(cd "$W/pkg" && ditto -c -k --keepParent "$NAME" "$W/$NAME.zip")
notarize_zip "$W/$NAME.zip"
tar -C "$W/pkg" -czf "$OUT/$NAME-macos-arm64.tar.gz" "$NAME"

say "menu bar app"
T="$W/src/apps/rawgpu-tray"
mkdir -p "$T/binaries"
cp "$P/bin/rawgpu" "$T/binaries/rawgpu-aarch64-apple-darwin" # signed above; Tauri signs it again inside the app
python3 "$W/src/tools/rust-licenses.py" "$T" >"$T/binaries/THIRD_PARTY_RUST.md"
(
  cd "$T"
  export CARGO_TARGET_DIR="$W/target"
  if [ -n "${SIGN_IDENTITY:-}" ]; then export APPLE_SIGNING_IDENTITY="$SIGN_IDENTITY"; fi
  if [ "$notarize" = 1 ]; then
    export APPLE_API_KEY="$ASC_KEY_ID" APPLE_API_ISSUER="$ASC_ISSUER_ID" APPLE_API_KEY_PATH="$ASC_KEY_PATH"
  fi
  $TAURI build --bundles app --config "{\"version\":\"$V\"}" >"$W/app.log" 2>&1 || { tail -60 "$W/app.log"; exit 1; }
)
APP="$W/target/release/bundle/macos/rawgpu.app"
# Without an identity Tauri leaves the linker's signature, which does not cover the bundle's resources.
[ -n "${SIGN_IDENTITY:-}" ] || codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP"
if [ "$notarize" = 1 ]; then
  grep -iE "notari|stapl" "$W/app.log" || true
  # Tauri notarizes and staples the app itself, but once (v0.1.0's first run) it exited 0 with no ticket
  # stapled. Then notarize it here, and staple with retries: a new ticket can take a while to be served.
  if ! xcrun stapler validate "$APP" >/dev/null 2>&1; then
    say "the app has no notarization ticket: notarizing it here"
    (cd "$(dirname "$APP")" && ditto -c -k --keepParent rawgpu.app "$W/app-notarize.zip")
    notarize_zip "$W/app-notarize.zip"
    for i in 1 2 3 4 5 6; do xcrun stapler staple "$APP" && break; [ "$i" = 6 ] && exit 1; sleep 20; done
  fi
  xcrun stapler validate "$APP"
  spctl --assess --type execute -vv "$APP"
fi
(cd "$(dirname "$APP")" && ditto -c -k --keepParent rawgpu.app "$OUT/rawgpu-app-$V-macos-arm64.zip")

(cd "$OUT" && shasum -a 256 "$NAME-macos-arm64.tar.gz" "rawgpu-app-$V-macos-arm64.zip" >SHA256SUMS)
say "done:"
ls -la "$OUT"
cat "$OUT/SHA256SUMS"
