#!/bin/sh
# Download AMD's device libraries (ROCm-Device-Libs) as LLVM bitcode, into
# local/device-libs/ (gitignored).
#
# OpenCL kernels call built-in functions (get_global_id(), barrier(), sin(),
# vload4(), atomics...) that live in these libraries. Clang links them into
# each kernel at compile time (--rocm-device-lib-path), so a code object for
# rawgpu is self-contained and needs no linker.
#
# The bitcode comes from Debian's rocm-device-libs-17 package, built from
# ROCm-Device-Libs 6.0 (git 5a852ed) with upstream LLVM 17. Newer clang reads
# it (Homebrew LLVM 20 is tested). The package's sha256 is pinned here, as are
# the files clang uses (it also insists hip.bc exists, though OpenCL does not
# link it). License: NCSA (LICENSE in the same directory).
#
# usage: tools/fetch-device-libs.sh [--verify]
set -eu

DEB=rocm-device-libs-17_6.0+git20231212.5a852ed-2_arm64.deb
URL="http://deb.debian.org/debian/pool/main/r/rocm-device-libs/$DEB"
DEB_SHA=7993b4ed1de284c06ab4a7cdcbb2c1f337f7fba6f1d7ed7ebaed993de0800b2c
INNER=usr/lib/llvm-17/lib/clang/17/amdgcn/bitcode
DEST="$(cd "$(dirname "$0")/.." && pwd)/local/device-libs"

FILES='
opencl.bc                          399348551ba1d43f235a61bb2a580eb11557be466447199b47f9b428345eddfb
ocml.bc                            c20619ba6999071851b870962622507425698d744b9a11deed3029f0f935203f
ockl.bc                            7fa0f3497b9301fcb3bdecc38160a4e384e17dd7dfcd15c7d68b18a910feeb83
oclc_isa_version_803.bc            894942fb7305d312d9ea3c8a6b749bdd0105cbaf54a2b258ec4e8197740f55f0
oclc_abi_version_400.bc            42f3e51e2759ab3a9b09c1a2360e0741f0fc0c6f92c050ab6da9a898a130442b
oclc_abi_version_500.bc            8c07fcca567f702d02512b7c2d68d372414ed0dd213ef57f67993d5e14cea979
oclc_correctly_rounded_sqrt_off.bc f552fcd370648274a995ab700dd05043532be6e3073d734f7c4ccd9ec8536ae8
oclc_correctly_rounded_sqrt_on.bc  2bf8d5d2d7cf4e6b6991dacb92083d4ba873b51d7fc3ad558176fbbceed2325f
oclc_daz_opt_off.bc                d02acaab76f73027fcb0968672bfa7fc0b445fb97688548a76b69f88ac4619d7
oclc_daz_opt_on.bc                 873cbac386754c467aa19d10cc3f62b871b97d4dddda0effe52685c54734770b
oclc_finite_only_off.bc            32bdd9f02da8a4da277dc74a7e3cad4d4da617f2940397382e7a2ca002f8abb1
oclc_finite_only_on.bc             f8b389933f3979d8741c475c33924bb3572f451d84abfcf3aa907e70ae2beddb
oclc_unsafe_math_off.bc            9c3fb521ad516b7642548712025b2f198b438b3edda75efeb7eb7b78f0904de4
oclc_unsafe_math_on.bc             7c91d425f60663a3fd8b5ce9e986b8701c3033caf85083043bf8b39292be3e0c
oclc_wavefrontsize64_off.bc        d04a87058023b8e9bbee9ac1cc8a2177d278a66f71d65bdf800f6699ba6a0338
oclc_wavefrontsize64_on.bc         9be6a4371fbd93c31bd0c3dd62b630632f35601f6d52c4f7f59c39e81596c551
hip.bc                             2e2e37f279e94e88ea2c36587ff1aeb7b039694d11c019940be1ad8839329a8a
'

check() { # every file present with its pinned sha256
  echo "$FILES" | while read -r f sum; do
    [ -n "$f" ] || continue
    [ -f "$DEST/$f" ] || { echo "missing:  $f" >&2; exit 1; }
    got=$(shasum -a 256 "$DEST/$f" | cut -d' ' -f1)
    [ "$got" = "$sum" ] || { echo "MISMATCH: $f (got $got)" >&2; exit 1; }
  done
}

if [ "${1:-}" = "--verify" ]; then
  check && echo "device libraries in $DEST verified"
  exit
fi
if check 2>/dev/null; then echo "device libraries already in $DEST"; exit 0; fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -sfL -o "$tmp/$DEB" "$URL" || { echo "download failed: $URL" >&2; exit 1; }
got=$(shasum -a 256 "$tmp/$DEB" | cut -d' ' -f1)
[ "$got" = "$DEB_SHA" ] || { echo "MISMATCH: $DEB (got $got)" >&2; exit 1; }
mkdir -p "$tmp/deb" "$tmp/root" "$DEST"
(cd "$tmp/deb" && tar -xf "../$DEB")
tar -xf "$tmp/deb"/data.tar.* -C "$tmp/root"
echo "$FILES" | while read -r f sum; do
  if [ -n "$f" ]; then cp "$tmp/root/$INNER/$f" "$DEST/$f"; fi
done
cp "$tmp/root/usr/share/doc/rocm-device-libs-17/copyright" "$DEST/LICENSE"
check
echo "device libraries in $DEST (ROCm-Device-Libs 6.0, Debian $DEB)"
