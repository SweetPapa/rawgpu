#!/bin/sh
# Syntax-check the vendored powerplay files against the shims; print the first errors of each.
cd "$(dirname "$0")/.."
U=src/pp/upstream
INC="-Isrc/pp/shim -I$U/hwmgr -I$U/smumgr -I$U/inc -I$U/include -I$U/legacy-dpm -Isrc/atom/upstream -I${PP_ASIC_REG:-src/pp/asic_reg}"
for f in ${@:-$U/hwmgr/*.c $U/smumgr/*.c}; do
  out=$(clang -std=gnu11 -fsyntax-only -Wno-everything $INC "$f" 2>&1 | grep -E 'error' | head -${PP_N:-4})
  [ -n "$out" ] && { echo "== $f"; echo "$out"; }
done
true
