# Releasing

A release is two signed and notarized files, plus their checksums:

| file | what |
|---|---|
| `rawgpu-<v>-macos-arm64.tar.gz` | `bin/rawgpu`, `lib/librawgpu.{a,dylib}`, `include/rawgpu.h`, `python/rawgpu`, `share/rawgpu/*.sh`, the example, `LICENSE` and the notices |
| `rawgpu-app-<v>-macos-arm64.zip` | `rawgpu.app`, the menu bar app, with the CLI and the setup scripts inside it (stapled) |
| `SHA256SUMS` | checksums of the two files above |

Code Search and rawgpu Studio are not in releases; both run from a source
checkout. Code Search needs PyTorch, a venv and an index, and is built with
`make install-apps`. Studio is started with `tools/studio.sh` (see
[TRAINING.md](TRAINING.md)). `rawgpu.nn`, the library Studio trains with, is
in the tarball.

## Cutting one

1. **On the card, by hand:** CI has no GPU, so check the hardware before
   tagging.
   ```sh
   make && make kernels
   ./build/rawgpu doctor
   ./build/rawgpu up && ./build/rawgpu kernel-test
   python3 python/test_rawgpu.py
   local/eg2-venv/bin/python models/embeddinggemma2/compare.py   # if the model changed
   ./build/rawgpu eject
   ```
2. **Tag and push:**
   ```sh
   git tag -a v0.1.0 -m "rawgpu 0.1.0" && git push origin v0.1.0
   ```
3. **The Release workflow** (`.github/workflows/release.yml`) does the rest:
   - builds with `make RELEASE=1`, so the binaries never look in the folder
     they were built in;
   - signs with the hardened runtime, as **Developer ID Application:
     Forrester Terry (6Y5SZ2K5XY)**. Any other team stops the run;
   - notarizes with the App Store Connect API key, and staples the app;
   - creates a **draft** release with install notes and the commits since
     the last tag.
4. **Read the draft on GitHub, then publish it.** A version with a hyphen
   (`0.2.0-rc1`) is marked as a prerelease.

**A dry run:** start the Release workflow from the Actions tab with a
version such as `0.0.0-test`. It builds, signs and notarizes, keeps the files
as the run's artifacts, and creates no release unless *publish* is ticked.

## Locally

The workflow runs `tools/package-release.sh`, and so can you:

```sh
make release V=0.1.0        # ad-hoc signed: fine on this Mac, blocked by Gatekeeper elsewhere
SIGN_IDENTITY="Developer ID Application: Forrester Terry (6Y5SZ2K5XY)" \
ASC_KEY_PATH=~/AuthKey_XXXX.p8 ASC_KEY_ID=XXXX ASC_ISSUER_ID=... make release V=0.1.0
```

It builds from a copy of the tracked files, so commit (or `git add`) first.
The output goes in `dist/`.

## Secrets

These are set on the repository by FloppyJam's
`scripts/setup-ci.sh --repo SweetPapa/rawgpu --apple --notary`, from the
`.p12` named in FloppyJam's `.env` and the `asc-mcp` keychain item:

| secret | what |
|---|---|
| `APPLE_CERT_P12`, `APPLE_CERT_PASSWORD` | the Developer ID Application certificate, base64, and its password |
| `APPLE_TEAM_ID` | `6Y5SZ2K5XY` |
| `ASC_KEY_P8`, `ASC_KEY_ID`, `ASC_ISSUER_ID` | the App Store Connect API key, for notarization |

Without them the workflow still builds, ad-hoc signed and not notarized, and
warns.

## CI

`.github/workflows/ci.yml` builds the CLI, library, kernels, example and both
apps. It also runs the no-GPU smoke tests and checks that a release build
holds no path to its build folder. macOS minutes cost 10x on a private
repository, so CI runs only when started by hand while the repository is
private, and on every push and pull request once it is public.
