# Third-party notices

rawgpu's own code is under the MIT license in `LICENSE`. It includes, or
is derived from, the code below. Each keeps its own license.

## Linux amdgpu (Advanced Micro Devices, Inc.), MIT

- **Copied byte for byte**, with AMD's copyright and permission notice kept in
  each file:
  - `src/atom/upstream/`: the ATOM BIOS interpreter and tables;
  - `src/pp/upstream/`: the SMU7 / Polaris powerplay code;
  - `src/upstream/`: amdgpu structure headers.
- **Generated** by `tools/gen_vi_regs.py` and `tools/gen_pp_regs.py` from
  amdgpu's register headers:
  - `src/amd_vi_regs.h`;
  - `src/pp/asic_reg/`.

All of it comes from torvalds/linux at
`05441d697f7f9a927264f4058cbbc6a03dfcac9d`. The notice the files carry:

```
Copyright 2008 Advanced Micro Devices, Inc. (and later years, per file)

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
```

## tinygrad (the tiny corp), MIT

- **`src/transport_tinygpu.c`** implements the wire protocol of the server in
  tinygrad's TinyGPU.app (`extra/usbgpu/tbgpu/installer/Shared/server.c`).
- **TinyGPU itself** is not included. `tools/install-tinygpu.sh` downloads
  tinygrad's signed release and checks it.

```
Copyright (c) 2024, the tiny corp

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

```

## highlight.js, BSD 3-Clause

`apps/code-search/ui/vendor/highlight.min.js` is highlight.js 11.11.1. Its
license is in `apps/code-search/ui/vendor/LICENSE`:

```
BSD 3-Clause License

Copyright (c) 2006, Ivan Sagalaev.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## The apps' Rust dependencies

The menu bar app and Code Search are built with Tauri and other Rust crates,
under MIT and/or Apache-2.0. A release lists them in
`THIRD_PARTY_RUST.md`, generated by `tools/rust-licenses.py`.

## Downloaded at setup, not included

These are fetched by scripts, pinned by sha256, and stay under their own
terms:
- **AMD GPU firmware:** linux-firmware, `LICENSE.amdgpu`. It may be
  redistributed but not reverse engineered. Fetched by
  `tools/fetch-firmware.sh`.
- **ROCm-Device-Libs bitcode:** NCSA. Fetched by
  `tools/fetch-device-libs.sh`.
- **TinyGPU.app:** tinygrad's release. Installed by
  `tools/install-tinygpu.sh`.
- **The EmbeddingGemma 2 weights:** Apache-2.0. Downloaded by
  Hugging Face for `models/embeddinggemma2/`.
