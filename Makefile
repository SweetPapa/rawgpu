CC ?= clang
CFLAGS ?= -O2 -g -std=gnu11 -Wall -Wextra -Wno-unused-parameter
# everything but the CLI's main.c goes into the library too
LIB_SRC := src/transport_tinygpu.c src/pci.c src/family.c src/amd_atom.c src/amd_fw.c \
       src/amd_vi.c src/amd_vi_gmc.c src/amd_vi_sdma.c src/amd_vi_smu.c src/amd_vi_gfx.c src/amd_kernel.c \
       src/amd_vi_ctx.c src/amd_vi_bench.c src/amd_vi_dpm.c src/rg_ranges.c src/amd_vi_ktest.c src/rawgpu_lib.c
SRC := src/main.c $(LIB_SRC)
HDR := src/rg.h src/amd_ids.inc src/atom/rg_atom.h src/amd_fw.h src/amd_vi.h src/amd_vi_regs.h src/amd_vi_regnames.inc \
       src/upstream/vi_structs.h src/amd_kernel.h src/amd_vi_ctx.h src/pp/rg_pp.h src/rg_ranges.h include/rawgpu.h
OBJ_DIR := build/obj
LIB_OBJ := $(addprefix $(OBJ_DIR)/,$(notdir $(LIB_SRC:.c=.o))) $(OBJ_DIR)/embedded.o
# where the library finds the kernel compiler and this checkout's device libraries and firmware by default
# A release build (make RELEASE=1) never looks in the folder it was built in: data comes from
# $RAWGPU_HOME or ~/Library/Application Support/rawgpu (see rg_data_dir() in src/amd_fw.c).
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null | sed 's/^v//' || echo dev)
ifeq ($(RELEASE),1)
DEFAULT_LOCAL :=
else
DEFAULT_LOCAL := $(CURDIR)/local
endif
LIB_DEFS = -DRAWGPU_DEFAULT_CLANG='"$(AMDGPU_CLANG)"' -DRAWGPU_DEFAULT_LOCAL='"$(DEFAULT_LOCAL)"' -DRAWGPU_VERSION='"$(VERSION)"'

all: build/rawgpu build/librawgpu.a build/librawgpu.dylib

$(OBJ_DIR)/%.o: src/%.c $(HDR)
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -Iinclude -Isrc $(LIB_DEFS) -c -o $@ $<

# the built-in kernels (rawgpu_sgemm()), compiled to gfx803 and embedded in the library
build/gen/embedded.c: build/kernels/sgemm.o tools/embed.py
	@mkdir -p build/gen
	python3 -I tools/embed.py $@ sgemm=build/kernels/sgemm.o

$(OBJ_DIR)/embedded.o: build/gen/embedded.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# amdgpu's ATOM interpreter, compiled unmodified against the kernel shims.
# -fno-strict-aliasing and -Wno-pointer-sign match how the kernel builds it.
ATOM_SRC := src/atom/upstream/atom.c src/atom/rg_atom.c
ATOM_CFLAGS := -O2 -g -std=gnu11 -fno-strict-aliasing -Wno-pointer-sign \
               -Isrc/atom/shim -Isrc/atom/upstream -Isrc/atom
ATOM_OBJ := build/atom.o build/rg_atom.o

# amdgpu's SMU7/Polaris powerplay code, compiled unmodified against kernel shims
# (src/pp/UPSTREAM.md), plus rawgpu's glue (src/pp/rg_pp*.c). The flags match how
# the kernel builds it (-fno-strict-aliasing, -Wno-pointer-sign).
PP := src/pp
PP_UP := $(PP)/upstream
PP_C := $(wildcard $(PP_UP)/hwmgr/*.c) $(wildcard $(PP_UP)/smumgr/*.c) $(PP)/rg_pp.c $(PP)/rg_pp_stubs.c
PP_H := $(wildcard $(PP_UP)/*/*.h $(PP_UP)/include/*/*.h $(PP_UP)/include/*/*/*.h $(PP)/shim/*.h $(PP)/shim/*/*.h $(PP)/asic_reg/*/*.h) \
        $(PP)/rg_pp.h src/atom/upstream/atom.h
PP_CFLAGS := -O2 -g -std=gnu11 -fno-strict-aliasing -Wall -Wno-unused-parameter -Wno-pointer-sign -Wno-unused-variable \
             -I$(PP) -I$(PP)/shim -I$(PP_UP)/hwmgr -I$(PP_UP)/smumgr -I$(PP_UP)/inc -I$(PP_UP)/include \
             -I$(PP_UP)/legacy-dpm -Isrc/atom/upstream -I$(PP)/asic_reg
PP_OBJ := $(addprefix build/pp/,$(notdir $(PP_C:.c=.o)))
vpath %.c $(PP_UP)/hwmgr $(PP_UP)/smumgr $(PP)

build/pp/%.o: %.c $(PP_H)
	@mkdir -p build/pp
	$(CC) $(PP_CFLAGS) -c -o $@ $<

build/rawgpu: $(OBJ_DIR)/main.o $(LIB_OBJ) $(ATOM_OBJ) $(PP_OBJ)
	$(CC) $(CFLAGS) -o $@ $^

build/librawgpu.a: $(LIB_OBJ) $(ATOM_OBJ) $(PP_OBJ)
	rm -f $@
	ar rcs $@ $^

build/librawgpu.dylib: $(LIB_OBJ) $(ATOM_OBJ) $(PP_OBJ)
	$(CC) -dynamiclib -install_name @rpath/librawgpu.dylib -o $@ $^

build/atom.o: src/atom/upstream/atom.c src/atom/shim/rg_kcompat.h
	@mkdir -p build
	$(CC) $(ATOM_CFLAGS) -c -o $@ $<

build/rg_atom.o: src/atom/rg_atom.c src/atom/rg_atom.h src/atom/rg_atom_names.inc src/atom/shim/rg_kcompat.h src/rg.h
	@mkdir -p build
	$(CC) $(ATOM_CFLAGS) -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -c -o $@ $<

# Regenerate the AMD ID table: make ids AMDGPU_DRV=/path/to/amdgpu_drv.c LINUX_SHA=<sha>
ids:
	python3 -I tools/gen_amd_ids.py $(AMDGPU_DRV) $(LINUX_SHA) > src/amd_ids.inc

# Regenerate the VI register subset: make regs LINUX_AMD=/path/with/asic_reg/headers LINUX_SHA=<sha>
regs:
	python3 -I tools/gen_vi_regs.py $(LINUX_AMD) $(LINUX_SHA) src/amd_vi_regnames.inc > src/amd_vi_regs.h

atom-names:
	python3 -I tools/gen_atom_names.py src/atom/upstream/atombios.h > src/atom/rg_atom_names.inc

# GPU kernels, compiled for gfx803 (Polaris) with an LLVM that has the AMDGPU
# target (Homebrew's llvm does; Apple's clang does not). Clang links AMD's
# device libraries (the OpenCL built-ins, tools/fetch-device-libs.sh) into
# each object at compile time, so no linker is needed: rawgpu loads the
# relocatable object directly.
AMDGPU_CLANG ?= /opt/homebrew/opt/llvm/bin/clang
DEVICE_LIBS := local/device-libs
# code object v5, explicitly: LLVM 21 and later default to v6, which the pinned ROCm 6.0 device libraries
# predate (they have oclc_abi_version_400/500 only)
AMDGPU_CFLAGS := -x cl -cl-std=CL1.2 -target amdgcn-amd-amdhsa -mcpu=gfx803 -mcode-object-version=5 -O2 -cl-kernel-arg-info \
                 --rocm-device-lib-path=$(DEVICE_LIBS)
KERNELS := $(patsubst kernels/%.cl,build/kernels/%.o,$(wildcard kernels/*.cl))

kernels: $(KERNELS)

$(DEVICE_LIBS)/opencl.bc:
	tools/fetch-device-libs.sh

build/kernels/%.o: kernels/%.cl $(DEVICE_LIBS)/opencl.bc
	@mkdir -p build/kernels
	$(AMDGPU_CLANG) $(AMDGPU_CFLAGS) -c -o $@ $<

# examples against the static library
examples: build/examples/saxpy

build/examples/%: examples/%.c build/librawgpu.a include/rawgpu.h
	@mkdir -p build/examples
	$(CC) $(CFLAGS) -Iinclude -o $@ $< build/librawgpu.a

# the menu bar app and Code Search (apps/README.md); Tauri's CLI comes through npx, pinned
TAURI := npx -y @tauri-apps/cli@2.12.4
APP_DIR := apps/target/release/bundle/macos
apps: build/rawgpu
	mkdir -p apps/rawgpu-tray/binaries
	cp build/rawgpu apps/rawgpu-tray/binaries/rawgpu-aarch64-apple-darwin
	python3 tools/rust-licenses.py apps/rawgpu-tray > apps/rawgpu-tray/binaries/THIRD_PARTY_RUST.md
	cd apps/rawgpu-tray && $(TAURI) build --bundles app
	cd apps/code-search && $(TAURI) build --bundles app

# a signed, notarized release into dist/ (docs/RELEASING.md): make release V=0.1.0
release:
	tools/package-release.sh $(V)

install-apps: apps
	ditto "$(APP_DIR)/rawgpu.app" "$(HOME)/Applications/rawgpu.app"
	ditto "$(APP_DIR)/Code Search.app" "$(HOME)/Applications/Code Search.app"
	@# unsigned Tauri bundles carry only the linker's signature, which leaves out their resources
	codesign --force --deep --sign - "$(HOME)/Applications/rawgpu.app" "$(HOME)/Applications/Code Search.app"

clean:
	rm -rf build

.PHONY: all ids regs atom-names kernels examples apps install-apps release clean
