CC ?= clang
CFLAGS ?= -O2 -g -std=gnu11 -Wall -Wextra -Wno-unused-parameter
SRC := src/main.c src/transport_tinygpu.c src/pci.c src/family.c src/amd_vi.c src/amd_atom.c
HDR := src/rg.h src/amd_ids.inc src/atom/rg_atom.h

# amdgpu's ATOM interpreter, compiled unmodified against the kernel shims.
# -fno-strict-aliasing and -Wno-pointer-sign match how the kernel builds it.
ATOM_SRC := src/atom/upstream/atom.c src/atom/rg_atom.c
ATOM_CFLAGS := -O2 -g -std=gnu11 -fno-strict-aliasing -Wno-pointer-sign \
               -Isrc/atom/shim -Isrc/atom/upstream -Isrc/atom
ATOM_OBJ := build/atom.o build/rg_atom.o

build/rawgpu: $(SRC) $(HDR) $(ATOM_OBJ)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $(SRC) $(ATOM_OBJ)

build/atom.o: src/atom/upstream/atom.c src/atom/shim/rg_kcompat.h
	@mkdir -p build
	$(CC) $(ATOM_CFLAGS) -c -o $@ $<

build/rg_atom.o: src/atom/rg_atom.c src/atom/rg_atom.h src/atom/rg_atom_names.inc src/atom/shim/rg_kcompat.h src/rg.h
	@mkdir -p build
	$(CC) $(ATOM_CFLAGS) -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -c -o $@ $<

# Regenerate the AMD ID table: make ids AMDGPU_DRV=/path/to/amdgpu_drv.c LINUX_SHA=<sha>
ids:
	python3 -I tools/gen_amd_ids.py $(AMDGPU_DRV) $(LINUX_SHA) > src/amd_ids.inc

atom-names:
	python3 -I tools/gen_atom_names.py src/atom/upstream/atombios.h > src/atom/rg_atom_names.inc

clean:
	rm -rf build

.PHONY: ids atom-names clean
