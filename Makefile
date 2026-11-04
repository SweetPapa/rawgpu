CC ?= clang
CFLAGS ?= -O2 -g -std=gnu11 -Wall -Wextra -Wno-unused-parameter
SRC := src/main.c src/transport_tinygpu.c src/pci.c src/family.c src/amd_vi.c src/amd_atom.c
HDR := src/rg.h src/amd_ids.inc

build/rawgpu: $(SRC) $(HDR)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $(SRC)

# Regenerate the AMD ID table: make ids AMDGPU_DRV=/path/to/amdgpu_drv.c LINUX_SHA=<sha>
ids:
	python3 -I tools/gen_amd_ids.py $(AMDGPU_DRV) $(LINUX_SHA) > src/amd_ids.inc

clean:
	rm -rf build

.PHONY: ids clean
