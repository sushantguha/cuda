# =============================================================================
# week5_frameforge — GPU image preprocessor for a simulated FSD camera stream
#
# Build (Release, default):   make
# Build (Debug + cuda-gdb):    make CONFIG=debug
# Run:                         make run     (or  ./build/frameforge)
# Memcheck:                    make memcheck
# Racecheck (Day 4+):          make racecheck
# Clean:                       make clean
#
# Source layout this Makefile understands:
#   ./main.cu                (current location)
#   ./src/*.cu  ./src/*.cpp  (Day 2+ — added automatically as files appear)
#   ./include/*.hpp ./include/*.cuh
# =============================================================================

# ---------- toolchain ----------
NVCC      := nvcc
CXX       := g++

# ---------- project layout ----------
PROJECT   := frameforge
BUILD_DIR := build
SRC_DIRS  := . src
INC_DIRS  := include

# ---------- source discovery (auto-picks up src/*.cu and src/*.cpp) ----------
CU_SRCS   := $(wildcard $(addsuffix /*.cu,  $(SRC_DIRS)))
CPP_SRCS  := $(wildcard $(addsuffix /*.cpp, $(SRC_DIRS)))

OBJS      := $(CU_SRCS:%.cu=$(BUILD_DIR)/%.cu.o) \
             $(CPP_SRCS:%.cpp=$(BUILD_DIR)/%.cpp.o)
DEPS      := $(OBJS:.o=.d)

# ---------- config: release (default) vs debug ----------
CONFIG    ?= release

ifeq ($(CONFIG),debug)
  HOST_OPT  := -O0 -g
  NVCC_OPT  := -O0 -g -G
else
  HOST_OPT  := -O3
  NVCC_OPT  := -O3 -lineinfo
endif

# ---------- flags ----------
# Target arch. Default is sm_75 = Google Colab's T4 (Turing).
#
# Why not `-arch=native`?  Locally this box has a GTX 780 (Kepler, sm_35) which
# CUDA 13 no longer supports — and `-arch=native` falls back to compute_35 when
# it can't detect a usable GPU, which then fails the compile. Local builds here
# are *syntax checks only*; the produced binary is meant to run on Colab.
#
# Override from the command line if you target a different Colab GPU:
#   make ARCH="-arch=sm_80"   # A100 (Colab Pro)
#   make ARCH="-arch=sm_89"   # L4   (newer Colab runtimes)
ARCH      ?= -arch=sm_75

INCFLAGS  := $(addprefix -I, $(INC_DIRS))

CXXFLAGS  := -std=c++20 $(HOST_OPT) $(INCFLAGS) -Wall -Wextra -MMD -MP
NVCCFLAGS := -std=c++17 $(NVCC_OPT) $(INCFLAGS) $(ARCH) \
             -Xcompiler "-Wall -Wextra" -MMD -MP

LDFLAGS   :=
LDLIBS    := -lcudart

# ---------- top-level targets ----------
.PHONY: all run memcheck racecheck clean print

all: $(BUILD_DIR)/$(PROJECT)

$(BUILD_DIR)/$(PROJECT): $(OBJS)
	@mkdir -p $(@D)
	$(NVCC) $(ARCH) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# ---------- pattern rules ----------
$(BUILD_DIR)/%.cu.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cpp.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ---------- convenience ----------
run: all
	./$(BUILD_DIR)/$(PROJECT)

memcheck: all
	compute-sanitizer --tool memcheck  ./$(BUILD_DIR)/$(PROJECT)

racecheck: all
	compute-sanitizer --tool racecheck ./$(BUILD_DIR)/$(PROJECT)

clean:
	rm -rf $(BUILD_DIR)

# `make print` — dump what was discovered. Handy when builds look wrong.
print:
	@echo "CU_SRCS  = $(CU_SRCS)"
	@echo "CPP_SRCS = $(CPP_SRCS)"
	@echo "OBJS     = $(OBJS)"
	@echo "CONFIG   = $(CONFIG)"
	@echo "ARCH     = $(ARCH)"

# auto-generated header dependencies
-include $(DEPS)
