FLAGS = -arch=sm_20

SRC_DIR = src
SRC := $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/*.cu)

SWalign: $(SRC)
	nvcc -o $@ $+ $(FLAGS)
