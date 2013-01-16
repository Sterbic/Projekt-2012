FLAGS = -arch=sm_20

SRC_DIR = src
SRC := $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/*.cu)
HEADERS := $(wildcard $(SRC_DIR)/*.h $(SRC_DIR)/*.cuh)

SWalign: $(HEADERS) $(SRC)
	nvcc -o $@ $(SRC) $(FLAGS)

Debug: $(HEADERS) $(SRC)
	nvcc -o SWalign $(SRC) $(FLAGS) -g

Clean:
	rm -r temp/
	mkdir temp
