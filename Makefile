FLAGS = -arch=sm_20

SRC_DIR = src
SRC := $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/*.cu)
HEADERS := $(wildcard $(SRC_DIR)/*.h $(SRC_DIR)/*.cuh)

SWalign: clean $(HEADERS) $(SRC)
	nvcc -o $@ $(SRC) $(FLAGS)

debug: clean $(HEADERS) $(SRC)
	nvcc -o SWalign $(SRC) $(FLAGS) -g

clean:
	rm -r temp/
	mkdir temp
