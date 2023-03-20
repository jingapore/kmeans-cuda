CC = nvcc
SRCS = ./src/*.cpp ./src/*.cu
INC = ./src/
OPTS = -O2 -arch=sm_60

EXEC = bin/kmeans

all: clean compile

compile:
	$(CC) $(SRCS) $(OPTS) -I$(INC) -o $(EXEC)

clean:
	rm -f $(EXEC)
