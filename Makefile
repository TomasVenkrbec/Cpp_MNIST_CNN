all: train

train:
	g++ src/train.cpp src/matrix.cpp src/dataset.cpp -o train

run: all
	./train