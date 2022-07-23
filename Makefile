all:
	g++ src/train.cpp src/matrix.cpp src/dataset.cpp src/layers.cpp -o train

run: all
	./train