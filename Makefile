all:
	g++ src/train.cpp src/matrix.cpp -o train

run: all
	./train