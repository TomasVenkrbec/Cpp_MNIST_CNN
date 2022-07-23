SOURCE= \
src/train.cpp \
src/matrix.cpp \
src/dataset.cpp \
src/layer.cpp \
src/model.cpp \
src/layers/convlayer.cpp \
src/layers/denselayer.cpp \

all:
	@g++ $(SOURCE) -o train

run: all
	./train