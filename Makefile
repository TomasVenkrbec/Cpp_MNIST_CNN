SOURCE= \
src/train.cpp \
src/matrix.cpp \
src/dataset.cpp \
src/model.cpp \
\
src/layer.cpp \
src/layers/convlayer.cpp \
src/layers/denselayer.cpp \
src/layers/avgpoollayer.cpp \
\
src/loss.cpp \
src/losses/categoricalcrossentropy.cpp \
\
src/optimizer.cpp \
src/optimizers/adam.cpp \

all:
	@g++ $(SOURCE) -o train

run: all
	./train