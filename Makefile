SOURCE= \
src/train.cpp \
src/matrix.cpp \
src/dataset.cpp \
src/model.cpp \
src/utils.cpp \
\
src/layer.cpp \
src/layers/conv.cpp \
src/layers/dense.cpp \
src/layers/avgpool.cpp \
src/layers/flatten.cpp \
src/layers/softmax.cpp \
\
src/loss.cpp \
src/losses/categoricalcrossentropy.cpp \
\
src/optimizer.cpp \
src/optimizers/adam.cpp \
\
src/activation.cpp \
src/activations/relu.cpp \
src/activations/sigmoid.cpp \
\
src/callback.cpp \
src/callbacks/accuracy.cpp \
\

all:
	@g++ -g $(SOURCE) -o train

run: all
	./train