SOURCE= \
src/train.cpp \
src/matrix.cpp \
src/dataset.cpp \
src/model.cpp \
\
src/layer.cpp \
src/layers/conv.cpp \
src/layers/dense.cpp \
src/layers/avgpool.cpp \
src/layers/flatten.cpp \
\
src/loss.cpp \
src/losses/categoricalcrossentropy.cpp \
\
src/optimizer.cpp \
src/optimizers/adam.cpp \
\
src/activation.cpp \
src/activations/relu.cpp \
src/activations/softmax.cpp \
\
src/callback.cpp \
src/callbacks/accuracy.cpp \
\

all:
	@g++ $(SOURCE) -o train

run: all
	./train