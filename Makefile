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
src/layers/input.cpp \
\
src/loss.cpp \
src/losses/categoricalcrossentropy.cpp \
\
src/optimizer.cpp \
src/optimizers/adam.cpp \
src/optimizers/sgd.cpp \
\
src/regularizer.cpp \
src/regularizers/l2.cpp \
\
src/activation.cpp \
src/activations/relu.cpp \
src/activations/leakyrelu.cpp \
src/activations/sigmoid.cpp \
\
src/callback.cpp \
src/callbacks/accuracy.cpp \
\
src/initializer.cpp \
src/initializers/randomnormal.cpp \

all:
	@g++ -g $(SOURCE) -o train

run: all
	./train