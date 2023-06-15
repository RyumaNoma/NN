#include "NeuralNetwork.hpp"
#include "Common.hpp"
// layer
#include "Dense.hpp"
#include "ReLU.hpp"
#include "SoftmaxCrossEntropyError.hpp"
// optimizer
#include "SGD.hpp"
#include "mnist/mnist_reader.hpp"
#define MNIST_DATA_LOCATION "./mnist/dataFashion/"

// TODO : experiment : äàê´âªå„ÇÃï™ïzämîF

int main(int argc, char* argv[]) {
    // MNIST_DATA_LOCATION set by MNIST cmake config
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    std::mt19937 engine;
    std::shared_ptr<optimizer::SGD> sgd(new optimizer::SGD());
    std::shared_ptr<layer::Dense> dense1(new layer::Dense(
        784,
        128,
        Initializer::Type::Xavier,
        engine,
        sgd,
        sgd));
    std::shared_ptr<layer::ReLU> relu(new layer::ReLU());
    std::shared_ptr<layer::Dense> dense2(new layer::Dense(
        128,
        10,
        Initializer::Type::Xavier,
        engine,
        sgd,
        sgd));
    std::shared_ptr<loss_layer::SoftmaxCrossEntropyError> scee(new loss_layer::SoftmaxCrossEntropyError());
    NeuralNetwork model;
    model.Add(dense1);
    model.Add(relu);
    model.Add(dense2);
    model.SetLossLayer(scee);

    Matrix input(5, 784);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 784; ++j) {
            input(i, j) = static_cast<double>(dataset.training_images[i][j]);
        }
    }
    input /= 255.0;

    Matrix answer(5, 1);
    for (int i = 0; i < 5; ++i) {
        answer(i, 0) = dataset.training_labels[i];
    }

    double loss = model.Inference(input, Common::onehot(answer, 10));
    std::cout << loss << std::endl;

    try {
        model.Gradient();
    }
    catch (std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    //for (int i = 0; i < pred.Row(); ++i) {
    //    double max = -1;
    //    int maxIdx = -1;
    //    for (int j = 0; j < pred.Col(); ++j) {
    //        if (pred(i, j) > max) {
    //            max = pred(i, j);
    //            maxIdx = j;
    //        }
    //    }
    //    std::cerr << "case" << i << ": " << maxIdx << "," << (int)dataset.training_labels[i] << std::endl;
    //}

    return 0;
}