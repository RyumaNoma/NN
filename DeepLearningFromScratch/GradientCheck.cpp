// 参考: https://www.tensorflow.org/tutorials/keras/classification?hl=ja
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
#define SAMPLES 2

int main(int argc, char* argv[]) {
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);
    // model
    std::mt19937 engine;
    std::shared_ptr<optimizer::SGD> sgd(new optimizer::SGD());
    std::shared_ptr<layer::Dense> dense1(new layer::Dense(
        784,
        5,
        Initializer::Type::Xavier,
        engine,
        sgd,
        sgd,
        "Dense Layer 1"));
    std::shared_ptr<layer::ReLU> relu(new layer::ReLU());
    std::shared_ptr<layer::Dense> dense2(new layer::Dense(
        5,
        10,
        Initializer::Type::Xavier,
        engine,
        sgd,
        sgd,
        "Dense Layer 2"));
    std::shared_ptr<loss_layer::SoftmaxCrossEntropyError> scee(new loss_layer::SoftmaxCrossEntropyError());
    NeuralNetwork model;
    model.Add(dense1);
    model.Add(relu);
    model.Add(dense2);
    model.SetLossLayer(scee);
    // input
    Matrix input(SAMPLES, 784);
    for (int i = 0; i < SAMPLES; ++i) {
        for (int j = 0; j < 784; ++j) {
            input(i, j) = static_cast<double>(dataset.training_images[i][j]);
        }
    }
    input /= 255.0;
    // answer
    Matrix answer(SAMPLES, 1);
    for (int i = 0; i < SAMPLES; ++i) {
        answer(i, 0) = dataset.training_labels[i];
    }
    answer = Common::onehot(answer, 10);

    double loss = model.Inference(input, answer);
    model.Gradient();
    std::cout << "calculated gradient" << std::endl << std::endl;

    //// dense1
    Matrix numGradWeight1 = Common::numerical_gradient(dense1->Weight(), model, input, answer);
    Matrix numGradBias1 = Common::numerical_gradient(dense1->Bias(), model, input, answer);
    std::cout << " [Dense Layer 1]" << std::endl;
    std::cout << "差の最大値：" << Abs(numGradWeight1 - dense1->GradientWeight()).Max() << std::endl;
    std::cout << "差の最小値：" << Abs(numGradBias1 - dense1->GradientBias()).Min() << std::endl;
    std::cout << std::endl;

    //// dense2
    Matrix numGradWeight2 = Common::numerical_gradient(dense2->Weight(), model, input, answer);
    Matrix numGradBias2 = Common::numerical_gradient(dense2->Bias(), model, input, answer);
    std::cout << " [Dense Layer 2]" << std::endl;
    std::cout << "差の最大値：" << Abs(numGradWeight2 - dense2->GradientWeight()).Max() << std::endl;
    std::cout << "差の最小値：" << Abs(numGradBias2 - dense2->GradientBias()).Min() << std::endl;

    return 0;
}