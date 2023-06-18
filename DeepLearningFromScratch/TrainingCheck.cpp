// éQçl: https://www.tensorflow.org/tutorials/keras/classification?hl=ja
#include "NeuralNetwork.hpp"
#include "Common.hpp"
#include "Debug.hpp"
#include "Evaluation.hpp"
// layer
#include "Dense.hpp"
#include "ReLU.hpp"
#include "SoftmaxCrossEntropyError.hpp"
// optimizer
#include "SGD.hpp"
#include "mnist/mnist_reader.hpp"
#define MNIST_DATA_LOCATION "./mnist/dataDigit/"
#define SAMPLES 60000
#define BATCH 128
#define MAX_ITERATIONS 2000
//#define EPOCH 200

int main(int argc, char* argv[]) {
    Debug::Reset();
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    // Training input
    Matrix input(SAMPLES, 784);
    for (int i = 0; i < SAMPLES; ++i) {
        for (int j = 0; j < 784; ++j) {
            input(i, j) = static_cast<double>(dataset.training_images[i][j]);
        }
    }
    input /= 255.0;
    // Training answer
    Matrix answer(SAMPLES, 1);
    for (int i = 0; i < SAMPLES; ++i) {
        answer(i, 0) = dataset.training_labels[i];
    }
    answer = Common::onehot(answer, 10);

    //// Test input
    //Matrix testInput(BATCH, 784);
    //for (int i = 0; i < BATCH; ++i) {
    //    for (int j = 0; j < 784; ++j) {
    //        testInput(i, j) = static_cast<double>(dataset.test_images[i][j]);
    //    }
    //}
    //input /= 255.0;
    //// Test answer
    //Matrix testAnswer(BATCH, 1);
    //for (int i = 0; i < BATCH; ++i) {
    //    testAnswer(i, 0) = dataset.test_labels[i];
    //}
    //std::cerr << "loaded data" << std::endl;

    // model
    std::mt19937 engine(1);
    std::shared_ptr<optimizer::SGD> sgd(new optimizer::SGD());
    std::shared_ptr<layer::Dense> dense1(new layer::Dense(
        784,
        100,
        Initializer::Type::He,
        engine,
        sgd,
        sgd,
        "Dense Layer 1"));
    std::shared_ptr<layer::ReLU> relu1(new layer::ReLU("ReLU 1"));
    std::shared_ptr<layer::Dense> dense2(new layer::Dense(
        100,
        100,
        Initializer::Type::He,
        engine,
        sgd,
        sgd,
        "Dense Layer 2"));
    std::shared_ptr<layer::ReLU> relu2(new layer::ReLU("ReLU 2"));
    std::shared_ptr<layer::Dense> dense3(new layer::Dense(
        100,
        100,
        Initializer::Type::He,
        engine,
        sgd,
        sgd,
        "Dense Layer 3"));
    std::shared_ptr<layer::ReLU> relu3(new layer::ReLU("ReLU 3"));
    std::shared_ptr<layer::Dense> dense4(new layer::Dense(
        100,
        100,
        Initializer::Type::He,
        engine,
        sgd,
        sgd,
        "Dense Layer 4"));
    std::shared_ptr<layer::ReLU> relu4(new layer::ReLU("ReLU 4"));
    std::shared_ptr<layer::Dense> dense5(new layer::Dense(
        100,
        10,
        Initializer::Type::He,
        engine,
        sgd,
        sgd,
        "Dense Layer 5"));
    std::shared_ptr<loss_layer::SoftmaxCrossEntropyError> scee(new loss_layer::SoftmaxCrossEntropyError());
    NeuralNetwork model;
    model.Add(dense1);
    model.Add(relu1);
    model.Add(dense2);
    model.Add(relu2);
    model.Add(dense3);
    model.Add(relu3);
    model.Add(dense4);
    model.Add(relu4);
    model.Add(dense5);
    model.SetLossLayer(scee);
    std::cerr << "built model" << std::endl;

    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
        std::vector<int> index = Common::random_index(BATCH, SAMPLES, engine);
        auto in = Common::pick(input, index);
        auto ans = Common::pick(answer, index);

        double loss = model.Inference(in, ans);
        model.Gradient();
        model.Update();

        if (iter % 100 == 0) {
            printf("[%4d]: %lf\n", iter, loss);
        }
    }
    //Debug::Reset();
    //Debug::Print(model.Serialize());

    return 0;
}