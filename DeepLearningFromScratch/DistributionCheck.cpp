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
#include <array>
#include <sstream>
#include "Initializer.hpp"
#define MNIST_DATA_LOCATION "./mnist/dataDigit/"
#define SAMPLES 60000
#define BATCH 128
#define MAX_ITERATIONS 1
//#define EPOCH 200

std::array<std::vector<double>, 4> result;

int main(int argc, char* argv[]) {
    Debug::Reset();

    std::mt19937 engine(1);
    std::normal_distribution<> dist(0, 1);
    Matrix x(100, 100);
    for (int i = 0; i < x.Size(); ++i) {
        x(i) = dist(engine);
        Debug::Print(x(i));
    }
    int nodeNum = 100;
    int hiddenLayerSize = 5;
    std::vector<Matrix> activations(hiddenLayerSize);
    //std::normal_distribution<> dist2(0, std::sqrt(2.0 / nodeNum));

    for (int i = 0; i < hiddenLayerSize; ++i) {
        if (i != 0) {
            x = activations[i - 1];
        }
        Matrix w(nodeNum, nodeNum);
        Initializer::Initialize(w, engine, Initializer::Type::He);
        //for (int i = 0; i < w.Size(); ++i) {
        //    w(i) = dist2(engine);
        //}
        Matrix z = Matrix::Dot(x, w);
        Matrix a(z.Shape());
        for (int i = 0; i < z.Size(); ++i) {
            a(i) = Common::relu(z(i));
        }
        activations[i] = a;

        std::ostringstream filename;
        filename << "Debug_relu_w_" << i + 1 << ".txt";
        Debug::FILENAME = filename.str();
        for (int i = 0; i < w.Size(); ++i) {
            Debug::Print(w(i));
        }
    }

    //
    //// model
    //std::mt19937 engine(1);
    //NeuralNetwork model;
    //std::shared_ptr<optimizer::SGD> sgd(new optimizer::SGD());
    //std::shared_ptr<layer::Dense> dense1(new layer::Dense(
    //    784,
    //    100,
    //    Initializer::Type::He,
    //    engine,
    //    sgd,
    //    sgd,
    //    "Dense Layer 1"));
    //std::shared_ptr<layer::ReLU> relu1(new layer::ReLU("ReLU 1"));
    //std::shared_ptr<layer::Dense> dense2(new layer::Dense(
    //    100,
    //    100,
    //    Initializer::Type::He,
    //    engine,
    //    sgd,
    //    sgd,
    //    "Dense Layer 2"));
    //std::shared_ptr<layer::ReLU> relu2(new layer::ReLU("ReLU 2"));
    //std::shared_ptr<layer::Dense> dense3(new layer::Dense(
    //    100,
    //    100,
    //    Initializer::Type::He,
    //    engine,
    //    sgd,
    //    sgd,
    //    "Dense Layer 3"));
    //std::shared_ptr<layer::ReLU> relu3(new layer::ReLU("ReLU 3"));
    //std::shared_ptr<layer::Dense> dense4(new layer::Dense(
    //    100,
    //    100,
    //    Initializer::Type::He,
    //    engine,
    //    sgd,
    //    sgd,
    //    "Dense Layer 4"));
    //std::shared_ptr<layer::ReLU> relu4(new layer::ReLU("ReLU 4"));
    //std::shared_ptr<layer::Dense> dense5(new layer::Dense(
    //    100,
    //    10,
    //    Initializer::Type::He,
    //    engine,
    //    sgd,
    //    sgd,
    //    "Dense Layer 5"));
    //std::shared_ptr<layer::ReLU> relu5(new layer::ReLU("ReLU 5"));
    //std::shared_ptr<loss_layer::SoftmaxCrossEntropyError> scee(new loss_layer::SoftmaxCrossEntropyError());
    //model.Add(dense1);
    //model.Add(relu1);
    //model.Add(dense2);
    //model.Add(relu2);
    //model.Add(dense3);
    //model.Add(relu3);
    //model.Add(dense4);
    //model.Add(relu4);
    //model.Add(dense5);
    //model.SetLossLayer(scee);
    //std::cerr << "built model" << std::endl;

    //std::normal_distribution<> dist(0, 1);
    //Matrix m(1000, 784);
    //for (int i = 0; i < m.Size(); ++i) {
    //    m(i) = dist(engine);
    //}

    //// 1
    //m = dense1->Forward(m);
    //m = relu1->Forward(m);
    //Debug::FILENAME = "./Debug1.txt";
    //for (int i = 0; i < 1000; ++i) {
    //    for (int j = 0; j < 100; ++j) {
    //        result[0].push_back(m(i, j));
    //        Debug::Print(m(i, j));
    //    }
    //}
    //// 2
    //m = dense2->Forward(m);
    //m = relu2->Forward(m);
    //Debug::FILENAME = "./Debug2.txt";
    //for (int i = 0; i < 1000; ++i) {
    //    for (int j = 0; j < 100; ++j) {
    //        result[1].push_back(m(i, j));
    //        Debug::Print(m(i, j));
    //    }
    //}
    //// 3
    //m = dense3->Forward(m);
    //m = relu3->Forward(m);
    //Debug::FILENAME = "./Debug3.txt";
    //for (int i = 0; i < 1000; ++i) {
    //    for (int j = 0; j < 100; ++j) {
    //        result[2].push_back(m(i, j));
    //        Debug::Print(m(i, j));
    //    }
    //}
    //// 4
    //m = dense4->Forward(m);
    //m = relu4->Forward(m);
    //Debug::FILENAME = "./Debug4.txt";
    //for (int i = 0; i < 1000; ++i) {
    //    for (int j = 0; j < 100; ++j) {
    //        result[3].push_back(m(i, j));
    //        Debug::Print(m(i, j));
    //    }
    //}

    //return 0;
}