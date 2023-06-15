#include "NeuralNetwork.hpp"
#include <sstream>

NeuralNetwork::NeuralNetwork()
	: layers()
	, lossLayer(nullptr)
{
}

void NeuralNetwork::Add(std::shared_ptr<layer::Layer> layer)
{
	layers.push_back(layer);
}

void NeuralNetwork::SetLossLayer(std::shared_ptr<loss_layer::LossLayer> lossLayer)
{
	this->lossLayer = lossLayer;
}

double NeuralNetwork::Fit(const Matrix& in, const Matrix& out)
{
	double loss = Inference(in, out);
	Gradient();
	Update();
	return loss;
}

Matrix NeuralNetwork::Predict(const Matrix& in)
{
	Matrix x = in;
	for (std::shared_ptr<layer::Layer> layer : layers) {
		x = layer->Forward(x);
	}
	return x;
}

double NeuralNetwork::Inference(const Matrix& in, const Matrix& out)
{
	Matrix pred = Predict(in);
	double loss = lossLayer->Forward(pred, out);
	return loss;
}

void NeuralNetwork::Gradient()
{
	Matrix dout = lossLayer->Backward();
	for (auto itr = layers.rbegin(); itr != layers.rend(); ++itr) {
		dout = (*itr)->Backward(dout);
	}
}

void NeuralNetwork::Update()
{
	for (std::shared_ptr<layer::Layer> layer : layers) {
		layer->Update();
	}
}

std::string NeuralNetwork::Serialize() const
{
	std::ostringstream oss;
	for (std::shared_ptr<layer::Layer> layer : layers) {
		oss << layer->Serialize();
	}
	oss << lossLayer->Serialize();
	return oss.str();
}
