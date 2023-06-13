#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork()
	: layers()
	, lossLayer(nullptr)
{
}

void NeuralNetwork::Add(std::shared_ptr<layer::Layer> layer)
{
	layers.push_back(layer);
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