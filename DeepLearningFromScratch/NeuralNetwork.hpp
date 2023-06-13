#pragma once
#include <vector>
#include <memory>
#include <map>
#include "Layer.hpp"
#include "LossLayer.hpp"

class NeuralNetwork
{
public:
	NeuralNetwork();

	void Add(std::shared_ptr<layer::Layer> layer);
	void SetLossLayer(std::shared_ptr<loss_layer::LossLayer> lossLayer);
	// �œK��1���[�v��
	// return: loss
	double Fit(const Matrix& in, const Matrix& out);
	// return: �����l
	Matrix Predict(const Matrix& in);
	// return: loss
	double Inference(const Matrix& in, const Matrix& out);
	void Gradient();
	void Update();

	// TODO: NN : Serialize����
	std::string Serialize() const;
private:	// loss���܂܂Ȃ��w
	std::vector<std::shared_ptr<layer::Layer>> layers;
	std::shared_ptr<loss_layer::LossLayer> lossLayer;
};

