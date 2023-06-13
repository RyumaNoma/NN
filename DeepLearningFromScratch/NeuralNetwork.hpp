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
	// 最適化1ループ分
	// return: loss
	double Fit(const Matrix& in, const Matrix& out);
	// return: 推測値
	Matrix Predict(const Matrix& in);
	// return: loss
	double Inference(const Matrix& in, const Matrix& out);
	void Gradient();
	void Update();

	// TODO: NN : Serialize実装
	std::string Serialize() const;
private:	// lossを含まない層
	std::vector<std::shared_ptr<layer::Layer>> layers;
	std::shared_ptr<loss_layer::LossLayer> lossLayer;
};

