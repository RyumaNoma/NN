#pragma once
#include "LossLayer.hpp"
namespace loss_layer {
	class SoftmaxCrossEntropyError
		: public LossLayer
	{
	public:
		SoftmaxCrossEntropyError(std::string name = "Softmax Cross Entropy Error Layer") : LossLayer(name) {}
		virtual double Forward(const Matrix& pred, const Matrix& answer) override;
		virtual Matrix Backward() override;
	};
}