#pragma once
#include "Layer.hpp"
#include "Activation.hpp"
#include "Initializer.hpp"
#include "Matrix.hpp"
#include "Optimizer.hpp"

namespace layer {
	class Dense
		: public Layer
	{
	public:
		Dense(
			int inputSize,
			int outputSize,
			Initializer::Type initializeType,
			std::mt19937& initializeRandomEngine,
			std::shared_ptr<optimizer::Optimizer> optWeight,
			std::shared_ptr<optimizer::Optimizer> optBias,
			std::string name = "Dense Layer");
		Matrix Forward(const Matrix& in) override;
		Matrix Backward(const Matrix& dout) override;
		void Update() override;

		Matrix& Weight() { return weight; }
		Matrix& Bias() { return bias; }
		Matrix& GradientWeight() { return dWeight; }
		Matrix& GradientBias() { return dBias; }
	private:
		Matrix weight, bias;
		Initializer initializer;
		Matrix input;
		Matrix dWeight, dBias;
		std::shared_ptr<optimizer::Optimizer> optWeight, optBias;
		Initializer::Type initializeType;
	};
}