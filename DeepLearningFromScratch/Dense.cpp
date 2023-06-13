#include "Dense.hpp"
#include "Matrix.hpp"

namespace layer {
	Dense::Dense(
		int outputSize,
		Initializer::Type initializeType,
		std::mt19937& initializeRandomEngine,
		std::shared_ptr<optimizer::Optimizer> optWeight,
		std::shared_ptr<optimizer::Optimizer> optBias,
		std::string name)
		: weight()
		, bias(1, outputSize, 0.0)
		, input()
		, dWeight()
		, dBias(1, outputSize)
		, optWeight(optWeight)
		, optBias(optBias)
		, initializeType(initializeType)
		, Layer(name)
	{
		Initializer::Initialize(weight, initializeRandomEngine, initializeType);
	}

	Matrix Dense::Forward(const Matrix& in)
	{
		Matrix broadcastBias(in.Row(), in.Col());
		for (int i = 0; i < broadcastBias.Row(); ++i) {
			for (int j = 0; j < broadcastBias.Col(); ++j) {
				broadcastBias(i, j) = bias(1, j);
			}
		}
		Matrix out;
		out = Matrix::Dot(in, weight) + broadcastBias;
		return out;
	}

	Matrix Dense::Backward(const Matrix& dout)
	{
		Matrix dInput = Matrix::Dot(dout, weight.T());
		dWeight = Matrix::Dot(input.T(), dout);
		dBias = dout.VerticalSum();
		return dInput;
	}
	void Dense::Update()
	{
		optWeight->Update(weight, dWeight);
		optBias->Update(bias, dBias);
	}
}