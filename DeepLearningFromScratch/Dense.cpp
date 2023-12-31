#include "Dense.hpp"
#include "Matrix.hpp"
#include <sstream>

namespace layer {
	Dense::Dense(
		const int inputSize,
		const int outputSize,
		const Initializer::Type initializeType,
		std::mt19937& initializeRandomEngine,
		std::shared_ptr<optimizer::Optimizer> optWeight,
		std::shared_ptr<optimizer::Optimizer> optBias,
		std::string name)
		: weight(inputSize, outputSize)
		, bias(1, outputSize, 0.0)
		, input(1, inputSize)
		, dWeight(inputSize, outputSize)
		, dBias(1, outputSize)
		, optWeight(optWeight)
		, optBias(optBias)
		, initializeType(initializeType)
		, Layer(name)
	{
		Initializer::Initialize(weight, initializeRandomEngine, initializeType);
	}

	// TODO: BroadCastをしないように演算子を作る
	Matrix Dense::Forward(const Matrix& in)
	{
		this->input = in;
		Matrix out = Matrix::Dot(in, weight);
		for (int i = 0; i < out.Row(); ++i) {
			for (int j = 0; j < out.Col(); ++j) {
				out(i, j) += bias(0, j);
			}
		}
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
	std::string Dense::Serialize() const
	{
		std::ostringstream oss;
		oss << "[name]" << '\n';
		oss << GetName() << '\n';
		oss << "[size]" << '\n';
		oss << weight.Row() << " " << weight.Col() << '\n';
		oss << "[initialize]" << '\n';
		oss << initializeType << '\n';
		oss << "[optWeight]" << '\n';
		oss << optWeight->GetAlgorithmName() << '\n';
		oss << "[optBias]" << '\n';
		oss << optBias->GetAlgorithmName() << '\n';
		oss << "[input]" << '\n';
		oss << "[weight]" << '\n';
		oss << weight << '\n';
		oss << "[bias]" << '\n';
		oss << bias << '\n';
		return oss.str();
	}
}