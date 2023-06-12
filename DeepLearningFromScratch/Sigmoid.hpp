#pragma once
#include "Layer.hpp"
#include "Matrix.hpp"

namespace layer {
	class Sigmoid
		: public Layer
	{
	public:
		Sigmoid(std::string name = "Sigmoid Layer") : Layer(name) {}
		Matrix Forward(const Matrix& in) override;
		Matrix Backward(const Matrix& dout) override;
		void Update() override {}
	private:
		Matrix out;
	};
}