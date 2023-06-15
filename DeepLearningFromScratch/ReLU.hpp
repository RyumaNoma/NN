#pragma once
#include "Layer.hpp"
#include "Matrix.hpp"

namespace layer {
	class ReLU
		: public Layer
	{
	public:
		ReLU(std::string name = "ReLU Layer") : Layer(name) {}
		Matrix Forward(const Matrix& in) override;
		Matrix Backward(const Matrix& dout) override;
		void Update() override {}
		std::string Serialize() const override;
	private:
		Matrix input;
	};
}