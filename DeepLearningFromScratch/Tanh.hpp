#pragma once
#include "Layer.hpp"
#include "Matrix.hpp"

namespace layer {
	class Tanh
		: public Layer {
	public:
		Tanh(std::string name = "Tanh Layer") : Layer(name) {}
		Matrix Forward(const Matrix& in) override;
		Matrix Backward(const Matrix& dout) override;
		void Update() override {}
		std::string Serialize() const override;
	private:
		Matrix input;
	};
}