#pragma once
#include "Layer.hpp"
#include "Matrix.hpp"

namespace layer {
	class Flatten
		: public Layer
	{
	public:
		Flatten(std::string name = "Flatten Layer");
		Matrix Forward(const Matrix& in) override;
		Matrix Backward(const Matrix& dout) override;
		void Update() override {}
		std::string Serialize() const override;
	private:
		Matrix::ShapeType inputShape;
	};
}

