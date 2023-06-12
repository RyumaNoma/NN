#include "Flatten.hpp"
#include "Matrix.hpp"

namespace layer {
	Flatten::Flatten(std::string name)
		: Layer(name)
	{
	}

	Matrix Flatten::Forward(const Matrix& in)
	{
		inputShape = in.Shape();
		return in.Flatten();
	}

	Matrix Flatten::Backward(const Matrix& dout)
	{
		Matrix dInput = dout;
		dInput.Reshape(inputShape);
		return dInput;
	}
}