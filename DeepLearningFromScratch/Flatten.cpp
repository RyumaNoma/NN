#include "Flatten.hpp"
#include "Matrix.hpp"
#include <sstream>

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
	std::string Flatten::Serialize() const
	{
		std::ostringstream oss;
		oss << "[name]" << '\n';
		oss << GetName() << '\n';
		return oss.str();
	}
}