#include "Tanh.hpp"
#include <cmath>
#include <sstream>
namespace layer {
	Matrix Tanh::Forward(const Matrix& in)
	{
		input = in;
		Matrix out(in.Shape());

		for (int i = 0; i < out.Size(); ++i) {
			out(i) = std::tanh(in(i));
		}
		return out;
	}

	Matrix Tanh::Backward(const Matrix& dout)
	{
		Matrix dInput(dout.Shape());

		for (int i = 0; i < dInput.Size(); ++i) {
			dInput(i) = 1.0 / std::cosh(input(i));
		}
		dInput *= dInput;
		return dInput * dout;
	}

	std::string Tanh::Serialize() const
	{
		std::ostringstream oss;
		oss << "[name]\n";
		oss << GetName() << '\n';
		return oss.str();
	}
}