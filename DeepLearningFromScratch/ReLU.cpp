#include "ReLU.hpp"

namespace layer
{
	Matrix ReLU::Forward(const Matrix& in)
	{
		input = in;
		Matrix out(in.Shape());

		for (int i = 0; i < out.Size(); ++i) {
			out(i) = std::max(0.0, in(i));
		}
	}
	Matrix ReLU::Backward(const Matrix& dout)
	{
		Matrix dInput(dout.Shape());

		for (int i = 0; i < dInput.Size(); ++i) {
			dInput(i) = (input(i) > 0);
		}
		return dInput * dout;
	}
}