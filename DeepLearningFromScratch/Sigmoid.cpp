#include "Sigmoid.hpp"
#include "Common.hpp"
Matrix layer::Sigmoid::Forward(const Matrix& in)
{
	Matrix out = in;
	for (int i = 0; i < out.Size(); ++i) {
		out(i) = Common::sigmoid(in(i));
	}
	return out;
}

Matrix layer::Sigmoid::Backward(const Matrix& dout)
{
	return dout * (1.0 - out) * out;
}
