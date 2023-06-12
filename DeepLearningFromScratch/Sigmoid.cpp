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
	Matrix dInput(dout.Shape());
	for (int i = 0; i < dout.Size(); ++i) {
		dInput(i) = out(i) * (1 - out(i));
	}
	return dout * dInput;
}
