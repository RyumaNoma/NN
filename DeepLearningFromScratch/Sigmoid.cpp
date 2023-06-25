#include "Sigmoid.hpp"
#include "Common.hpp"
#include <sstream>

Matrix layer::Sigmoid::Forward(const Matrix& in)
{
	this->out.Reshape(in.Shape());//bug
	for (int i = 0; i < out.Size(); ++i) {
		this->out(i) = Common::sigmoid(in(i));
	}
	return this->out;
}

Matrix layer::Sigmoid::Backward(const Matrix& dout)
{
	return dout * (1.0 - out) * out;
}

std::string layer::Sigmoid::Serialize() const
{
	std::ostringstream oss;
	oss << "[name]" << '\n';
	oss << GetName() << '\n';
	return oss.str();
}
