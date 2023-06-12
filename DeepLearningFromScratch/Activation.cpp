#include "Activation.hpp"
#include "Matrix.hpp"
#include "Common.hpp"

Activation::Activation() noexcept
	: type(Type::identity)
{
}

Activation::Activation(std::string activation) noexcept
	: type(String2Type(activation))
{
}

void Activation::Activate(Matrix& m)
{
	switch (type)
	{
	case Activation::Type::ReLU:
		ActivateReLU(m);
		break;
	case Activation::Type::sigmoid:
		ActivateSigmoid(m);
		break;
	case Activation::Type::identity:
		break;
	default:
		break;
	}
}

Activation::Type Activation::String2Type(std::string str) noexcept
{
	for (char& c : str) {
		c = std::tolower(c);
	}
	
	Type convertedType = Type::identity;
	if (str == "relu") {
		convertedType = Type::ReLU;
	}
	else if (str == "sigmoid") {
		convertedType = Type::sigmoid;
	}
	else if (str == "identity") {
		convertedType = Type::identity;
	}

	return convertedType;
}

void Activation::ActivateReLU(Matrix& m) {
	for (int i = 0; i < m.Size(); ++i) {
		m(i) = Common::relu(m(i));
	}
}

void Activation::ActivateSigmoid(Matrix& m) {
	for (int i = 0; i < m.Size(); ++i) {
		m(i) = Common::sigmoid(m(i));
	}
}
