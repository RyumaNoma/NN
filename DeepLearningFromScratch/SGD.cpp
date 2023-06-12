#include "SGD.hpp"
#include "Matrix.hpp"

namespace optimizer {
	SGD::SGD(double lr)
		: lr(lr)
	{
	}

	void SGD::Update(Matrix& params, const Matrix& gradient)
	{
		params -= lr * gradient;
	}
	std::ostream& operator<<(std::ostream& os, const SGD& sgd)
	{
		os << "[SGD]";
		return os;
	}
}