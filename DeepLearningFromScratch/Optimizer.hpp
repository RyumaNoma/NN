#pragma once
#include <string>

class Matrix;
namespace optimizer {
	class Optimizer {
	public:
		Optimizer() {}
		virtual ~Optimizer() {}
		virtual void Update(Matrix& params, const Matrix& gradient) = 0;
		virtual std::string GetAlgorithmName() const = 0;
	};
}