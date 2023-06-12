#pragma once

class Matrix;
namespace optimizer {
	class Optimizer {
	public:
		Optimizer() {}
		virtual ~Optimizer() {}
		virtual void Update(Matrix& params, const Matrix& gradient) = 0;
	};
}