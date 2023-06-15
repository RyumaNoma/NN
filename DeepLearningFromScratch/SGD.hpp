#pragma once
#include "Optimizer.hpp"
#include <iostream>

namespace optimizer {
	class SGD
		: public Optimizer {
	public:
		SGD(double lr = 0.01);
		void Update(Matrix& params, const Matrix& gradient);
		std::string GetAlgorithmName() const override { return "SGD"; }
		friend std::ostream& operator << (std::ostream& os, const SGD& sgd);
	private:
		double lr;
	};
}