#include "Evaluation.hpp"
#include "Matrix.hpp"

Matrix Evaluation::ArgMax(const Matrix& prob)
{
	Matrix out(prob.Row(), 1);
	for (int i = 0; i < prob.Row(); ++i) {
		double maxProb = -1;
		int maxIdx = -1;
		for (int j = 0; j < prob.Col(); ++j) {
			if (prob(i, j) > maxProb) {
				maxProb = prob(i, j);
				maxIdx = j;
			}
		}
		out(i, 0) = maxIdx;
	}
	return out;
}

double Evaluation::Accuracy(const Matrix& verticalVector1, const Matrix& verticalVector2)
{
	if (verticalVector1.Row() != verticalVector2.Row()) {
		throw std::runtime_error("Accuracy: different row");
	}
	if (verticalVector1.Col() != 1) {
		throw std::runtime_error("Accuracy[1]: Column size must be 1");
	}
	if (verticalVector2.Col() != 1) {
		throw std::runtime_error("Accuracy[2]: Column size must be 1");
	}
	int N = verticalVector1.Row();
	double acc = 0;
	for (int i = 0; i < N; ++i) {
		acc += (verticalVector1(i, 0) == verticalVector2(i, 0));
	}
	return acc / N;
}
