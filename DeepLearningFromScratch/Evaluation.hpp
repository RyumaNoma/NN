#pragma once
class Matrix;

class Evaluation
{
public:
	static Matrix ArgMax(const Matrix& prob);
	static double Accuracy(const Matrix& verticalVector1, const Matrix& verticalVector2);
};

