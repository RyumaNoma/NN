#pragma once
#include <memory>
#include "Optimizer.hpp"

class Matrix;
class Random;
class NeuralNetwork;

class Common
{
public:
	// Šˆ«‰»ŠÖ”
	static double relu(double d) noexcept;
	static double sigmoid(double d) noexcept;
	static Matrix softmax(const Matrix& m);
	static double cross_entropy_error(const Matrix& pred, const Matrix& answer);
	static double mean_squared_error(const Matrix& pred, const Matrix& answer);
	static Matrix random_pick(int pickNum, const Matrix& data, Random& rand);
	static Matrix numerical_gradient(Matrix& params, NeuralNetwork& nn, const Matrix& in, const Matrix& answer);
};

