#pragma once
#include <memory>
#include "Optimizer.hpp"

class Matrix;
class Random;
class NeuralNetwork;

class Common
{
public:
	// 活性化関数
	static double relu(double d) noexcept;
	static double sigmoid(double d) noexcept;
	static Matrix softmax(const Matrix& m);
	// answerはonehot表現
	static double cross_entropy_error(const Matrix& pred, const Matrix& answer);
	static double mean_squared_error(const Matrix& pred, const Matrix& answer);
	static Matrix random_pick(int pickNum, const Matrix& data, Random& rand);
	static Matrix numerical_gradient(Matrix& params, NeuralNetwork& nn, const Matrix& in, const Matrix& answer);
	static Matrix onehot(const Matrix& verticalVector, int numClasses);
};

