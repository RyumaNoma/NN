#pragma once
#include <memory>
#include <random>
#include "Optimizer.hpp"

class Matrix;
class NeuralNetwork;

class Common
{
public:
	// äàê´âªä÷êî
	static double relu(double d) noexcept;
	static double sigmoid(double d) noexcept;
	static Matrix softmax(const Matrix& m);
	// answerÇÕonehotï\åª
	static double cross_entropy_error(const Matrix& pred, const Matrix& answer);
	static double mean_squared_error(const Matrix& pred, const Matrix& answer);
	static std::vector<int> random_index(int pickNum, int dataSize, std::mt19937& rand);
	static Matrix pick(const Matrix& data, const std::vector<int>& index);
	static Matrix random_pick(int pickNum, const Matrix& data, std::mt19937& rand);
	static Matrix numerical_gradient(Matrix& params, NeuralNetwork& nn, const Matrix& in, const Matrix& answer);
	static Matrix onehot(const Matrix& verticalVector, int numClasses);
};

