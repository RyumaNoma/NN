#include "Common.hpp"
#include "Matrix.hpp"
#include "NeuralNetwork.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

double Common::relu(const double d) noexcept {
	return std::max(0.0, d);
}

double Common::sigmoid(const double d) noexcept {
	return 1 / (1 + std::exp(-d));
}

void Common::softmax(Matrix& m)
{
	Matrix max = m.HorizontalMax();

	for (int i = 0; i < m.Row(); ++i) {
		for (int j = 0; j < m.Col(); ++j) {
			m(i, j) = std::exp(m(i, j) - max(i));
		}
	}
	Matrix sum = m.HorizontalSum();
	for (int i = 0; i < m.Row(); ++i) {
		for (int j = 0; j < m.Col(); ++j) {
			m(i, j) /= sum(i, 0);
		}
	}	
}

double Common::cross_entropy_error(const Matrix& pred, const Matrix& answer)
{
	if (pred.Col() != answer.Col()) {
		throw std::runtime_error("different classes");
	}
	if (pred.Row() != answer.Row()) {
		throw std::runtime_error("diffrent batch size");
	}
	double error = 0;
	for (int i = 0; i < pred.Row(); ++i) {
		for (int j = 0; j < pred.Col(); ++j) {
			error += answer(i, j) * std::log(pred(i, j) + 1e-7);
		}
	}
	return -error / pred.Row();
}

double Common::mean_squared_error(const Matrix& pred, const Matrix& answer)
{
	if (pred.Col() != answer.Col()) {
		throw std::runtime_error("different classes");
	}
	if (pred.Row() != answer.Row()) {
		throw std::runtime_error("diffrent batch size");
	}
	double mse = 0;
	for (int i = 0; i < pred.Row(); ++i) {
		for (int j = 0; j < pred.Col(); ++j) {
			mse += (answer(i, j) - pred(i, j)) * (answer(i, j) - pred(i, j));
		}
	}
	return 0.5 * mse / pred.Row();
}

std::vector<int> Common::random_index(int pickNum, int dataSize, std::mt19937& rand)
{
	std::vector<int> idx(dataSize);
	std::iota(idx.begin(), idx.end(), 0);
	if (dataSize <= pickNum) return idx;

	std::shuffle(idx.begin(), idx.end(), rand);
	idx.resize(pickNum);
	return idx;
}

Matrix Common::pick(const Matrix& data, const std::vector<int>& index)
{
	Matrix out(index.size(), data.Col());
	for (int i = 0; i < index.size(); ++i) {
		std::copy(data.cbegin(i), data.cend(i), out.cbegin(i));
	}
	return out;
}

Matrix Common::random_pick(int pickNum, const Matrix& data, std::mt19937& rand)
{
	return pick(data, random_index(pickNum, data.Row(), rand));
}

Matrix Common::numerical_gradient(Matrix& params, NeuralNetwork& nn, const Matrix& in, const Matrix& answer)
{
	double h = 1e-4;
	Matrix grad(params.Shape());

	for (int i = 0; i < params.Size(); ++i) {
		double tmpVal = params(i);
		
		params(i) = tmpVal + h;
		double lossHigh = nn.Inference(in, answer);

		params(i) = tmpVal - h;
		double lossLow = nn.Inference(in, answer);

		grad(i) = (lossHigh - lossLow) / (2 * h);
		params(i) = tmpVal;
	}
	return grad;
}

Matrix Common::onehot(const Matrix& verticalVector, int numClasses)
{
	Matrix converted(verticalVector.Row(), numClasses, 0.0);

	for (int i = 0; i < verticalVector.Row(); ++i) {
		converted(i, static_cast<int>(verticalVector(i, 0))) = 1.0;
	}
	return converted;
}

std::ostream& operator<<(std::ostream& os, const std::tuple<int, int>& t2)
{
	os << "(" << std::get<0>(t2) << "," << std::get<1>(t2) << ")";
	return os;
}
