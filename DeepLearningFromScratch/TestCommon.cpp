#include "TestCommon.hpp"
#include "Matrix.hpp"
#include "Common.hpp"
#include "Random.hpp"
#include <iostream>

void TestCommon::Softmax()
{
	double data[6] = { 1010, 1000, 990, 1000, 1010, 990 };
	Matrix in(2, 3, data);

	Matrix out = Common::softmax(in);
	std::cerr << out << std::endl;
	std::cerr << "finish softmax test" << std::endl;
}

void TestCommon::CrossEntropyError()
{
	double pred[10] = {
		0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0
	};
	double answer[10] = {
		0, 0, 1, 0, 0, 0, 0, 0, 0, 0
	};
	Matrix p(1, 10, pred);
	Matrix a(1, 10, answer);

	double cee = Common::cross_entropy_error(p, a);
	std::cerr << cee << std::endl;
	std::cerr << "finish cross etropy error test" << std::endl;
}

void TestCommon::MSE()
{
	double pred[10] = {
		0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0
	};
	double answer[10] = {
		0, 0, 1, 0, 0, 0, 0, 0, 0, 0
	};
	Matrix p(1, 10, pred);
	Matrix a(1, 10, answer);

	double mse = Common::mean_squared_error(p, a);
	std::cerr << mse << std::endl;
	std::cerr << "finish MSE test" << std::endl;
}

void TestCommon::RandomPick()
{
	double d[10] = { 1,2,3,4,5,6,7,8,9,0 };
	Matrix m(10, 1);
	std::mt19937 e;
	Matrix picked = Common::random_pick(5, m, e);
	std::cerr << picked << std::endl;
	std::cerr << "finish random pick test" << std::endl;
}

void TestCommon::All()
{
	Softmax();
	CrossEntropyError();
	MSE();
	RandomPick();
}
