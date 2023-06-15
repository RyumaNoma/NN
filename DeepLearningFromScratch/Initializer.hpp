#pragma once
#include <random>
#include <iostream>
class Matrix;
/*
Xavier: Sigmoid, Tanh
He    : ReLU
*/
class Initializer
{
public:
	enum class Type {
		Xavier,
		He
	};
	static void Initialize(Matrix& m, std::mt19937& rnd, Type type);
private:
	static void InitializeXavier(Matrix& m, std::mt19937& rnd);
	static void InitializeHe(Matrix& m, std::mt19937& rnd);
};
std::ostream& operator << (std::ostream& os, const Initializer::Type type);
