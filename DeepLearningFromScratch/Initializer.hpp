#pragma once
#include <random>
class Matrix;

// TODO: 初期化方法の勉強&実装
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

