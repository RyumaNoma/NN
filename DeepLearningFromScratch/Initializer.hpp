#pragma once
#include <random>
class Matrix;

// TODO: ‰Šú‰»•û–@‚Ì•×‹­&À‘•
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

