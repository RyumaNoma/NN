#include "Initializer.hpp"
#include "Matrix.hpp"
#include "Random.hpp"
#include <random>

void Initializer::Initialize(Matrix& m, std::mt19937& rnd, Type type)
{
	switch (type)
	{
	case Initializer::Type::Xavier:
		InitializeXavier(m, rnd);
		break;
	case Initializer::Type::He:
		InitializeHe(m, rnd);
		break;
	default:
		break;
	}
}

void Initializer::InitializeXavier(Matrix& m, std::mt19937& rnd)
{
	double std = 1.0 / std::sqrt(m.Row());
	// ïΩãœ0 ïWèÄïŒç∑std
	std::normal_distribution<> dist(0.0, std);
	
	for (int i = 0; i < m.Size(); ++i) {
		m(i) = dist(rnd);
	}
}

void Initializer::InitializeHe(Matrix& m, std::mt19937& rnd)
{
	double std = std::sqrt(2.0 / m.Row());
	// ïΩãœ0 ïWèÄïŒç∑std
	std::normal_distribution<> dist(0.0, std);

	for (int i = 0; i < m.Size(); ++i) {
		m(i) = dist(rnd);
	}
}

std::ostream& operator<<(std::ostream& os, const Initializer::Type type)
{
	switch (type)
	{
	case Initializer::Type::Xavier:
		os << "Xavier";
		break;
	case Initializer::Type::He:
		os << "He";
		break;
	default:
		break;
	}
	return os;
}
