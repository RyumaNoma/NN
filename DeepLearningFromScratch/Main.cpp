#include "Matrix.hpp"
#include "Initializer.hpp"
#include "Timer.hpp"
#include <random>

int main(void) {
	Matrix x(100, 100);
	Matrix w(100, 100);
	std::mt19937 e;
	Initializer::Initialize(x, e, Initializer::Type::Xavier);
	Initializer::Initialize(w, e, Initializer::Type::Xavier);

	std::vector<Matrix> v(100);
	Timer t;
	for (int i = 0; i < 100; ++i) {
		v[i] = Matrix::Dot(x, w);
		//std::cerr << "v[i]:" << v[i].data << std::endl;
	}
	std::cout << t.ElapsedMilliseconds() << "[ms]" << std::endl;
	std::cout << v[0](0) << std::endl;
}