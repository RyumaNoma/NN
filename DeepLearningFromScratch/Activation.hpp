#pragma once
#include <string>
class Matrix;

class Activation {
public:
	enum class Type {
		ReLU,
		sigmoid,
		identity
	};
	Activation() noexcept;
	Activation(std::string activation) noexcept;

	void Activate(Matrix& m);
private:
	static Type String2Type(std::string str) noexcept;
	static void ActivateReLU(Matrix& m);
	static void ActivateSigmoid(Matrix& m);

	Type type;
};

