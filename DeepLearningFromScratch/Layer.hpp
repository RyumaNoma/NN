#pragma once
#include <vector>
#include <string>
#include <memory>
class Matrix;

// TODO: virtual void Initialize() = 0;
namespace layer {
	class Layer {
	public:
		Layer(std::string name = "Layer") : name(name) {}
		virtual ~Layer() {}
		virtual Matrix Forward(const Matrix& in) = 0;
		virtual Matrix Backward(const Matrix& dout) = 0;
		virtual void Update() = 0;
		virtual std::string Serialize() const = 0;
		std::string GetName() const { return name; }
	protected:
		std::string name;
	};
}