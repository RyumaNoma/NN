#pragma once
#include <string>
#include "Matrix.hpp"

// TODO: LossLayer : MSE�̎���
namespace loss_layer {
	class LossLayer {
	public:
		LossLayer(std::string name) : name(name) {}
		virtual ~LossLayer() {}
		virtual double Forward(const Matrix& pred, const Matrix& answer) = 0;
		virtual Matrix Backward() = 0;
		virtual std::string Serialize() const = 0;
		std::string GetName() const { return name; }
	protected:
		std::string name;
		Matrix pred, answer;
	};
}