#include "SoftmaxCrossEntropyError.hpp"
#include "Common.hpp"
#include "Debug.hpp"
#include <sstream>

double loss_layer::SoftmaxCrossEntropyError::Forward(const Matrix& pred, const Matrix& answer)
{
    this->pred = Common::softmax(pred);
    this->answer = answer;
    return Common::cross_entropy_error(this->pred, this->answer);
}

Matrix loss_layer::SoftmaxCrossEntropyError::Backward()
{
    return (this->pred - this->answer) / this->answer.Row();
}

std::string loss_layer::SoftmaxCrossEntropyError::Serialize() const
{
	std::ostringstream oss;
	oss << "[name]" << '\n';
	oss << GetName() << '\n';
	return oss.str();
}
