#include "SoftmaxCrossEntropyError.hpp"
#include "Common.hpp"

double loss_layer::SoftmaxCrossEntropyError::Forward(const Matrix& pred, const Matrix& answer)
{
    this->pred = Common::softmax(pred);
    this->answer = answer;
    return Common::cross_entropy_error(this->pred, answer);
}

Matrix loss_layer::SoftmaxCrossEntropyError::Backward()
{
    return (this->pred - this->answer) / this->answer.Row();
}
