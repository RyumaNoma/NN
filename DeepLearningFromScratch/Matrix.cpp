#include "Matrix.hpp"
#include <stdexcept>

Matrix::Matrix() noexcept
	: row(0)
	, col(0)
	, data(nullptr)
{
}

Matrix::Matrix(int row, int col) noexcept
	: row(row)
	, col(col)
	, data(nullptr)
{
	if (row * col > 0) {
		data = new double[row * col];
	}
}

Matrix::Matrix(const std::vector<int>& shape) noexcept
	: row(shape.at(0))
	, col(shape.at(1))
	, data(nullptr)
{
	if (row * col > 0) {
		data = new double[row * col];
	}
}

Matrix::Matrix(const Matrix& m) noexcept
	: row(m.row)
	, col(m.col)
	, data(nullptr)
{
	if (row * col > 0) {
		data = new double[row * col];
	}
	if (data) {
		for (int i = 0; i < row * col; ++i) {
			data[i] = m.data[i];
		}
	}
}

Matrix::Matrix(int row, int col, double const* d) noexcept
	: row(row)
	, col(col)
	, data(nullptr)
{
	if (row * col > 0) {
		data = new double[row * col];
	}
	if (data) {
		for (int i = 0; i < row * col; ++i) {
			data[i] = d[i];
		}
	}
}

Matrix::Matrix(int row, int col, double fill) noexcept
	: row(row)
	, col(col)
	, data(nullptr)
{
	if (row * col > 0) {
		data = new double[row * col];
	}
	if (data) {
		for (int i = 0; i < row * col; i++) {
			data[i] = fill;
		}
	}
}

Matrix::~Matrix() {
	delete[] data;
}

void Matrix::Resize(int newRow, int newCol) {
	if (newRow * newCol <= 0) {
		throw std::runtime_error("negative size");
	}
	delete[] data;
	data = nullptr;
	row = newRow;
	col = newCol;
	data = new double[newRow * newCol];
}

void Matrix::Reshape(int newRow, int newCol) {
	if (newRow * newCol != row * col) {
		throw std::runtime_error("diffrent size");
	}
	row = newRow;
	col = newCol;
}

void Matrix::Reshape(const std::vector<int>& shape)
{
	if (shape[0] * shape[1] != row * col) {
		throw std::runtime_error("diffrent size");
	}
	row = shape[0];
	col = shape[1];
}

Matrix Matrix::T() const
{
	Matrix out(col, row);
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			out(j, i) = data[i * col + j];
		}
	}
	return out;
}

Matrix Matrix::Flatten() const
{
	Matrix out(1, row * col);
	for (int i = 0; i < row * col; ++i) {
		out(i) = data[i];
	}
	return out;
}

double Matrix::Sum() const
{
	double sum = 0;
	for (int i = 0; i < row * col; ++i) {
		sum += data[i];
	}
	return sum;
}

Matrix Matrix::VerticalSum() const
{
	Matrix sum(col, 1, 0.0);
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			sum(j) += data[i * col + j];
		}
	}
	return sum;
}

Matrix Matrix::HorizontalSum() const
{
	Matrix sum(row, 1, 0.0);
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			sum(i) += data[i * col + j];
		}
	}
	return sum;
}

double Matrix::Max() const
{
	double max = data[0];
	for (int i = 0; i < row * col; ++i) {
		max = std::max(max, data[i]);
	}
	return max;
}

Matrix Matrix::VerticalMax() const
{
	Matrix max(1, col);
	for (int j = 0; j < col; ++j) {
		max(j) = data[j];
		for (int i = 0; i < row; ++i) {
			max(j) = std::max(max(j), data[i * col + j]);
		}
	}
	return max;
}

Matrix Matrix::HorizontalMax() const
{
	Matrix max(1, row);
	for (int i = 0; i < row; ++i) {
		max(i) = data[i * col + 0];
		for (int j = 0; j < col; ++j) {
			max(i) = std::max(max(i), data[i * col + j]);
		}
	}
	return max;
}

double Matrix::Min() const
{
	double min = data[0];
	for (int i = 0; i < row * col; ++i) {
		min = std::min(min, data[i]);
	}
	return min;
}

Matrix Matrix::VerticalMin() const
{
	Matrix min(1, col);
	for (int j = 0; j < col; ++j) {
		min(j) = data[j];
		for (int i = 0; i < row; ++i) {
			min(j) = std::min(min(j), data[i * col + j]);
		}
	}
	return min;
}

Matrix Matrix::HorizontalMin() const
{
	Matrix min(1, row);
	for (int i = 0; i < row; ++i) {
		min(i) = data[i * col + 0];
		for (int j = 0; j < col; ++j) {
			min(i) = std::min(min(i), data[i * col + j]);
		}
	}
	return min;
}

Matrix Matrix::operator+(const Matrix& m) const {
	if (row != m.row || col != m.col) {
		throw std::runtime_error("different shape");
	}
	Matrix result(row, col);
	for (int i = 0; i < row * col; ++i) {
		result.data[i] = this->data[i] + m.data[i];
	}
	return result;
}

Matrix Matrix::operator-(const Matrix& m) const {
	if (row != m.row || col != m.col) {
		throw std::runtime_error("different shape");
	}
	Matrix result(row, col);
	for (int i = 0; i < row * col; ++i) {
		result.data[i] = this->data[i] - m.data[i];
	}
	return result;
}

Matrix Matrix::operator*(const Matrix& m) const {
	if (col != m.col || row != m.row) {
		throw std::runtime_error("different shape");
	}
	Matrix result(row, col);
	for (int i = 0; i < row * col; ++i) {
		result.data[i] = this->data[i] * m.data[i];
	}
	return result;
}

Matrix Matrix::operator*(double coef) const {
	Matrix result(row, col);
	for (int i = 0; i < row * col; ++i) {
		result.data[i] = this->data[i] * coef;
	}
	return result;
}

Matrix Matrix::Dot(const Matrix& left, const Matrix& right) {
	if (left.col != right.row) {
		throw std::runtime_error("not match shape");
	}
	Matrix result(left.row, right.col);
	for (int i = 0; i < left.row; ++i) {
		for (int j = 0; j < right.col; ++j) {
			result.data[i * right.col + j] = 0;
			for (int k = 0; k < left.col; ++k) {
				result.data[i * right.col + j] += left.data[i * left.col + k] * right.data[k * right.col + j];
			}
		}
	}
	return result;
}

Matrix Matrix::operator/(double coef) const {
	Matrix result(row, col);
	for (int i = 0; i < row * col; ++i) {
		result.data[i] = this->data[i] / coef;
	}
	return result;
}

Matrix& Matrix::operator=(const Matrix& m) {
	if (row != m.row || col != m.col) {
		Resize(m.row, m.col);
	}
	for (int i = 0; i < row * col; ++i) {
		data[i] = m.data[i];
	}
	return *this;
}

Matrix& Matrix::operator+=(const Matrix& m) {
	if (row != m.row || col != m.col) {
		throw std::runtime_error("different shape");
	}
	for (int i = 0; i < row * col; ++i) {
		this->data[i] += m.data[i];
	}
	return *this;
}

Matrix& Matrix::operator-=(const Matrix& m) {
	if (row != m.row || col != m.col) {
		throw std::runtime_error("different shape");
	}
	for (int i = 0; i < row * col; ++i) {
		this->data[i] -= m.data[i];
	}
	return *this;
}

Matrix& Matrix::operator*=(const Matrix& m) {
	if (row != m.row || col != m.col) {
		throw std::runtime_error("different shape");
	}
	for (int i = 0; i < row * col; ++i) {
		this->data[i] *= m.data[i];
	}
	return *this;
}

Matrix& Matrix::operator*=(double coef)
{
	for (int i = 0; i < row * col; ++i) {
		this->data[i] *= coef;
	}
	return *this;
}

Matrix& Matrix::operator/=(double coef)
{
	for (int i = 0; i < row * col; ++i) {
		this->data[i] /= coef;
	}
	return *this;
}

bool Matrix::operator==(const Matrix& m) const noexcept {
	if (row != m.row) return false;
	if (col != m.col) return false;
	for (int i = 0; i < row * col; ++i) {
		if (this->data[i] != m.data[i]) return false;
	}
	return true;
}

bool Matrix::operator!=(const Matrix& m) const noexcept {
	return !(*this == m);
}

double& Matrix::operator()(int i) const {
	if (i >= Size()) {
		throw std::runtime_error("out of data");
	}
	return data[i];
}

double& Matrix::operator()(int i, int j) const {
	if (i * col + j >= Size()) {
		throw std::runtime_error("out of data");
	}
	return data[i * col + j];
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
	for (int i = 0; i < m.row; ++i) {
		os << " {";
		for (int j = 0; j < m.col; ++j) {
			os << m.data[i * m.col + j] << ", ";
		}
		os << "}\n";
	}
	return os;
}

Matrix operator * (double coef, const Matrix& m) {
	return m * coef;
}