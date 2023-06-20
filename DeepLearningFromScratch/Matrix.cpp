#include "Matrix.hpp"
#include <stdexcept>
#include <algorithm>
#include <numeric>

Matrix::Matrix() noexcept
	: row(0)
	, col(0)
	, capacity(0)
	, data(nullptr)
{
}

Matrix::Matrix(int row, int col) noexcept
	: row(row)
	, col(col)
	, capacity(0)
	, data(nullptr)
{
	if (row * col > 0) {
		capacity = row * col;
		data = new double[capacity];
	}
}

Matrix::Matrix(const ShapeType& shape) noexcept
	: row(std::get<0>(shape))
	, col(std::get<1>(shape))
	, capacity(0)
	, data(nullptr)
{
	if (row * col > 0) {
		capacity = row * col;
		data = new double[capacity];
	}
}

Matrix::Matrix(const Matrix& m) noexcept
	: row(m.row)
	, col(m.col)
	, capacity(0)
	, data(nullptr)
{
	if (row * col > 0) {
		capacity = row * col;
		data = new double[capacity];
	}
	if (data) {
		std::copy(m.begin(), m.end(), this->begin());
	}
}

Matrix::Matrix(int row, int col, double const* rawData) noexcept
	: row(row)
	, col(col)
	, capacity(0)
	, data(nullptr)
{
	if (row * col > 0) {
		capacity = row * col;
		data = new double[capacity];
	}
	if (data) {
		std::copy(rawData, rawData + capacity, data);
	}
}

Matrix::Matrix(int row, int col, double fill) noexcept
	: row(row)
	, col(col)
	, capacity(0)
	, data(nullptr)
{
	if (row * col > 0) {
		capacity = row * col;
		data = new double[capacity];
	}
	if (data) {
		std::fill(begin(), end(), fill);
	}
}

Matrix::Matrix(int row, int col, const std::vector<int>& flattenData) noexcept
	: row(row)
	, col(col)
	, capacity(0)
	, data(nullptr)
{
	if (row * col > 0) {
		capacity = row * col;
		data = new double[capacity];
	}
	if (data) {
		std::copy(flattenData.begin(), flattenData.end(), this->begin());
	}
}

Matrix::~Matrix() {
	delete[] data;
}

void Matrix::Resize(int newRow, int newCol) {
	int newSize = newRow * newCol;
	if (newSize <= 0) {
		throw std::runtime_error("[Resize]not positive size");
	}
	if (newSize <= capacity) {
		row = newRow;
		col = newCol;
	}
	else {
		delete[] data;
		data = nullptr;
		row = newRow;
		col = newCol;
		capacity = newSize;
		data = new double[capacity];
	}
}

void Matrix::Reshape(int newRow, int newCol) {
	if (newRow * newCol <= capacity) {
		throw std::runtime_error("[Reshape]diffrent size");
	}
	row = newRow;
	col = newCol;
}

void Matrix::Reshape(const ShapeType& shape)
{
	Reshape(std::get<0>(shape), std::get<1>(shape));
}

Matrix Matrix::T() const
{
	Matrix out(col, row);
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			out.data[j * row + i] = data[i * col + j];
		}
	}
	return out;
}

Matrix Matrix::Flatten() const
{
	Matrix out(1, row * col);
	std::copy(begin(), end(), out.data);
	return out;
}

double Matrix::Sum() const
{
	return std::accumulate(begin(), end(), 0.0);
}

Matrix Matrix::VerticalSum() const
{
	Matrix sum(1, col, 0.0);
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			sum.data[j] += data[i * col + j];
		}
	}
	return sum;
}

Matrix Matrix::HorizontalSum() const
{
	Matrix sum(row, 1, 0.0);
	for (int i = 0; i < row; ++i) {
		sum.data[i] = std::accumulate(cbegin(i), cend(i), 0.0);
	}
	return sum;
}

double Matrix::Max() const
{
	return *std::max_element(begin(), end());
}

Matrix Matrix::VerticalMax() const
{
	Matrix max(1, col);
	for (int j = 0; j < col; ++j) {
		max.data[j] = data[j];
		for (int i = 1; i < row; ++i) {
			max.data[j] = std::max(max.data[j], data[i * col + j]);
		}
	}
	return max;
}

Matrix Matrix::HorizontalMax() const
{
	Matrix max(row, 1);
	for (int i = 0; i < row; ++i) {
		max.data[i] = *std::max_element(cbegin(i), cend(i));
	}
	return max;
}

double Matrix::Min() const
{
	return *std::min_element(begin(), end());
}

Matrix Matrix::VerticalMin() const
{
	Matrix min(1, col);
	for (int j = 0; j < col; ++j) {
		min.data[j] = data[j];
		for (int i = 1; i < row; ++i) {
			min.data[j] = std::min(min(j), data[i * col + j]);
		}
	}
	return min;
}

Matrix Matrix::HorizontalMin() const
{
	Matrix min(row, 1);
	for (int i = 0; i < row; ++i) {
		min.data[i] = data[i * col + 0];
		for (int j = 0; j < col; ++j) {
			min.data[i] = std::min(min.data[i], data[i * col + j]);
		}
	}
	return min;
}

double Matrix::Average() const
{
	return Sum() / Size();
}

Matrix Matrix::operator+(const Matrix& m) const {
	if (row != m.row || col != m.col) {
		throw std::runtime_error("[operator +]different shape");
	}
	Matrix result(row, col);
	for (int i = 0; i < row * col; ++i) {
		result.data[i] = this->data[i] + m.data[i];
	}
	return result;
}

Matrix Abs(const Matrix& m)
{
	Matrix out(m.Shape());
	for (int i = 0; i < m.Size(); ++i) {
		out.data[i] = std::abs(m(i));
	}
	return out;
}

Matrix operator+(double d, const Matrix& m)
{
	return m + d;
}

Matrix Matrix::operator+(double d) const
{
	Matrix result(row, col);
	for (int i = 0; i < row * col; ++i) {
		result.data[i] = this->data[i] + d;
	}
	return result;
}

Matrix Matrix::operator-(const Matrix& m) const {
	if (row != m.row || col != m.col) {
		throw std::runtime_error("[operator -]different shape");
	}
	Matrix result(row, col);
	for (int i = 0; i < row * col; ++i) {
		result.data[i] = this->data[i] - m.data[i];
	}
	return result;
}

Matrix operator-(double d, const Matrix& m)
{
	Matrix result(m.row, m.col);
	for (int i = 0; i < m.row * m.col; ++i) {
		result.data[i] = d - m.data[i];
	}
	return result;
}

Matrix Matrix::operator-(double d) const
{
	Matrix result(row, col);
	for (int i = 0; i < row * col; ++i) {
		result.data[i] = this->data[i] - d;
	}
	return result;
}

Matrix Matrix::operator*(const Matrix& m) const {
	if (col != m.col || row != m.row) {
		throw std::runtime_error("[operator *]different shape");
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

Matrix operator * (double coef, const Matrix& m) {
	return m * coef;
}

Matrix Matrix::Dot(const Matrix& left, const Matrix& right) {
	if (left.col != right.row) {
		throw std::runtime_error("[Dot]not match shape");
	}
	Matrix rightT = right.T();
	Matrix result(left.row, rightT.row);
	for (int i = 0; i < left.row; ++i) {
		for (int j = 0; j < rightT.row; ++j) {
			result.data[i * rightT.row + j] =
				std::inner_product(left.cbegin(i), left.cend(i), rightT.cbegin(j), 0.0);
		}
	}
	return result;
}

Matrix Matrix::operator/(const Matrix& m) const
{
	if (row != m.row || col != m.col) {
		throw std::runtime_error("[operator /]different shape");
	}
	Matrix result(row, col);
	for (int i = 0; i < row * col; ++i) {
		result.data[i] = this->data[i] / m.data[i];
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

Matrix operator/(double d, const Matrix& m)
{
	Matrix result(m.row, m.col);
	for (int i = 0; i < m.row * m.col; ++i) {
		result.data[i] = d / m.data[i];
	}
	return result;
}

Matrix& Matrix::operator=(const Matrix& m) {
	if (row != m.row || col != m.col) {
		Resize(m.row, m.col);
	}
	std::copy(m.begin(), m.end(), data);
	return *this;
}

Matrix& Matrix::operator+=(const Matrix& m) {
	if (row != m.row || col != m.col) {
		throw std::runtime_error("[operator +=]different shape");
	}
	for (int i = 0; i < row * col; ++i) {
		this->data[i] += m.data[i];
	}
	return *this;
}

Matrix& Matrix::operator-=(const Matrix& m) {
	if (row != m.row || col != m.col) {
		throw std::runtime_error("[operator -=]different shape");
	}
	for (int i = 0; i < row * col; ++i) {
		this->data[i] -= m.data[i];
	}
	return *this;
}

Matrix& Matrix::operator*=(const Matrix& m) {
	if (row != m.row || col != m.col) {
		throw std::runtime_error("[operator *=]different shape");
	}
	for (int i = 0; i < row * col; ++i) {
		this->data[i] *= m.data[i];
	}
	return *this;
}

Matrix& Matrix::operator*=(double coef)
{
	for (double i : *this) {
		i *= coef;
	}
	return *this;
}

Matrix& Matrix::operator/=(const Matrix& m)
{
	if (row != m.row || col != m.col) {
		throw std::runtime_error("[operator /=]different shape");
	}
	for (int i = 0; i < row * col; ++i) {
		this->data[i] /= m.data[i];
	}
	return *this;
}

Matrix& Matrix::operator/=(double coef)
{
	for (double& i : *this) {
		i /= coef;
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
		throw std::runtime_error("[operator (i)]out of data");
	}
	return data[i];
}

double& Matrix::operator()(int i, int j) const {
	if (i * col + j >= Size()) {
		throw std::runtime_error("[operator (i,j)]out of data");
	}
	return data[i * col + j];
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
	os << "{\n";
	for (int i = 0; i < m.row; ++i) {
		os << " {";
		for (int j = 0; j < m.col; ++j) {
			os << m.data[i * m.col + j] << ", ";
		}
		os << "}\n";
	}
	os << "}";
	return os;
}