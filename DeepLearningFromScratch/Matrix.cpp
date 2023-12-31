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

Matrix::Matrix(const int row, const int col) noexcept
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

Matrix::Matrix(Matrix&& m) noexcept
	: row(m.row)
	, col(m.col)
	, capacity(m.capacity)
	, data(m.data)
{
	m.data = nullptr;
}

Matrix::Matrix(const int row, const int col, double const* rawData) noexcept
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

Matrix::Matrix(const int row, const int col, double fill) noexcept
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

Matrix::Matrix(const int row, const int col, const std::vector<int>& flattenData) noexcept
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

void Matrix::Resize(const int newRow, const int newCol) {
	const int newSize = newRow * newCol;
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

void Matrix::Reshape(const int newRow, const int newCol) {
	if (newRow * newCol != capacity) {
		throw std::runtime_error("[Reshape]diffrent size");
	}
	row = newRow;
	col = newCol;
}

void Matrix::Reshape(const ShapeType& shape)
{
	Reshape(std::get<0>(shape), std::get<1>(shape));
}

Matrix Matrix::T() const noexcept
{
	Matrix out(col, row);
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			out.data[j * row + i] = data[i * col + j];
		}
	}
	return out;
}

Matrix Matrix::Flatten() const noexcept
{
	Matrix out(1, row * col);
	std::copy(begin(), end(), out.data);
	return out;
}

double Matrix::Sum() const noexcept
{
	return std::accumulate(begin(), end(), 0.0);
}

Matrix Matrix::VerticalSum() const noexcept
{
	Matrix sum(1, col, 0.0);
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			sum.data[j] += data[i * col + j];
		}
	}
	return sum;
}

Matrix Matrix::HorizontalSum() const noexcept
{
	Matrix sum(row, 1);
	for (int i = 0; i < row; ++i) {
		sum.data[i] = std::accumulate(cbegin(i), cend(i), 0.0);
	}
	return sum;
}

double Matrix::Max() const noexcept
{
	return *std::max_element(begin(), end());
}

Matrix Matrix::VerticalMax() const noexcept
{
	Matrix max(1, col);
	for (int j = 0; j < col; ++j) {
		double* maxIter = data + j;
		for (int i = 1; i < row; ++i) {
			if (data[i * col + j] > *maxIter) {
				maxIter = data + (i * col + j);
			}
		}
		max.data[j] = *maxIter;
	}
	return max;
}

Matrix Matrix::HorizontalMax() const noexcept
{
	Matrix max(row, 1);
	for (int i = 0; i < row; ++i) {
		max.data[i] = *std::max_element(cbegin(i), cend(i));
	}
	return max;
}

double Matrix::Min() const noexcept
{
	return *std::min_element(begin(), end());
}

Matrix Matrix::VerticalMin() const noexcept
{
	Matrix min(1, col);
	for (int j = 0; j < col; ++j) {
		double* minIter = data + j;
		for (int i = 1; i < row; ++i) {
			if (data[i * col + j] > *minIter) {
				minIter = data + (i * col + j);
			}
		}
		min.data[j] = *minIter;
	}
	return min;
}

Matrix Matrix::HorizontalMin() const noexcept
{
	Matrix min(row, 1);
	for (int i = 0; i < row; ++i) {
		min.data[i] = *std::min_element(cbegin(i), cend(i));
	}
	return min;
}

double Matrix::Average() const noexcept
{
	return Sum() / Size();
}

Matrix Abs(const Matrix& m) noexcept
{
	Matrix out(m.Shape());
	for (int i = 0; i < m.Size(); ++i) {
		out.data[i] = std::abs(m(i));
	}
	return out;
}

Matrix operator + (const Matrix& lhs, const Matrix& rhs) {
	if (lhs.row != rhs.row || lhs.col != rhs.col) {
		throw std::runtime_error("[operator +]different shape");
	}
	Matrix result(lhs.row, lhs.col);
	const int size = lhs.row * lhs.col;
	for (int i = 0; i < size; ++i) {
		result.data[i] = lhs.data[i] + rhs.data[i];
	}
	return result;
}

Matrix operator+(const Matrix&  m, const double d)
{
	Matrix result(m.row, m.col);
	const int size = m.row * m.col;
	for (int i = 0; i < size; ++i) {
		result.data[i] = m.data[i] + d;
	}
	return result;
}

Matrix operator+(const double d, const Matrix& m)
{
	return m + d;
}

Matrix operator - (const Matrix& lhs, const Matrix& rhs) {
	if (lhs.row != rhs.row || lhs.col != rhs.col) {
		throw std::runtime_error("[operator -]different shape");
	}
	Matrix result(lhs.row, lhs.col);
	const int size = lhs.row * lhs.col;
	for (int i = 0; i < size; ++i) {
		result.data[i] = lhs.data[i] - rhs.data[i];
	}
	return result;
}

Matrix operator-(const Matrix& m, const double d)
{
	Matrix result(m.row, m.col);
	const int size = m.row * m.col;
	for (int i = 0; i < size; ++i) {
		result.data[i] = m.data[i] - d;
	}
	return result;
}

Matrix operator-(const double d, const Matrix& m)
{
	Matrix result(m.row, m.col);
	const int size = m.row * m.col;
	for (int i = 0; i < size; ++i) {
		result.data[i] = d - m.data[i];
	}
	return result;
}

Matrix operator * (const Matrix& lhs, const Matrix& rhs) {
	if (lhs.col != rhs.col || lhs.row != rhs.row) {
		throw std::runtime_error("[operator *]different shape");
	}
	Matrix result(lhs.row, lhs.col);
	const int size = lhs.row * lhs.col;
	for (int i = 0; i < size; ++i) {
		result.data[i] = lhs.data[i] * rhs.data[i];
	}
	return result;
}

Matrix operator * (const Matrix& m, const double d) {
	Matrix result(m.row, m.col);
	const int size = m.row * m.col;
	for (int i = 0; i < size; ++i) {
		result.data[i] = m.data[i] * d;
	}
	return result;
}

Matrix operator * (double d, const Matrix& m) {
	return m * d;
}

Matrix operator / (const Matrix& lhs, const Matrix& rhs) {
	if (lhs.row != rhs.row || lhs.col != rhs.col) {
		throw std::runtime_error("[operator /]different shape");
	}
	Matrix result(lhs.row, lhs.col);
	const int size = lhs.row * lhs.col;
	for (int i = 0; i < size; ++i) {
		result.data[i] = lhs.data[i] / rhs.data[i];
	}
	return result;
}

Matrix operator / (const Matrix& m, const double d) {
	Matrix result(m.row, m.col);
	const int size = m.row * m.col;
	for (int i = 0; i < size; ++i) {
		result.data[i] = m.data[i] / d;
	}
	return result;
}

Matrix operator/(const double d, const Matrix& m) {
	Matrix result(m.row, m.col);
	const int size = m.row * m.col;
	for (int i = 0; i < size; ++i) {
		result.data[i] = d / m.data[i];
	}
	return result;
}


Matrix Matrix::Dot(const Matrix& lhs, const Matrix& rhs) {
	if (lhs.col != rhs.row) {
		throw std::runtime_error("[Dot]not match shape");
	}
	const Matrix rhsT = rhs.T();
	Matrix result(lhs.row, rhsT.row);
	for (int i = 0; i < lhs.row; ++i) {
		for (int j = 0; j < rhsT.row; ++j) {
			result.data[i * rhsT.row + j] =
				std::inner_product(lhs.cbegin(i), lhs.cend(i), rhsT.cbegin(j), 0.0);
		}
	}
	return result;
}

void Matrix::Dot(const Matrix& lhs, const Matrix& rhs, Matrix& result) {
	if (lhs.col != rhs.row) {
		throw std::runtime_error("[Dot]not match shape");
	}
	const Matrix rhsT = rhs.T();
	result.Resize(lhs.row, rhsT.row);
	for (int i = 0; i < lhs.row; ++i) {
		for (int j = 0; j < rhsT.row; ++j) {
			result.data[i * rhsT.row + j] =
				std::inner_product(lhs.cbegin(i), lhs.cend(i), rhsT.cbegin(j), 0.0);
		}
	}
}

Matrix& Matrix::operator=(const Matrix& m) noexcept {
	if (row != m.row || col != m.col) {
		Resize(m.row, m.col);
	}
	std::copy(m.begin(), m.end(), data);
	return *this;
}

Matrix& Matrix::operator=(Matrix&& m) noexcept
{
	delete[] data;
	data = m.data;
	row = m.row;
	col = m.col;
	capacity = m.capacity;
	m.data = nullptr;
	return *this;
}

Matrix& Matrix::operator+=(const Matrix& m) {
	if (row != m.row || col != m.col) {
		throw std::runtime_error("[operator +=]different shape");
	}
	const int size = row * col;
	for (int i = 0; i < size; ++i) {
		this->data[i] += m.data[i];
	}
	return *this;
}

Matrix& Matrix::operator-=(const Matrix& m) {
	if (row != m.row || col != m.col) {
		throw std::runtime_error("[operator -=]different shape");
	}
	const int size = row * col;
	for (int i = 0; i < size; ++i) {
		this->data[i] -= m.data[i];
	}
	return *this;
}

Matrix& Matrix::operator*=(const Matrix& m) {
	if (row != m.row || col != m.col) {
		throw std::runtime_error("[operator *=]different shape");
	}
	const int size = row * col;
	for (int i = 0; i < size; ++i) {
		this->data[i] *= m.data[i];
	}
	return *this;
}

Matrix& Matrix::operator*=(double d)
{
	for (double& i : *this) {
		i *= d;
	}
	return *this;
}

Matrix& Matrix::operator/=(const Matrix& m)
{
	if (row != m.row || col != m.col) {
		throw std::runtime_error("[operator /=]different shape");
	}
	const int size = row * col;
	for (int i = 0; i < size; ++i) {
		this->data[i] /= m.data[i];
	}
	return *this;
}

Matrix& Matrix::operator/=(double d)
{
	for (double& i : *this) {
		i /= d;
	}
	return *this;
}

bool operator==(const Matrix& lhs, const Matrix& rhs) noexcept {
	if (lhs.row != rhs.row) return false;
	if (lhs.col != rhs.col) return false;
	const int size = lhs.row * lhs.col;
	for (int i = 0; i < size; ++i) {
		if (lhs.data[i] != rhs.data[i]) return false;
	}
	return true;
}

bool operator != (const Matrix& lhs, const Matrix& rhs) noexcept {
	return !(lhs == rhs);
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