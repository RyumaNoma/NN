#pragma once
#include <iostream>
#include <vector>
#include <tuple>

// *演算子はアダマール積
// TODO: matrixをサイズ固定にするかを決める
class Matrix {
public:
	using ShapeType = std::tuple<int, int>;
	Matrix() noexcept;
	Matrix(const int row, const int col) noexcept;
	Matrix(const ShapeType& shape) noexcept;
	Matrix(const Matrix& m) noexcept;
	Matrix(Matrix&& m) noexcept;
	Matrix(const int row, const int col, double const* rawData) noexcept;
	Matrix(const int row, const int col, double fill) noexcept;
	Matrix(const int row, const int col, const std::vector<int>& flattenData) noexcept;
	~Matrix();

	void Resize(const int newRow, const int newCol);
	void Reshape(const int newRow, const int newCol);
	void Reshape(const ShapeType& shape);

	int Row() const noexcept { return row; }
	int Col() const noexcept { return col; }
	int Size() const noexcept { return row * col; }
	int Capacity() const noexcept { return capacity; }
	ShapeType Shape() const noexcept { return ShapeType(row, col); }

	Matrix T() const noexcept;
	Matrix Flatten() const noexcept;

	double Sum() const noexcept;
	// TODO: 向きの変更
	Matrix VerticalSum() const noexcept;
	Matrix HorizontalSum() const noexcept;

	double Max() const noexcept;
	Matrix VerticalMax() const noexcept;
	Matrix HorizontalMax() const noexcept;

	double Min() const noexcept;
	Matrix VerticalMin() const noexcept;
	Matrix HorizontalMin() const noexcept;

	double Average() const noexcept;

	friend Matrix Abs(const Matrix& m) noexcept;

	inline double* begin() const { return data; }
	inline double* end() const { return data + (row * col); }
	inline double* cbegin(const int row) const { return data + (row * this->col); }
	inline double* cend(const int row) const { return data + (row * this->col + this->col); }

	friend Matrix operator + (const Matrix& lhs, const Matrix& rhs);
	friend Matrix operator + (const Matrix& m, const double d);
	friend Matrix operator + (const double d, const Matrix& m);

	friend Matrix operator - (const Matrix& lhs, const Matrix& rhs);
	friend Matrix operator - (const Matrix& m, const double d);
	friend Matrix operator - (const double d, const Matrix& m);

	friend Matrix operator * (const Matrix& lhs, const Matrix& rhs);
	friend Matrix operator * (const Matrix& m, const double d);
	friend Matrix operator * (const double d, const Matrix& m);

	friend Matrix operator / (const Matrix& lhs, const Matrix& rhs);
	friend Matrix operator / (const Matrix& m, const double d);
	friend Matrix operator / (const double d, const Matrix& m);

	static Matrix Dot(const Matrix& lhs, const Matrix& rhs);
	static void Dot(const Matrix& lhs, const Matrix& rhs, Matrix& result);

	Matrix& operator = (const Matrix& m) noexcept;
	Matrix& operator = (Matrix&& m) noexcept;
	Matrix& operator += (const Matrix& m);
	Matrix& operator -= (const Matrix& m);
	Matrix& operator *= (const Matrix& m);
	Matrix& operator *= (double d);
	Matrix& operator /= (const Matrix& m);
	Matrix& operator /= (double d);
	friend bool operator == (const Matrix& lhs, const Matrix& rhs) noexcept;
	friend bool operator != (const Matrix& lhs, const Matrix& rhs) noexcept;
	inline double& operator () (int i) const {
		if (i >= Size()) {
			throw std::runtime_error("[operator (i)]out of data");
		}
		return data[i];
	}
	inline double& operator () (int i, int j) const {
		if (i * col + j >= Size()) {
			throw std::runtime_error("[operator (i,j)]out of data");
		}
		return data[i * col + j];
	}
	friend std::ostream& operator << (std::ostream& os, const Matrix& m);
	friend class TestMatrix;
private:
	int row;
	int col;
	int capacity;
	double* data;
};