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
	Matrix(int row, int col) noexcept;
	Matrix(const ShapeType& shape) noexcept;
	Matrix(const Matrix& m) noexcept;
	Matrix(int row, int col, double const* rawData) noexcept;
	Matrix(int row, int col, double fill) noexcept;
	Matrix(int row, int col, const std::vector<int>& flattenData) noexcept;
	~Matrix();

	// ATTENTION: 必ず元のデータの削除と新しい領域の確保を行う
	// 元のデータも残らない
	// TODO: vectorのresizeと同じような実装
	void Resize(int newRow, int newCol);
	void Reshape(int newRow, int newCol);
	void Reshape(const ShapeType& shape);

	int Row() const noexcept { return row; }
	int Col() const noexcept { return col; }
	int Size() const noexcept { return row * col; }
	ShapeType Shape() const noexcept { return ShapeType(row, col); }

	Matrix T() const;
	Matrix Flatten() const;

	double Sum() const;
	// TODO: 向きの変更
	Matrix VerticalSum() const;
	Matrix HorizontalSum() const;

	double Max() const;
	Matrix VerticalMax() const;
	Matrix HorizontalMax() const;

	double Min() const;
	Matrix VerticalMin() const;
	Matrix HorizontalMin() const;

	double Average() const;

	friend Matrix Abs(const Matrix& m);

	Matrix operator + (const Matrix& m) const;
	friend Matrix operator + (double d, const Matrix& m);
	Matrix operator + (double d) const;
	Matrix operator - (const Matrix& m) const;
	Matrix operator - (double d) const;
	friend Matrix operator - (double d, const Matrix& m);
	Matrix operator * (const Matrix& m) const;
	Matrix operator * (double coef) const;
	friend Matrix operator * (double coef, const Matrix& m);
	Matrix operator / (const Matrix& m) const;
	Matrix operator / (double coef) const;
	friend Matrix operator / (double d, const Matrix& m);
	static Matrix Dot(const Matrix& left, const Matrix& right);

	Matrix& operator = (const Matrix& m);
	Matrix& operator += (const Matrix& m);
	Matrix& operator -= (const Matrix& m);
	Matrix& operator *= (const Matrix& m);
	Matrix& operator *= (double coef);
	Matrix& operator /= (const Matrix& m);
	Matrix& operator /= (double coef);
	bool operator == (const Matrix& m) const noexcept;
	bool operator != (const Matrix& m) const noexcept;
	double& operator () (int i) const;
	double& operator () (int i, int j) const;
	friend std::ostream& operator << (std::ostream& os, const Matrix& m);
	friend class TestMatrix;
private:
	int row;
	int col;
	double* data;
};