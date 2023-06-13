#pragma once
#include <iostream>
#include <vector>

// *���Z�q�̓A�_�}�[����
class Matrix {
public:
	Matrix() noexcept;
	Matrix(int row, int col) noexcept;
	Matrix(const std::vector<int>& shape) noexcept;
	Matrix(const Matrix& m) noexcept;
	Matrix(int row, int col, double const* d) noexcept;
	Matrix(int row, int col, double fill) noexcept;
	~Matrix();

	// ATTENTION: �K�����̃f�[�^�̍폜�ƐV�����̈�̊m�ۂ��s��
	// ���̃f�[�^���c��Ȃ�
	void Resize(int newRow, int newCol);
	void Reshape(int newRow, int newCol);
	void Reshape(const std::vector<int>& shape);

	int Row() const noexcept { return row; }
	int Col() const noexcept { return col; }
	int Size() const noexcept { return row * col; }
	std::vector<int> Shape() const noexcept { return { row, col }; }

	Matrix T() const;
	Matrix Flatten() const;

	double Sum() const;
	Matrix VerticalSum() const;
	Matrix HorizontalSum() const;

	double Max() const;
	Matrix VerticalMax() const;
	Matrix HorizontalMax() const;

	double Min() const;
	Matrix VerticalMin() const;
	Matrix HorizontalMin() const;

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