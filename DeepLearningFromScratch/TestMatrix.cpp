#include "TestMatrix.hpp"
#include "Matrix.hpp"
#include <iostream>
#include <cassert>

void TestMatrix::Constructor() {
	Matrix m1(3, 2);
	std::cerr << m1 << std::endl;

	Matrix m2(3, 2, 0.0);
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 2; ++j) {
			assert(m2(i, j) == 0.0);
		}
	}

	double raw[5] = { 1.0, 2.0, 4.0, 5.0, 3.0 };
	Matrix m3(1, 5, raw);
	for (int i = 0; i < 5; ++i) {
		assert(m3(i) == raw[i]);
	}

	Matrix m4(m2);
	assert(m4.Shape() == m2.Shape());
	assert(m2 == m4);
	assert(m2.data != m4.data);

	Matrix m5(0, 5);
	assert(m5.data == nullptr);
	std::cerr << "finish Constructor test" << std::endl;
}

void TestMatrix::Resize() {
	Matrix m1(2, 4, 5.0);
	m1.Resize(4, 5);
	assert(m1.Row() == 4);
	assert(m1.Col() == 5);

	m1.Resize(1, 2);
	assert(m1.Row() == 1);
	assert(m1.Col() == 2);
	std::cerr << "finish resize test" << std::endl;
}

void TestMatrix::Add() {
	double data[6] = {1.0, 2, 3, 4, 5, 6};
	Matrix a(2, 3, data);
	Matrix b = a + a + a;
	for (int i = 0; i < 6; ++i) {
		assert(a(i) * 3 == b(i));
	}

	Matrix c(1, 3);
	try {
		Matrix d = a + c;
	}
	catch (const std::exception& e) {
		std::cerr << "Add: " << e.what() << std::endl;
	}
	std::cerr << "finish add test" << std::endl;
}

void TestMatrix::Sub() {
	double data[6] = { 1, 2, 3, 4, 5, 6 };
	double data2[6] = { 7, 8, 9, 10, 11, 12 };
	Matrix a(2, 3, data);
	Matrix b(2, 3, data2);
	Matrix c = a - b;
	for (int i = 0; i < 6; ++i) {
		assert(c(i) == data[i] - data2[i]);
	}

	Matrix d(1, 3);
	try {
		Matrix e = a - d;
	}
	catch (const std::exception& e) {
		std::cerr << "Sub: " << e.what() << std::endl;
	}
	std::cerr << "finish sub test" << std::endl;
}

void TestMatrix::Mul() {
	double data[6] = { 1, 2, 3, 4, 5, 6 };
	double data2[6] = { 7, 8, 9, 10, 11, 12 };
	Matrix a(2, 3, data);
	Matrix b(2, 3, data2);
	Matrix c = a * b;
	for (int i = 0; i < 6; ++i) {
		assert(c(i) == data[i] * data2[i]);
	}

	Matrix d(1, 3);
	try {
		Matrix e = a * d;
	}
	catch (const std::exception& e) {
		std::cerr << "Mul: " << e.what() << std::endl;
	}
	std::cerr << "finish mul test" << std::endl;
}

void TestMatrix::Dot() {
	double data[6] = { 1, 2, 3, 4, 5, 6 };
	double data2[6] = { 7, 8, 9, 10, 11, 12 };
	double ans[4] = { 58, 64, 139, 154 };

	Matrix a(2, 3, data);
	Matrix b(3, 2, data2);
	Matrix c = Matrix::Dot(a, b);
	assert(c.Shape() == Matrix::ShapeType({2, 2}));
	for (int i = 0; i < 4; ++i) {
		assert(c(i) == ans[i]);
	}

	Matrix d(1, 2);
	try {
		Matrix e = Matrix::Dot(a, d);
	}
	catch (const std::exception& e) {
		std::cerr << "Dot: " << e.what() << std::endl;
	}
	std::cerr << "finish dot test" << std::endl;
}

void TestMatrix::Equal() {
	Matrix a(4, 5);
	Matrix b(a);
	Matrix c;
	c = b;

	assert(b == a);
	assert(c == a);
	assert(b.data != a.data);
	assert(c.data != a.data);
	std::cerr << "finish equal test" << std::endl;
}

void TestMatrix::T()
{
	double data[6] = { 1, 2, 3, 4, 5, 6 };
	double ans[6] = { 1, 4, 2, 5, 3, 6 };
	
	Matrix a(2, 3, data);
	Matrix b = a.T();

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 2; ++j) {
			assert(b(i, j) == ans[i * 2 + j]);
		}
	}
	std::cerr << "finish T test" << std::endl;
}

void TestMatrix::Sum()
{
	double data[6] = { 1, 2, 3, 4, 5, 6 };
	double ans0[3] = { 5, 7, 9 };
	double ans1[2] = { 6, 15 };
	Matrix a(2, 3, data);
	Matrix a0(3, 1, ans0);
	Matrix a1(2, 1, ans1);

	assert(a.Sum() == 21);
	assert(a.VerticalSum() == a0);
	assert(a.HorizontalSum() == a1);

	std::cerr << "finish sum test" << std::endl;
}

void TestMatrix::Max()
{
	double data[6] = { 1, 2, 3, 4, 5, 6 };
	double dataV[3] = { 4, 5, 6 };
	double dataH[2] = { 3,6 };

	Matrix a(2, 3, data);
	Matrix ansV(1, 3, dataV);
	Matrix ansH(1, 2, dataH);

	assert(a.Max() == 6);
	assert(a.VerticalMax().Shape() == ansV.Shape());
	assert(a.VerticalMax() == ansV);
	assert(a.HorizontalMax().Shape() == ansH.Shape());
	assert(a.HorizontalMax() == ansH);
	std::cerr << "finish max test" << std::endl;
}

void TestMatrix::Min()
{
	double data[6] = { 1, 2, 3, 4, 5, 6 };
	double dataV[3] = { 1, 2, 3 };
	double dataH[2] = { 1, 4 };

	Matrix a(2, 3, data);
	Matrix ansV(1, 3, dataV);
	Matrix ansH(1, 2, dataH);

	assert(a.Min() == 1);
	assert(a.VerticalMin().Shape() == ansV.Shape());
	assert(a.VerticalMin() == ansV);
	assert(a.HorizontalMin().Shape() == ansH.Shape());
	assert(a.HorizontalMin() == ansH);
	std::cerr << "finish min test" << std::endl;
}

void TestMatrix::All() {
	Constructor();
	Resize();
	Add();
	Sub();
	Mul();
	Dot();
	Equal();
	T();
	Sum();
	Max();
	Min();
}
