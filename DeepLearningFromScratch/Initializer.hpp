#pragma once
class Matrix;

// TODO: 初期化方法の勉強&実装
// 参照渡し？
class Initializer
{
public:
	Initializer(std::string type);
	void Initialize(Matrix& m);
};

