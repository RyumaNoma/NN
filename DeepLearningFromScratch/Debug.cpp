// Debug.cpp : スタティック ライブラリ用の関数を定義します。
//
#include "Debug.hpp"

std::string Debug::FILENAME = "./Debug.txt";

void Debug::Reset()
{
	std::ofstream out(FILENAME);
	out.close();
}

void Debug::Print(const std::string& msg)
{
	std::ofstream out(FILENAME, std::ios::app);
	out << msg << std::endl;
	out.close();
}

void Debug::Print()
{
	std::ofstream out(FILENAME, std::ios::app);
	out << std::endl;
	out.close();
}