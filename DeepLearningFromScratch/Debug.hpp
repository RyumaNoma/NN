#pragma once
#include <fstream>
#include <string>
/*
できること
・出力先ファイルの指定
・文字列の出力
・数字を楽に出力したい
・スペース空きで表示したい
*/
class Debug
{
public:
	static std::string FILENAME;

	// ファイルの内容をリセットする
	static void Reset();

	// 文字列のみの表示
	static void Print(const std::string& msg);

	// 文字列と変数の内容を表示
	template<class Ty>
	static void Print(const std::string& msg, const Ty& value);
	
	// スペース空きで連続表示
	// 行末にスペースと改行が出力される
	template<class Head, class... Tail>
	static void Print(Head&& head, Tail&& ...tail);

	// 最後の出力用
	static void Print();
};

template<class Ty>
void Debug::Print(const std::string& msg, const Ty& value)
{
	std::ofstream out(FILENAME, std::ios::app);
	out << msg << ":" << value << std::endl;
	out.close();
}

template<class Head, class... Tail>
void Debug::Print(Head&& head, Tail&& ...tail)
{
	std::ofstream out(FILENAME, std::ios::app);
	out << head << ' ';
	out.close();
	Print(std::forward<Tail>(tail)...);
}