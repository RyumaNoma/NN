#pragma once
#include <fstream>
#include <string>
/*
�ł��邱��
�E�o�͐�t�@�C���̎w��
�E������̏o��
�E�������y�ɏo�͂�����
�E�X�y�[�X�󂫂ŕ\��������
*/
class Debug
{
public:
	static std::string FILENAME;

	// �t�@�C���̓��e�����Z�b�g����
	static void Reset();

	// ������݂̂̕\��
	static void Print(const std::string& msg);

	// ������ƕϐ��̓��e��\��
	template<class Ty>
	static void Print(const std::string& msg, const Ty& value);
	
	// �X�y�[�X�󂫂ŘA���\��
	// �s���ɃX�y�[�X�Ɖ��s���o�͂����
	template<class Head, class... Tail>
	static void Print(Head&& head, Tail&& ...tail);

	// �Ō�̏o�͗p
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