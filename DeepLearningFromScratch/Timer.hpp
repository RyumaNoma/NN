#pragma once
#include <chrono>

/// <summary>
/// 時間管理用クラス．
/// std::chronoのラッパー
/// </summary>
class Timer
{
private:
	std::chrono::system_clock::time_point start_;
public:
	Timer();

	/// <summary>
	/// 開始時間のリセット．
	/// </summary>
	void Reset();

	/// <summary>
	/// ミリ秒単位で経過時間を求める
	/// </summary>
	/// <returns>前回Reset()を呼び出してからの経過時間</returns>
	std::chrono::milliseconds::rep ElapsedMilliseconds() const;

	/// <summary>
	/// マイクロ秒単位で経過時間を求める
	/// </summary>
	/// <returns>前回Reset()を呼び出してからの経過時間</returns>
	std::chrono::microseconds::rep ElapsedMicroseconds() const;
};