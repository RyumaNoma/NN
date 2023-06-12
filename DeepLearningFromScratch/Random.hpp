#pragma once
#include <vector>

class Random
{
public:
	Random(unsigned int seed = 722);
	void SetSeed(unsigned int seed);

	unsigned int operator () ();
	// [0, MAX)
	unsigned int operator () (unsigned int MAX);
	// [l, r)
	unsigned int operator () (int l, int r);
	// Šm—¦p‚Åtrue‚ð•Ô‚·
	bool Prob(const double p);

	template<class T>
	void Shuffle(std::vector<T>& v) {
		for (int i = v.size() - 1; i >= 0; i--) {
			std::swap(v[i], v[operator()(i)]);
		}
	}
private:
	unsigned int a, b, c, d;

};

