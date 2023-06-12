#include "Random.hpp"

Random::Random(unsigned int seed) {
	SetSeed(seed);
}

void Random::SetSeed(unsigned int seed)
{
    a = seed = 1812433253u * (seed ^ (seed >> 30));
    b = seed = 1812433253u * (seed ^ (seed >> 30)) + 1;
    c = seed = 1812433253u * (seed ^ (seed >> 30)) + 2;
    d = seed = 1812433253u * (seed ^ (seed >> 30)) + 3;
}

unsigned int Random::operator () ()
{
    unsigned int t = (a ^ (a << 11));
    a = b;
    b = c;
    c = d;
    return d = (d ^ (d >> 19)) ^ (t ^ (t >> 8));
}

unsigned int Random::operator()(unsigned int MAX)
{
    return operator()() % MAX;
}

unsigned int Random::operator () (int l, int r)
{
    return operator()() % (r - l) - l;
}

bool Random::Prob(const double p)
{
    return static_cast<unsigned int>(p * (1LL << 32)) > operator()();
}
