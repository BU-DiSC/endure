#include <iostream>
#include <random>
#include <unistd.h>

#include "spdlog/spdlog.h"
#include "zipf.hpp"

class SmallFoo
{
public:
    SmallFoo(int max) {this->dist = opencog::zipf_distribution<int, double>(max);}

    int gen(std::mt19937 & gen) {return this->dist(gen); }
private:
    opencog::zipf_distribution<int, double> dist;
};

class Foo
{
public:
    Foo(int max);

    int gen_val();

    std::mt19937 gen;
    SmallFoo * dist;
};


Foo::Foo(int max)
{
    this->gen = std::mt19937(0);
    this->dist = new SmallFoo(max);
}

int Foo::gen_val()
{
    return this->dist->gen(this->gen);
}

int main()
{
    spdlog::set_pattern("[%T.%e]%^[%l]%$ %v");
    spdlog::set_level(spdlog::level::trace);

    Foo f(100);
    for (int idx = 0; idx < 10; idx++)
    {
        spdlog::info("Roll : {}", f.gen_val());
    }

    return 0;
}
