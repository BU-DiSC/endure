#ifndef DATA_GENERATOR_H_
#define DATA_GENERATOR_H_

#include <cassert>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "zipf.hpp"
#include "spdlog/spdlog.h"

#define KEY_DOMAIN 1000000000

class Distribution
{
public:
    virtual ~Distribution() {}
    virtual int gen(std::mt19937 & engine) = 0;
};


class Uniform : public Distribution
{
public:
    Uniform(int max);
    ~Uniform() {}
    int gen(std::mt19937 & engine);
private:
    std::uniform_int_distribution<int> dist;
};


class Zipf : public Distribution
{
public:
    Zipf(int max);
    ~Zipf() {}
    int gen(std::mt19937 & engine);
private:
    opencog::zipf_distribution<int, double> dist;
};


class DataGenerator
{
public:
    int seed;

    virtual ~DataGenerator() {}

    virtual std::string gen_key() = 0;

    virtual std::string gen_val(size_t value_size) = 0;

    virtual std::string gen_new_dup_key() = 0;

    virtual std::string gen_existing_key() = 0;

    std::pair<std::string, std::string> gen_kv_pair(size_t kv_size);
};


class RandomGenerator : public DataGenerator
{
public:
    RandomGenerator(int seed);
    RandomGenerator() : RandomGenerator(0) {}
    ~RandomGenerator() {}

    std::string gen_key();

    std::string gen_val(size_t value_size);

    std::string gen_new_dup_key() {return this->gen_key();}

    std::string gen_existing_key() {return this->gen_key();}

private:
    // We generate a distribution with a large gap in the middle in order fo the test suite to have the functionality of
    // giving keys that are still in the domain but gurantee an empty read
    std::uniform_int_distribution<int> dist;
    std::mt19937 engine;
};


class KeyFileGenerator : public DataGenerator
{
public:
    KeyFileGenerator(std::string key_file, int start_idx, int num_keys, int seed, std::string mode);
    KeyFileGenerator(std::string key_file, int num_keys, int seed, std::string mode)
        : KeyFileGenerator(key_file, num_keys, num_keys, seed, mode) {}

    ~KeyFileGenerator();

    std::string gen_key();

    std::string gen_val(size_t value_size);

    std::string gen_new_dup_key();

    std::string gen_existing_key();

private:
    std::string mode;
    std::mt19937 engine;
    std::vector<int>::iterator key_gen;
    std::vector<int> keys;
    std::vector<int> existing_keys;
    Distribution * dist_new;
    Distribution * dist_existing;

    void read_file(std::string key_file, int offset, int num_keys);
};

#endif /* DATA_GENERATOR_H_ */
