#ifndef DATA_GENERATOR_H_
#define DATA_GENERATOR_H_

#include <cassert>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "spdlog/spdlog.h"

#define KEY_DOMAIN 1000000000
#define KEY_MIDDLE_RIGHT 5600000
#define KEY_MIDDLE_LEFT 4600000
#define KEY_BOTTOM 0

class DataGenerator
{
public:
    int seed;

    virtual std::string generate_key(const std::string key_prefix) = 0;

    virtual std::string generate_val(size_t value_size, const std::string value_prefix) = 0;

    std::pair<std::string, std::string> generate_kv_pair(size_t kv_size);

    std::pair<std::string, std::string> generate_kv_pair(
        size_t kv_size,
        const std::string key_prefix,
        const std::string value_prefix);
};


class RandomGenerator : public DataGenerator
{
public:
    RandomGenerator(int seed);
    RandomGenerator();

    std::string generate_key(const std::string key_prefix);

    std::string generate_val(size_t value_size, const std::string value_prefix);

    std::string generate_rnd();

private:
    // We generate a distribution with a large gap in the middle in order fo the test suite to have the functionality of
    // giving keys that are still in the domain but gurantee an empty read
    std::uniform_int_distribution<int> dist_left;
    std::uniform_int_distribution<int> dist_right;
    std::mt19937 engine;
};


class KeyFileGenerator : public DataGenerator
{
public:
    KeyFileGenerator(std::string key_file, int num_keys, int seed);
    KeyFileGenerator(std::string key_file, int start_idx, int num_keys, int seed);

    std::string generate_key(const std::string key_prefix);

    std::string generate_val(size_t value_size, const std::string value_prefix);

private:
    std::mt19937 engine;
    std::vector<int>::iterator key_gen;
    std::vector<int> keys;
};


#endif /* DATA_GENERATOR_H_ */
