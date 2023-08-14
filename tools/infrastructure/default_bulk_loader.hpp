#ifndef DEFAULT_BULK_LOADER_H_
#define DEFAULT_BULK_LOADER_H_

#include <iostream>
#include <mutex>
#include <vector>

#include "spdlog/spdlog.h"
#include "rocksdb/db.h"

#include "data_generator.hpp"


class DefaultBulkLoader
{
public:
    std::vector<std::string> keys;

    DefaultBulkLoader(
        DataGenerator &data_gen) : data_gen(data_gen) {};

    rocksdb::Status default_bulk_loader(rocksdb::DB *db, size_t num_entries, std::string db_path);

private:
    DataGenerator &data_gen;
};

#endif /*  DEFAULT_BULK_LOADER_H_ */
