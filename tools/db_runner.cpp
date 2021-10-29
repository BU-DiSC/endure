#include <chrono>
#include <ctime>
#include <iostream>
#include <random>
#include <regex>
#include <unistd.h>

#include "clipp.h"
#include "spdlog/spdlog.h"

#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/table.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/env.h"
#include "rocksdb/iostats_context.h"
#include "rocksdb/perf_context.h"

#include "tmpdb/fluid_lsm_compactor.hpp"
#include "infrastructure/data_generator.hpp"

#define PAGESIZE 4096

typedef struct environment
{
    std::string db_path;

    size_t non_empty_reads = 0;
    size_t empty_reads = 0;
    size_t range_reads = 0;
    size_t writes = 0;
    size_t prime_reads = 0;

    int rocksdb_max_levels = 16;
    int parallelism = 1;

    int compaction_readahead_size = 64;
    int seed = 42;
    int max_open_files = 512;

    std::string write_out_path;
    bool write_out = false;

    int verbose = 0;

    bool prime_db = false;

    std::string key_file;
    bool use_key_file = false;
} environment;


environment parse_args(int argc, char * argv[])
{
    using namespace clipp;
    using std::to_string;

    environment env;
    bool help = false;

    auto general_opt = "general options" % (
        (option("-v", "--verbose") & integer("level", env.verbose))
            % ("Logging levels (DEFAULT: INFO, 1: DEBUG, 2: TRACE)"),
        (option("-h", "--help").set(help, true)) % "prints this message"
    );

    auto execute_opt = "execute options:" % (
        (value("db_path", env.db_path)) % "path to the db",
        (option("-e", "--empty_reads") & integer("num", env.empty_reads))
            % ("empty queries, [default: " + to_string(env.empty_reads) + "]"),
        (option("-r", "--non_empty_reads") & integer("num", env.non_empty_reads))
            % ("non-empty queries, [default: " + to_string(env.non_empty_reads) + "]"),
        (option("-q", "--range_reads") & integer("num", env.range_reads))
            % ("range reads, [default: " + to_string(env.range_reads) + "]"),
        (option("-w", "--writes") & integer("num", env.writes))
            % ("empty queries, [default: " + to_string(env.writes) + "]"),
        (option("-o", "--output").set(env.write_out) & value("file", env.write_out_path))
            % ("optional write out all recorded times [default: off]"),
        (option("-p", "--prime").set(env.prime_db) & value("num", env.prime_reads))
            % ("optional warm up the database with reads [default: off]")
    );

    auto minor_opt = "minor options:" % (
        (option("--parallelism") & integer("threads", env.parallelism))
            % ("Threads allocated for RocksDB [default: " + to_string(env.parallelism) + "]"),
        (option("--compact-readahead") & integer("size", env.compaction_readahead_size))
            % ("Use 2048 for HDD, 64 for flash [default: " + to_string(env.compaction_readahead_size) + "]"),
        (option("--rand_seed") & integer("seed", env.seed))
            % ("Random seed for experiment reproducability [default: " + to_string(env.seed) + "]"),
        (option("--key-file").set(env.use_key_file, true) & value("file", env.key_file))
            % "use keyfile to speed up bulk loading"
    );

    auto cli = (
        general_opt,
        execute_opt,
        minor_opt
    );

    if (!parse(argc, argv, cli) || help)
    {
        auto fmt = doc_formatting{}.doc_column(42);
        std::cout << make_man_page(cli, "exp_robust", fmt);
        exit(EXIT_FAILURE);
    }

    return env;
}


rocksdb::Status open_db(environment env,
    tmpdb::FluidOptions *& fluid_opt,
    tmpdb::FluidLSMCompactor *& fluid_compactor,
    rocksdb::Options & rocksdb_opt,
    rocksdb::DB *& db)
{
    spdlog::debug("Opening database");
    // rocksdb::Options rocksdb_opt;
    // rocksdb_opt.statistics = rocksdb::CreateDBStatistics();
    fluid_opt = new tmpdb::FluidOptions(env.db_path + "/fluid_config.json");

    rocksdb_opt.create_if_missing = false;
    rocksdb_opt.error_if_exists = false;
    rocksdb_opt.compaction_style = rocksdb::kCompactionStyleNone;
    rocksdb_opt.compression = rocksdb::kNoCompression;

    rocksdb_opt.use_direct_reads = true;
    rocksdb_opt.num_levels = env.rocksdb_max_levels;
    rocksdb_opt.IncreaseParallelism(env.parallelism);

    rocksdb_opt.write_buffer_size = fluid_opt->buffer_size; //> "Level 0" or the in memory buffer
    rocksdb_opt.num_levels = env.rocksdb_max_levels;

    // Disable and enable certain settings for closer to vanilla LSM 
    rocksdb_opt.use_direct_reads = true;
    rocksdb_opt.use_direct_io_for_flush_and_compaction = true;
    rocksdb_opt.max_open_files = env.max_open_files;
    rocksdb_opt.advise_random_on_open = false;
    rocksdb_opt.random_access_max_buffer_size = 0;
    rocksdb_opt.avoid_unnecessary_blocking_io = true;

    // Prevent rocksdb from limiting file size as we manage it ourselves
    rocksdb_opt.target_file_size_base = UINT64_MAX;

    // Note that level 0 in RocksDB is the first level on disk.
    // Here we want level 0 to contain T sst files before trigger a compaction. 
    rocksdb_opt.level0_file_num_compaction_trigger = fluid_opt->lower_level_run_max + 1;

    // Number of files in level 0 to slow down writes. Since we're prioritizing compactions we will wait for those to
    // finish up first by slowing down the write speed
    rocksdb_opt.level0_slowdown_writes_trigger = 2 * (fluid_opt->lower_level_run_max + 1);
    rocksdb_opt.level0_stop_writes_trigger = 3 * (fluid_opt->lower_level_run_max + 1);

    fluid_compactor = new tmpdb::FluidLSMCompactor(*fluid_opt, rocksdb_opt);
    rocksdb_opt.listeners.emplace_back(fluid_compactor);

    rocksdb::BlockBasedTableOptions table_options;
    if (fluid_opt->levels > 0)
    {
        table_options.filter_policy.reset(
            rocksdb::NewMonkeyFilterPolicy(
                fluid_opt->bits_per_element,
                fluid_opt->size_ratio,
                fluid_opt->levels + 1));
    }
    else
    {
        table_options.filter_policy.reset(
            rocksdb::NewMonkeyFilterPolicy(
                fluid_opt->bits_per_element,
                fluid_opt->size_ratio, 
                tmpdb::FluidLSMCompactor::estimate_levels(
                    fluid_opt->num_entries,
                    fluid_opt->size_ratio,
                    fluid_opt->entry_size,
                    fluid_opt->buffer_size) + 1));
    }
    table_options.no_block_cache = true;
    rocksdb_opt.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_options));

    rocksdb::Status status = rocksdb::DB::Open(rocksdb_opt, env.db_path, &db);
    if (!status.ok())
    {
        spdlog::error("Problems opening DB");
        spdlog::error("{}", status.ToString());
        delete db;
        return status;
    }

    return status;
}


std::vector<std::string> get_all_valid_keys(environment env)
{
    // TODO: In reality we should be saving the list of all keys while building the DB to speed up testing. No need to
    // go through and retrieve all keys manually
    spdlog::debug("Grabbing existing keys");
    std::vector<std::string> existing_keys;
    std::string key;
    std::ifstream key_file(env.db_path + "/existing_keys.data");

    if (key_file.is_open())
    {
        while (std::getline(key_file, key))
        {
            existing_keys.push_back(key);
        }
    }

    std::sort(existing_keys.begin(), existing_keys.end());

    return existing_keys;
}


void append_valid_keys(environment env, std::vector<std::string> & new_keys)
{
    spdlog::debug("Adding new keys to existing key file");
    std::ofstream key_file;

    key_file.open(env.db_path + "/existing_keys.data", std::ios::app);

    for (auto key : new_keys)
    {
        key_file << key << std::endl;
    }

    key_file.close();
}


int run_random_non_empty_reads(environment env, std::vector<std::string> existing_keys, rocksdb::DB * db)
{
    spdlog::info("{} Non-Empty Reads", env.non_empty_reads);
    rocksdb::Status status;

    std::string value;
    std::mt19937 engine;
    std::uniform_int_distribution<int> dist(0, existing_keys.size() - 1);

    auto non_empty_read_start = std::chrono::high_resolution_clock::now();
    for (size_t read_count = 0; read_count < env.non_empty_reads; read_count++)
    {
        status = db->Get(rocksdb::ReadOptions(), existing_keys[dist(engine)], &value);
    }
    auto non_empty_read_end = std::chrono::high_resolution_clock::now();
    auto non_empty_read_duration = std::chrono::duration_cast<std::chrono::milliseconds>(non_empty_read_end - non_empty_read_start);
    spdlog::info("Non empty read time elapsed : {} ms", non_empty_read_duration.count());

    return non_empty_read_duration.count();
}


int run_random_empty_reads(environment env, rocksdb::DB * db, tmpdb::FluidOptions * fluid_opt)
{
    spdlog::info("{} Empty Reads", env.empty_reads);
    rocksdb::Status status;

    std::string value, key;
    std::mt19937 engine;

    DataGenerator * data_gen;
    if (env.use_key_file)
    {
        data_gen = new KeyFileGenerator(env.key_file, fluid_opt->num_entries, env.empty_reads, 0);
    }
    else
    {
        spdlog::warn("No keyfile, empty reads are not guaranteed");
        data_gen = new RandomGenerator();
    }

    auto empty_read_start = std::chrono::high_resolution_clock::now();
    for (size_t read_count = 0; read_count < env.empty_reads; read_count++)
    {
        status = db->Get(rocksdb::ReadOptions(), data_gen->generate_key(""), &value);
    }
    auto empty_read_end = std::chrono::high_resolution_clock::now();
    auto empty_read_duration = std::chrono::duration_cast<std::chrono::milliseconds>(empty_read_end - empty_read_start);
    spdlog::info("Empty read time elapsed : {} ms", empty_read_duration.count());

    return empty_read_duration.count();
}


int run_range_reads(environment env,
                    std::vector<std::string> existing_keys,
                    tmpdb::FluidOptions * fluid_opt,
                    rocksdb::DB * db)
{
    spdlog::info("{} Range Queries", env.range_reads);
    rocksdb::ReadOptions read_opt;
    rocksdb::Status status;
    std::string lower_key, upper_key;
    int key_idx, valid_keys = 0;

    // We use existing keys to 100% enforce all range queries to be short range queries
    int key_hop = (PAGESIZE / fluid_opt->entry_size);
    spdlog::debug("Keys per range query : {}", key_hop);

    std::string value;
    std::mt19937 engine;
    std::uniform_int_distribution<int> dist(0, existing_keys.size() - 1 - key_hop);

    read_opt.fill_cache = false;
    read_opt.total_order_seek = true;

    auto range_read_start = std::chrono::high_resolution_clock::now();
    for (size_t range_count = 0; range_count < env.range_reads; range_count++)
    {
        key_idx = dist(engine);
        lower_key = existing_keys[key_idx];
        upper_key = existing_keys[key_idx + key_hop];
        read_opt.iterate_upper_bound = new rocksdb::Slice(upper_key);
        rocksdb::Iterator * it = db->NewIterator(read_opt);
        for (it->Seek(rocksdb::Slice(lower_key)); it->Valid(); it->Next())
        {
            value = it->value().ToString();
            valid_keys++;
        }
        delete it;
    }
    auto range_read_end = std::chrono::high_resolution_clock::now();
    auto range_read_duration = std::chrono::duration_cast<std::chrono::milliseconds>(range_read_end - range_read_start);
    spdlog::info("Range reads time elapsed : {} ms", range_read_duration.count());
    spdlog::trace("Valid Keys {}", valid_keys);

    return range_read_duration.count();
}



std::pair<int, int> run_random_inserts(environment env,
                       tmpdb::FluidOptions * fluid_opt,
                       tmpdb::FluidLSMCompactor * fluid_compactor,
                       rocksdb::DB * db)
{
    spdlog::info("{} Write Queries", env.writes);
    rocksdb::WriteOptions write_opt;
    rocksdb::Status status;
    std::vector<std::string> new_keys;
    write_opt.sync = false;
    write_opt.low_pri = true; //> every insert is less important than compaction
    write_opt.disableWAL = true; 
    write_opt.no_slowdown = false; //> enabling this will make some insertions fail

    int max_writes_failed = env.writes * 0.1;
    int writes_failed = 0;

    spdlog::debug("Writing {} key-value pairs", env.writes);
    KeyFileGenerator data_gen(env.key_file, fluid_opt->num_entries, env.writes, 0);

    auto start_write_time = std::chrono::high_resolution_clock::now();
    for (size_t write_idx = 0; write_idx < env.writes; write_idx++)
    {
        std::pair<std::string, std::string> entry = data_gen.generate_kv_pair(fluid_opt->entry_size);
        new_keys.push_back(entry.first);
        status = db->Put(write_opt, entry.first, entry.second);
        if (!status.ok())
        {
            spdlog::warn("Unable to put key {}", write_idx);
            spdlog::error("{}", status.ToString());
            writes_failed++;
            if (writes_failed > max_writes_failed)
            {
                spdlog::error("10\% of total writes have failed, aborting");
                db->Close();
                delete db;
                exit(EXIT_FAILURE);
            }
        }
    }
    auto end_write_time = std::chrono::high_resolution_clock::now();
    auto write_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_write_time - start_write_time);

    auto remaining_compactions_start = std::chrono::high_resolution_clock::now();
    // We perform one more flush and wait for any last minute remaining compactions due to RocksDB interntally renaming
    // SST files during parallel compactions
    spdlog::debug("Flushing DB...");
    rocksdb::FlushOptions flush_opt;
    flush_opt.wait = true;
    flush_opt.allow_write_stall = true;

    db->Flush(flush_opt);

    spdlog::debug("Waiting for all remaining background compactions to finish before after writes");
    while(fluid_compactor->compactions_left_count > 0);

    spdlog::debug("Checking final state of the tree and if it requires any compactions...");
    while(fluid_compactor->requires_compaction(db))
    {
        while(fluid_compactor->compactions_left_count > 0);
    }

    auto remaining_compactions_end= std::chrono::high_resolution_clock::now();
    auto remaining_compactions_duration = std::chrono::duration_cast<std::chrono::milliseconds>(remaining_compactions_end - remaining_compactions_start);
    spdlog::info("Write time elapsed : {} ms", write_duration.count());

    append_valid_keys(env, new_keys);
    fluid_opt->num_entries += new_keys.size();

    return std::pair<int, int>(write_duration.count(), remaining_compactions_duration.count());
}


int prime_database(environment env, rocksdb::DB * db)
{
    rocksdb::ReadOptions read_opt;
    rocksdb::Status status;

    std::string value;
    std::mt19937 engine;
    std::uniform_int_distribution<int> dist(0, 2 * KEY_DOMAIN);

    spdlog::info("Priming database with {} reads", env.prime_reads);
    for (size_t read_count = 0; read_count < env.prime_reads; read_count++)
    {
        status = db->Get(read_opt, std::to_string(dist(engine)), &value);
    }

    return 0;
}


void print_db_status(rocksdb::DB * db)
{
    spdlog::debug("Files per level");
    rocksdb::ColumnFamilyMetaData cf_meta;
    db->GetColumnFamilyMetaData(&cf_meta);

    std::vector<std::string> file_names;
    int level_idx = 1;
    for (auto & level : cf_meta.levels)
    {
        std::string level_str = "";
        for (auto & file : level.files)
        {
            level_str += file.name + ", ";
        }
        level_str = level_str == "" ? "EMPTY" : level_str.substr(0, level_str.size() - 2);
        spdlog::debug("Level {} : {} Files : {}", level_idx, level.files.size(), level_str);
        level_idx++;
    }
}


int main(int argc, char * argv[])
{
    spdlog::set_pattern("[%T.%e]%^[%l]%$ %v");
    environment env = parse_args(argc, argv);

    spdlog::info("Welcome to the db_runner!");
    if(env.verbose == 1)
    {
        spdlog::info("Log level: DEBUG");
        spdlog::set_level(spdlog::level::debug);
    }
    else if(env.verbose == 2)
    {
        spdlog::info("Log level: TRACE");
        spdlog::set_level(spdlog::level::trace);
    }
    else
    {
        spdlog::set_level(spdlog::level::info);
    }

    rocksdb::DB * db = nullptr;
    tmpdb::FluidOptions * fluid_opt = nullptr;
    tmpdb::FluidLSMCompactor * fluid_compactor = nullptr;

    rocksdb::Options rocksdb_opt;
    rocksdb_opt.statistics = rocksdb::CreateDBStatistics();
    rocksdb::Status status = open_db(env, fluid_opt, fluid_compactor, rocksdb_opt, db);
    rocksdb::SetPerfLevel(rocksdb::PerfLevel::kEnableTimeExceptForMutex);

    if (env.prime_db)
    {
        prime_database(env, db);
    }

    int empty_read_duration = 0, read_duration = 0, range_duration = 0;
    int write_duration = 0, compact_duration = 0;
    std::vector<std::string> existing_keys;
    
    if ((env.non_empty_reads > 0) || (env.range_reads > 0))
    {
        existing_keys = get_all_valid_keys(env);
    }

    rocksdb_opt.statistics->Reset();
    rocksdb::get_iostats_context()->Reset();
    rocksdb::get_perf_context()->Reset();
    if (env.empty_reads > 0)
    {
        empty_read_duration = run_random_empty_reads(env, db, fluid_opt); 
    }

    if (env.non_empty_reads > 0)
    {
        read_duration = run_random_non_empty_reads(env, existing_keys, db);
    }

    if (env.range_reads > 0)
    {
        range_duration = run_range_reads(env, existing_keys, fluid_opt, db);
    }

    if (env.writes > 0)
    {
        std::pair<int, int> inserts_duration = run_random_inserts(env, fluid_opt, fluid_compactor, db);
        write_duration = inserts_duration.first;
        compact_duration = inserts_duration.second;
    }

    if (spdlog::get_level() <= spdlog::level::debug)
    {
        print_db_status(db);
    }

    // std::cout << rocksdb_opt.statistics->ToString() << std::endl;
    // std::cout << rocksdb::get_perf_context()->ToString() << std::endl;
    // std::cout << rocksdb::get_iostats_context()->ToString() << std::endl;

    std::map<std::string, uint64_t> stats;
    rocksdb_opt.statistics->getTickerMap(&stats);

    spdlog::info("(l0, l1, l2plus) : ({}, {}, {})",
        stats["rocksdb.l0.hit"],
        stats["rocksdb.l1.hit"],
        stats["rocksdb.l2andup.hit"]);
    spdlog::info("(bf_true_neg, bf_pos, bf_true_pos) : ({}, {}, {})",
        stats["rocksdb.bloom.filter.useful"],
        stats["rocksdb.bloom.filter.full.positive"],
        stats["rocksdb.bloom.filter.full.true.positive"]);
    spdlog::info("(bytes_written, compact_read, compact_write, flush_write) : ({}, {}, {}, {})", 
        stats["rocksdb.bytes.written"],
        stats["rocksdb.compact.read.bytes"],
        stats["rocksdb.compact.write.bytes"],
        stats["rocksdb.flush.write.bytes"]);
    spdlog::info("(block_read_count) : ({})", rocksdb::get_perf_context()->block_read_count);
    spdlog::info("(z0, z1, q, w) : ({}, {}, {}, {})", empty_read_duration, read_duration, range_duration, write_duration);
    spdlog::info("(remaining_compactions_duration) : ({})", compact_duration);

    rocksdb::ColumnFamilyMetaData cf_meta;
    db->GetColumnFamilyMetaData(&cf_meta);

    std::string run_per_level = "[";
    for (auto & level : cf_meta.levels)
    {
        run_per_level += std::to_string(level.files.size()) + ", ";
    }
    run_per_level = run_per_level.substr(0, run_per_level.size() - 2) + "]";
    spdlog::info("runs_per_level : {}", run_per_level);
    fluid_opt->write_config(env.db_path + "/fluid_config.json");

    db->Close();
    delete db;

    return 0;
}
