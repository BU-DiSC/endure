#include <chrono>
#include <iostream>
#include <ctime>
#include <filesystem>
#include <unistd.h>

#include "clipp.h"
#include "spdlog/spdlog.h"

#include "rocksdb/db.h"
#include "rocksdb/table.h"
#include "rocksdb/filter_policy.h"
#include "tmpdb/fluid_lsm_compactor.hpp"
#include "infrastructure/bulk_loader.hpp"
#include "infrastructure/data_generator.hpp"

typedef struct environment
{
    std::string db_path;
    tmpdb::bulk_load_type bulk_load_mode;

    // Build mode
    double T = 2;
    double K = 1;
    double Z = 1;
    size_t B = 1 << 20; //> 1 MiB
    size_t E = 1 << 10; //> 1 KiB
    double bits_per_element = 5.0;
    size_t N = 1e6;
    size_t L = 0;

    int verbose = 0;
    bool destroy_db = false;

    int max_rocksdb_levels = 16;
    int parallelism = 1;

    int seed = 0;
    tmpdb::file_size_policy file_size_policy_opt = tmpdb::file_size_policy::INCREASING;
    uint64_t fixed_file_size = std::numeric_limits<uint64_t>::max();

    bool early_fill_stop = false;

    std::string key_file;
    bool use_key_file = false;

} environment;


environment parse_args(int argc, char * argv[])
{
    using namespace clipp;
    using std::to_string;

    size_t minimum_entry_size = 32;

    environment env;
    bool help = false;

    auto general_opt = "general options" % (
        (option("-v", "--verbose") & integer("level", env.verbose))
            % ("Logging levels (DEFAULT: INFO, 1: DEBUG, 2: TRACE)"),
        (option("-h", "--help").set(help, true)) % "prints this message"
    );

    auto build_opt = (
        "build options:" % (
            (value("db_path", env.db_path)) % "path to the db",
            (option("-T", "--size-ratio") & number("ratio", env.T))
                % ("size ratio, [default: " + fmt::format("{:.0f}", env.T) + "]"),
            (option("-K", "--lower_level_lim") & number("lim", env.K))
                % ("lower levels file limit, [default: " + fmt::format("{:.0f}", env.K) + "]"),
            (option("-Z", "--last_level_lim") & number("lim", env.Z))
                % ("last level file limit, [default: " + fmt::format("{:.0f}", env.Z) + "]"),
            (option("-B", "--buffer-size") & integer("size", env.B))
                % ("buffer size (in bytes), [default: " + to_string(env.B) + "]"),
            (option("-E", "--entry-size") & integer("size", env.E))
                % ("entry size (bytes) [default: " + to_string(env.E) + ", min: 32]"),
            (option("-b", "--bpe") & number("bits", env.bits_per_element))
                % ("bits per entry per bloom filter [default: " + fmt::format("{:.1f}", env.bits_per_element) + "]"),
            (option("-d", "--destroy").set(env.destroy_db)) % "destroy the DB if it exists at the path"
        ),
        "db fill options (pick one):" % (
            one_of(
                (option("-N", "--entries").set(env.bulk_load_mode, tmpdb::bulk_load_type::ENTRIES) & integer("num", env.N))
                    % ("total entries, default pick [default: " + to_string(env.N) + "]"),
                (option("-L", "--levels").set(env.bulk_load_mode, tmpdb::bulk_load_type::LEVELS) & integer("num", env.L)) 
                    % ("total filled levels [default: " + to_string(env.L) + "]")
            )
        )
    );

    auto minor_opt = (
        "minor options:" % (
            (option("--max_rocksdb_level") & integer("num", env.max_rocksdb_levels))
                % ("limits the maximum levels rocksdb has [default: " + to_string(env.max_rocksdb_levels) + "]"),
            (option("--parallelism") & integer("num", env.parallelism))
                % ("parallelism for writing to db [default: " + to_string(env.parallelism) + "]"),
            (option("--seed") & integer("num", env.seed))
                % "seed for generating data [default: random from time]",
            (option("--early_fill_stop").set(env.early_fill_stop, true))
                % "Stops bulk loading early if N is met [default: False]",
            (option("--key-file").set(env.use_key_file, true) & value("file", env.key_file))
                % "use keyfile to speed up bulk loading"
        )
    );

    auto file_size_policy_opt =
    (
        "file size policy (pick one)" % 
        one_of(
            (
                option("--increasing_files").set(env.file_size_policy_opt, tmpdb::file_size_policy::INCREASING)
                    % "file size will match run size as LSM tree grows (default)",
                (option("--fixed_files").set(env.file_size_policy_opt, tmpdb::file_size_policy::FIXED)
                    & opt_integer("size", env.fixed_file_size))
                    % "fixed file size specified after fixed_files flag [default size MAX uint64]",
                option("--buffer_files").set(env.file_size_policy_opt, tmpdb::file_size_policy::BUFFER)
                    % "file size matches the buffer size"
            )
        )
    );

    auto cli = (
        general_opt,
        build_opt,
        minor_opt,
        file_size_policy_opt
    );

    if (!parse(argc, argv, cli))
        help = true;

    if (env.E < minimum_entry_size)
    {
        help = true;
        spdlog::error("Entry size is less than {} bytes", minimum_entry_size);
    }

    if (help)
    {
        auto fmt = doc_formatting{}.doc_column(42);
        std::cout << make_man_page(cli, "db_builder", fmt);
        exit(EXIT_FAILURE);
    }

    return env;
}


void fill_fluid_opt(environment env, tmpdb::FluidOptions &fluid_opt)
{
    fluid_opt.size_ratio = env.T;
    fluid_opt.largest_level_run_max = env.Z;
    fluid_opt.lower_level_run_max = env.K;
    fluid_opt.buffer_size = env.B;
    fluid_opt.entry_size = env.E;
    fluid_opt.bits_per_element = env.bits_per_element;
    fluid_opt.bulk_load_opt = env.bulk_load_mode;
    if (fluid_opt.bulk_load_opt == tmpdb::bulk_load_type::ENTRIES)
    {
        fluid_opt.num_entries = env.N;
        fluid_opt.levels = tmpdb::FluidLSMCompactor::estimate_levels(
            env.N, env.T, env.E, env.B);
    }
    else
    {
        fluid_opt.levels = env.L;
        fluid_opt.num_entries = tmpdb::FluidLSMCompactor::calculate_full_tree(
            env.T, env.E, env.B, env.L);
    }
    fluid_opt.file_size_policy_opt = env.file_size_policy_opt;
    fluid_opt.fixed_file_size = env.fixed_file_size;
}


void write_existing_keys(environment & env, FluidLSMBulkLoader * fluid_compactor)
{
    std::ofstream key_file;
    key_file.open(env.db_path + "/existing_keys.data");
    
    spdlog::info("Writing out {} existing keys", fluid_compactor->keys.size());
    for (auto key : fluid_compactor->keys)
    {
        key_file << key << std::endl;
    }

    key_file.close();
}


void build_db(environment & env)
{
    spdlog::info("Building DB: {}", env.db_path);
    rocksdb::Options rocksdb_opt;
    tmpdb::FluidOptions fluid_opt;

    rocksdb_opt.create_if_missing = true;
    rocksdb_opt.error_if_exists = true;
    rocksdb_opt.compaction_style = rocksdb::kCompactionStyleNone;
    rocksdb_opt.compression = rocksdb::kNoCompression;
    rocksdb_opt.level0_file_num_compaction_trigger = -1; // Bulk loading so we manually trigger compactions when need be
    rocksdb_opt.IncreaseParallelism(env.parallelism);

    rocksdb_opt.disable_auto_compactions = true;
    rocksdb_opt.write_buffer_size = env.B; 
    rocksdb_opt.num_levels = env.max_rocksdb_levels;
   // Prevents rocksdb from limiting file size
    rocksdb_opt.target_file_size_base = UINT64_MAX;

    fill_fluid_opt(env, fluid_opt);
    DataGenerator *gen;
    if (env.use_key_file)
    {
        gen = new KeyFileGenerator(env.key_file, 2 * env.N, env.seed) ;
    }
    else
    {
        gen = new RandomGenerator(env.seed);
    }
    FluidLSMBulkLoader *fluid_compactor = new FluidLSMBulkLoader(*gen, fluid_opt, rocksdb_opt, env.early_fill_stop);
    rocksdb_opt.listeners.emplace_back(fluid_compactor);

    rocksdb::BlockBasedTableOptions table_options;
    if (env.L > 0)
    {
        table_options.filter_policy.reset(
            rocksdb::NewMonkeyFilterPolicy(
                env.bits_per_element,
                (int) env.T,
                env.L + 1));
    }
    else
    {
        table_options.filter_policy.reset(
            rocksdb::NewMonkeyFilterPolicy(
                env.bits_per_element,
                (int) env.T,
                FluidLSMBulkLoader::estimate_levels(env.N, env.T, env.E, env.B) + 1));
    }
    table_options.no_block_cache = true;
    rocksdb_opt.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_options));

    rocksdb::DB *db = nullptr;
    rocksdb::Status status = rocksdb::DB::Open(rocksdb_opt, env.db_path, &db);
    if (!status.ok())
    {
        spdlog::error("Problems opening DB");
        spdlog::error("{}", status.ToString());
        delete db;
        exit(EXIT_FAILURE);
    }

    if (env.bulk_load_mode == tmpdb::bulk_load_type::LEVELS)
    {
        status = fluid_compactor->bulk_load_levels(db, env.L);
    }
    else
    {
        status = fluid_compactor->bulk_load_entries(db, env.N);
    }

    if (!status.ok())
    {
        spdlog::error("Problems bulk loading: {}", status.ToString());
        delete db;
        exit(EXIT_FAILURE);
    }

    spdlog::info("Waiting for all compactions to finish before closing");
    // Wait for all compactions to finish before flushing and closing DB
    while(fluid_compactor->compactions_left_count > 0);

    if (spdlog::get_level() <= spdlog::level::debug)
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
            spdlog::debug("Level {} : {}", level_idx, level_str);
            level_idx++;
        }
    }

    write_existing_keys(env, fluid_compactor);
    fluid_opt.write_config(env.db_path + "/fluid_config.json");

    db->Close();
    delete db;
}


int main(int argc, char * argv[])
{
    spdlog::set_pattern("[%T.%e]%^[%l]%$ %v");
    environment env = parse_args(argc, argv);

    spdlog::info("Welcome to db_builder!");
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
        spdlog::info("Log level: INFO");
        spdlog::set_level(spdlog::level::info);
    }

    if (env.destroy_db)
    {
        spdlog::info("Destroying DB: {}", env.db_path);
        rocksdb::DestroyDB(env.db_path, rocksdb::Options());
    }

    build_db(env);

    // if (env.use_key_file)
    // {
    //     spdlog::info("Copying key file to DB path");
    //     std::string cmd = std::string("cp '") + env.key_file + "' '" + env.db_path + "/keys.data'";
    //     system(cmd.c_str());
    // }
    return EXIT_SUCCESS;
}
