#include "tmpdb/fluid_lsm_compactor.hpp"

using namespace tmpdb;


FluidCompactor::FluidCompactor(const FluidOptions fluid_opt, const rocksdb::Options rocksdb_opt)
    : fluid_opt(fluid_opt), rocksdb_opt(rocksdb_opt), rocksdb_compact_opt()
{
    this->rocksdb_compact_opt.compression = this->rocksdb_opt.compression;
    this->rocksdb_compact_opt.output_file_size_limit = this->rocksdb_opt.target_file_size_base;
}


int FluidLSMCompactor::largest_occupied_level(rocksdb::DB *db) const
{
    rocksdb::ColumnFamilyMetaData cf_meta;
    db->GetColumnFamilyMetaData(&cf_meta);
    int largest_level_idx = 0;

    for (size_t level_idx = cf_meta.levels.size() - 1; level_idx > 0; level_idx--)
    {
        if (cf_meta.levels[level_idx].files.empty()) {continue;}
        largest_level_idx = level_idx;
        break;
    }
    
    if (largest_level_idx == 0)
    {
        if (cf_meta.levels[0].files.empty())
        {
            spdlog::error("Database is empty, exiting");
            exit(EXIT_FAILURE);
        }
    }

    return largest_level_idx;
}


CompactionTask *FluidLSMCompactor::PickCompaction(rocksdb::DB *db, const std::string &cf_name, const size_t level_idx)
{
    this->meta_data_mutex.lock();
    int live_runs;
    int T = this->fluid_opt.size_ratio;
    int largest_level_idx = this->largest_occupied_level(db);

    rocksdb::ColumnFamilyMetaData cf_meta;
    db->GetColumnFamilyMetaData(&cf_meta);

    std::vector<std::string> input_file_names;

    size_t level_size = 0;
    for (auto &file : cf_meta.levels[level_idx].files)
    {
        if (file.being_compacted) {continue;}
        input_file_names.push_back(file.name);
        level_size += file.size;
    }
    live_runs = input_file_names.size();

    if (fluid_opt.file_size_policy_opt == INCREASING)
    {
        bool lower_levels_need_compact = (((int) level_idx < largest_level_idx) && (live_runs > this->fluid_opt.lower_level_run_max));
        bool last_levels_need_compact = (((int) level_idx == largest_level_idx) && (live_runs > this->fluid_opt.largest_level_run_max));

        if (!lower_levels_need_compact && !last_levels_need_compact)
        {
            this->meta_data_mutex.unlock();
            return nullptr;
        }
    }
    else
    {
        uint64_t level_capacity = pow(T, level_idx) * (T - 1) * this->fluid_opt.buffer_size;
        spdlog::info("Level Capacity at level {} : {} MB", level_idx, level_capacity >> 20);
        bool level_need_compaction = (level_size > (pow(T, level_idx) * (T - 1) * this->fluid_opt.buffer_size));
        if (!level_need_compaction)
        {
            this->meta_data_mutex.unlock();
            return nullptr;
        }
    }

    if (fluid_opt.file_size_policy_opt == INCREASING)
    {

        size_t level_capacity = (T - 1) * std::pow(T, level_idx + 1) * (this->fluid_opt.buffer_size);
        if ((int) level_idx == this->largest_occupied_level(db)) //> Last level we restrict number of runs to Z
        {
            this->rocksdb_compact_opt.output_file_size_limit = static_cast<uint64_t>(level_capacity) / this->fluid_opt.largest_level_run_max;
        }
        else
        {
            this->rocksdb_compact_opt.output_file_size_limit = static_cast<uint64_t>(level_capacity) / this->fluid_opt.lower_level_run_max;
        }

        // We give an extra 5% memory per file in order to accomodate meta data
        this->rocksdb_compact_opt.output_file_size_limit *= 1.05;
    }
    else if (fluid_opt.file_size_policy_opt == BUFFER)
    {
        this->rocksdb_compact_opt.output_file_size_limit = rocksdb_opt.write_buffer_size;
    }
    else
    {
        this->rocksdb_compact_opt.output_file_size_limit = fluid_opt.fixed_file_size;
    }

    this->meta_data_mutex.unlock();
    spdlog::trace("Created CompactionTask L{} -> L{}", level_idx + 1, level_idx + 2);
    return new CompactionTask(
        db, this, cf_name, input_file_names, level_idx + 1, this->rocksdb_compact_opt, level_idx, false, false);
}


void FluidLSMCompactor::OnFlushCompleted(rocksdb::DB *db, const ROCKSDB_NAMESPACE::FlushJobInfo &info)
{
    int largest_level_idx = this->largest_occupied_level(db);

    for (int level_idx = largest_level_idx; level_idx > -1; level_idx--)
    {
        CompactionTask *task = PickCompaction(db, info.cf_name, level_idx);

        if (!task) {continue;}

        task->retry_on_fail = info.triggered_writes_slowdown;
        ScheduleCompaction(task);
    }

    return;
}


void FluidLSMCompactor::CompactFiles(void *arg)
{
    std::unique_ptr<CompactionTask> task(reinterpret_cast<CompactionTask *>(arg));
    assert(task);
    assert(task->db);
    assert(task->output_level > (int) task->origin_level_id);

    std::vector<std::string> *output_file_names = new std::vector<std::string>();
    rocksdb::Status s = task->db->CompactFiles(
        task->compact_options,
        task->input_file_names,
        task->output_level,
        -1,
        output_file_names
    );

    if (!s.ok() && !s.IsIOError() && task->retry_on_fail && !s.IsInvalidArgument())
    {
        // If a compaction task with its retry_on_fail=true failed,
        // try to schedule another compaction in case the reason
        // is not an IO error.

        spdlog::warn("CompactFile L{} -> L{} with {} files did not finish: {}",
            task->origin_level_id + 1,
            task->output_level + 1,
            task->input_file_names.size(),
            s.ToString());
        CompactionTask *new_task = task->compactor->PickCompaction(
            task->db,
            task->column_family_name,
            task->origin_level_id
        );
        new_task->is_a_retry = true;
        task->compactor->ScheduleCompaction(new_task);

        return;
    }

    spdlog::trace("CompactFiles L{} -> L{} finished | Status: {}",
                  task->origin_level_id + 1, task->output_level + 1, s.ToString());
    ((FluidLSMCompactor *) task->compactor)->compactions_left_count--;

    return;
}


void FluidLSMCompactor::ScheduleCompaction(CompactionTask *task)
{
    if (!task->is_a_retry)
    {
        this->compactions_left_count++;
    }
    this->rocksdb_opt.env->Schedule(&FluidLSMCompactor::CompactFiles, task);

    return;
}


size_t FluidLSMCompactor::estimate_levels(size_t N, double T, size_t E, size_t B)
{
    if ((N * E) < B)
    {
        spdlog::warn("Number of entries (N = {}) fits in the in-memory buffer, defaulting to 1 level", N);
        return 1;
    }

    size_t estimated_levels = std::ceil(std::log((N * E / B) + 1) / std::log(T));

    return estimated_levels;
}


bool FluidLSMCompactor::requires_compaction(rocksdb::DB *db)
{
    this->meta_data_mutex.lock();
    int largest_level_idx = this->largest_occupied_level(db);
    this->meta_data_mutex.unlock();
    bool task_scheduled = false;

    for (int level_idx = largest_level_idx; level_idx > -1; level_idx--)
    {
        CompactionTask *task = PickCompaction(db, "default", level_idx);

        if (!task) {continue;}

        ScheduleCompaction(task);
        task_scheduled = true;
    }

    return task_scheduled;
}


size_t FluidLSMCompactor::calculate_full_tree(double T, size_t E, size_t B, size_t L)
{
    int full_tree_size = 0;
    size_t entries_in_buffer = B / E;

    for(size_t level = 1; level < L + 1; level++)
    {
        full_tree_size += entries_in_buffer * (T - 1) * (std::pow(T, level - 1));
    }

    return full_tree_size;
}