#include "bulk_loader.hpp"


rocksdb::Status FluidLSMBulkLoader::bulk_load_entries(rocksdb::DB *db, size_t num_entries)
{
    spdlog::info("Bulk loading DB with {} entries", num_entries);
    rocksdb::Status status;

    size_t E = this->fluid_opt.entry_size;
    size_t B = this->fluid_opt.buffer_size;
    double T = this->fluid_opt.size_ratio;
    size_t estimated_levels = tmpdb::FluidLSMCompactor::estimate_levels(num_entries, T, E, B);
    spdlog::debug("Estimated levels: {}", estimated_levels);

    size_t entries_in_buffer = (B / E);
    spdlog::debug("Number of entries that can fit in the buffer: {}", entries_in_buffer);

    std::vector<size_t> capacity_per_level(estimated_levels);
    capacity_per_level[0] = (entries_in_buffer) * (T - 1);
    for (size_t level_idx = 1; level_idx < estimated_levels; level_idx++)
    {
        capacity_per_level[level_idx] = capacity_per_level[level_idx - 1] * T;
    }

    size_t full_num_entries = tmpdb::FluidLSMCompactor::calculate_full_tree(T,
        E, B, estimated_levels);
    
    double percent_full = (double) num_entries / full_num_entries;
    spdlog::debug("Percentage full : {}", percent_full);
    for (size_t level_idx = 0; level_idx < estimated_levels; level_idx++)
    {
        capacity_per_level[level_idx] = capacity_per_level[level_idx] * percent_full;
    }

    if (spdlog::get_level() <= spdlog::level::debug)
    {
        std::string capacity_str = "";
        for (auto &capacity : capacity_per_level)
        {
            capacity_str += std::to_string(capacity) + ", ";
        }
        capacity_str = capacity_str.substr(0, capacity_str.size() - 2);
        spdlog::debug("Entries per level : [{}]", capacity_str);
    }

    status = this->bulk_load(db, capacity_per_level, estimated_levels, num_entries);

    return status;
}


rocksdb::Status FluidLSMBulkLoader::bulk_load_levels(rocksdb::DB *db, size_t num_levels)
{
    spdlog::info("Bulk loading DB with {} levels", num_levels);
    rocksdb::Status status;

    size_t entries_in_buffer = (this->fluid_opt.buffer_size / this->fluid_opt.entry_size);
    spdlog::debug("Number of entries that can fit in the buffer: {}", entries_in_buffer);

    std::vector<size_t> capacity_per_level(num_levels);
    capacity_per_level[0] = entries_in_buffer * (this->fluid_opt.size_ratio - 1);
    for (size_t level_idx = 1; level_idx < num_levels; level_idx++)
    {
        capacity_per_level[level_idx] = capacity_per_level[level_idx - 1] * this->fluid_opt.size_ratio;
    }

    if (spdlog::get_level() <= spdlog::level::debug)
    {
        std::string capacity_str = "";
        for (auto &capacity : capacity_per_level)
        {
            capacity_str += std::to_string(capacity) + ", ";
        }
        capacity_str = capacity_str.substr(0, capacity_str.size() - 2);
        spdlog::debug("Entries per level : [{}]", capacity_str);
    }

    status = this->bulk_load(db, capacity_per_level, num_levels, INT_MAX);

    return status;
}


rocksdb::Status FluidLSMBulkLoader::bulk_load(
    rocksdb::DB *db,
    std::vector<size_t> capacity_per_level,
    size_t num_levels,
    size_t max_entries)
{
    rocksdb::Status status;
    size_t level_idx;
    size_t num_runs;
    size_t num_entries_loaded = 0;

    // Fill up levels starting from the BOTTOM
    for (size_t level = num_levels; level > 0; level--)
    {
        level_idx = level - 1;
        if (capacity_per_level[level_idx] == 0) { continue; }
        spdlog::debug("Bulk loading level {} with {} entries.", level, capacity_per_level[level_idx]);

        if (level == num_levels) //> Last level has Z max runs
        {
            num_runs = this->fluid_opt.largest_level_run_max;
        }
        else //> Every other level inbetween has K max runs
        {
            num_runs = this->fluid_opt.lower_level_run_max;
        }

        status = this->bulk_load_single_level(db, level_idx, capacity_per_level[level_idx], num_runs);
        num_entries_loaded += capacity_per_level[level_idx];
        if (this->stop_after_level_filled && num_entries_loaded > max_entries)
        {
            spdlog::debug("Already reached max entries, stopping bulk loading.");
            break;
        }
    }

    return status;
}


rocksdb::Status FluidLSMBulkLoader::bulk_load_single_level(
    rocksdb::DB *db,
    size_t level_idx,
    size_t capacity_per_level,
    size_t num_runs)
{
    rocksdb::Status status;
    size_t entries_per_run = capacity_per_level / num_runs;
    size_t level = level_idx + 1;

    for (size_t run_idx = 0; run_idx < num_runs; run_idx++)
    {
        spdlog::trace("Loading RUN {} at LEVEL {} : {} entries (run size ~ {:.3f} MB)",
            run_idx, level, entries_per_run,
            (entries_per_run * this->fluid_opt.entry_size) / static_cast<double>(1 << 20));

        status = this->bulk_load_single_run(db, entries_per_run);
    }

    // Force all runs in this level to be mapped to their respective level
    rocksdb::ColumnFamilyMetaData cf_meta;
    db->GetColumnFamilyMetaData(&cf_meta);

    std::vector<std::string> file_names;
    for (auto & file : cf_meta.levels[0].files)
    {
        if (file.being_compacted) { continue; }
        file_names.push_back(file.name);
    }

    if (fluid_opt.file_size_policy_opt == tmpdb::file_size_policy::INCREASING)
    {
        // We add an extra 5% to size per output file size in order to compensate for meta-data
        this->rocksdb_compact_opt.output_file_size_limit = 1.05 * entries_per_run * this->fluid_opt.entry_size;
        if (level == 1)
        {
            // Note for an increasing file size, we do not want to trigger a compaction at level 1
            return status;
        }
    }
    else if (fluid_opt.file_size_policy_opt == tmpdb::file_size_policy::BUFFER)
    {
        if (level == 1)
        {
            // Note for an increasing file size, we do not want to trigger a compaction at level 1
            return status;
        }
        this->rocksdb_compact_opt.output_file_size_limit = this->fluid_opt.buffer_size;
    }
    else
    {
        this->rocksdb_compact_opt.output_file_size_limit = this->fluid_opt.fixed_file_size;
    }

    tmpdb::CompactionTask *task = new tmpdb::CompactionTask(
        db, this, "default", file_names, level_idx, this->rocksdb_compact_opt, 0, true, false);
    this->ScheduleCompaction(task);

    return status;
}


rocksdb::Status FluidLSMBulkLoader::bulk_load_single_run(rocksdb::DB *db, size_t num_entries)
{
    rocksdb::WriteOptions write_opt;
    write_opt.sync = false;
    write_opt.disableWAL = true;
    write_opt.no_slowdown = false;
    write_opt.low_pri = true; // every insert is less important than compaction

    size_t buffer_size = this->fluid_opt.entry_size * num_entries * 8;
    rocksdb::Status status = db->SetOptions({{"write_buffer_size", std::to_string(buffer_size)}});

    size_t batch_size = std::min((size_t) BATCH_SIZE, num_entries);
    for (size_t entry_num = 0; entry_num < num_entries; entry_num += batch_size)
    {
        rocksdb::WriteBatch batch(0, UINT64_MAX);
        for (int i = 0; i < (int) batch_size; i++)
        {
            std::pair<std::string, std::string> key_value =
                this->data_gen.generate_kv_pair(this->fluid_opt.entry_size);
            batch.Put(key_value.first, key_value.second);
            this->keys.push_back(key_value.first);
        }
        status = db->Write(write_opt, &batch);
        if (!status.ok())
        {
            spdlog::error("{}", status.ToString());
        }
    }

    spdlog::trace("Flushing after writing batch");
    rocksdb::FlushOptions flush_opt;
    flush_opt.wait = true;
    db->Flush(flush_opt);

    return status;
}


void FluidLSMBulkLoader::CompactFiles(void *arg)
{
    std::unique_ptr<tmpdb::CompactionTask> task(reinterpret_cast<tmpdb::CompactionTask *>(arg));
    assert(task);
    assert(task->db);
    // assert(task->output_level > (int) task->origin_level_id);

    std::vector<std::string> *output_file_names = new std::vector<std::string>();
    rocksdb::Status s = task->db->CompactFiles(
        task->compact_options,
        task->input_file_names,
        task->output_level,
        -1,
        output_file_names
    );

    // spdlog::trace("CompactFiles {} -> {}", task->origin_level_id, task->output_level);
    if (!s.ok() && !s.IsIOError() && task->retry_on_fail)
    {
        // If a compaction task with its retry_on_fail=true failed,
        // try to schedule another compaction in case the reason
        // is not an IO error.

        spdlog::warn("CompactFile {} -> {} with {} files did not finish: {}",
            task->origin_level_id + 1,
            task->output_level + 1,
            task->input_file_names.size(),
            s.ToString());
        tmpdb::CompactionTask *new_task = new tmpdb::CompactionTask(
            task->db,
            task->compactor,
            task->column_family_name,
            task->input_file_names,
            task->output_level,
            task->compact_options,
            task->origin_level_id,
            task->retry_on_fail,
            true
        );
        task->compactor->ScheduleCompaction(new_task);

        return;
    }
    ((FluidLSMBulkLoader *) task->compactor)->compactions_left_count--;
    ((FluidLSMBulkLoader *) task->compactor)->compactions_left_mutex.unlock();

    spdlog::trace("CompactFiles level {} -> {} finished with status : {}",
        task->origin_level_id + 1,
        task->output_level + 1,
        s.ToString());

    return;
}


void FluidLSMBulkLoader::ScheduleCompaction(tmpdb::CompactionTask *task)
{
    if (!task->is_a_retry)
    {
        this->compactions_left_mutex.lock();
        this->compactions_left_count++;
    }
    this->rocksdb_opt.env->Schedule(&FluidLSMBulkLoader::CompactFiles, task);


    return;
}