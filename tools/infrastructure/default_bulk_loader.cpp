#include "default_bulk_loader.hpp"


rocksdb::Status DefaultBulkLoader::default_bulk_loader(rocksdb::DB *db, size_t num_entries, std::string db_path)
{
    rocksdb::WriteOptions write_opt;
    write_opt.sync = false;
    write_opt.disableWAL = false;
    write_opt.no_slowdown = false;
    write_opt.low_pri = false;

//    size_t buffer_size = 1 << 20;
//    rocksdb::Status status = db->SetOptions({{"write_buffer_size", std::to_string(buffer_size)}});
//
//    size_t batch_size = std::min((size_t) BATCH_SIZE, num_entries);
//    for (size_t entry_num = 0; entry_num < num_entries; entry_num += batch_size)
//    {
//        rocksdb::WriteBatch batch(0, UINT64_MAX);
//        for (int i = 0; i < (int) batch_size; i++)
//        {
//            std::pair<std::string, std::string> key_value =
//                this->data_gen.gen_kv_pair(this->fluid_opt.entry_size);
//            batch.Put(key_value.first, key_value.second);
//            this->keys.push_back(key_value.first);
//        }
//        status = db->Write(write_opt, &batch);
//        if (!status.ok())
//        {
//            spdlog::error("{}", status.ToString());
//        }
//    }

    rocksdb::Options options;

    rocksdb::SstFileWriter sst_file_writer(rocksdb::EnvOptions(), options);
    // Path to where we will write the SST file
    std::string file_path = db_path + "/existing_keys";
    size_t E = 1 << 10;

    // Open the file for writing
    rocksdb::Status s = sst_file_writer.Open(file_path);
    if (!s.ok()) {
        printf("Error while opening file %s, Error: %s\n", file_path.c_str(),
               s.ToString().c_str());
        return s;
    }

    // Insert rows into the SST file, note that inserted keys must be
    // strictly increasing (based on options.comparator)
    for (size_t entry_num = 0; entry_num < num_entries; entry_num++){
      std::pair<std::string, std::string> key_value =
                      this->data_gen.gen_kv_pair(1<<10);
                  rocksdb::Status status = sst_file_writer.Add(key_value.first, key_value.second);
      if (!status.ok()) {
        printf("Error while adding Key: %s, Error: %s\n", key_value.first.c_str(),
               s.ToString().c_str());
        return s;
      }
    }

    // Close the file
    rocksdb::Status status = db->IngestExternalFile({file_path + ".sst"}, rocksdb::IngestExternalFileOptions());
    status = sst_file_writer.Finish();
    if (!status.ok()) {
        printf("Error while finishing file %s, Error: %s\n", file_path.c_str(),
               s.ToString().c_str());
        return status;
    }

    spdlog::trace("Flushing after writing batch");
    rocksdb::FlushOptions flush_opt;
    flush_opt.wait = true;
    db->Flush(flush_opt);

    return status;
}