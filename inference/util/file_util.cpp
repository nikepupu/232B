#include "file_util.hpp"

#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

namespace AOG_LIB {
namespace SAOT {
namespace UTIL {

namespace fs = boost::filesystem;

void CreateDir(const std::string &dir_name) {
  fs::path dir(dir_name.c_str());
  if (!fs::is_directory(dir)) {
    fs::path parent_path = dir.parent_path();
    fs::create_directories(parent_path);
  }
}

void GetFileList(std::vector<std::string> &files, const std::string &dir_name,
                 const std::string &ext, bool fullpath) {
  fs::path target_dir(dir_name.c_str());
  fs::directory_iterator it(target_dir), eod;

  bool get_all_files = (ext.compare("*") == 0);

  BOOST_FOREACH(const fs::path &p, std::make_pair(it, eod)) {
    if (fs::is_regular_file(p)) {
      if (get_all_files) {
        const std::string &file_path =
            fullpath ? p.string() : p.filename().string();
        files.push_back(file_path);
      } else {
        if (ext.compare(p.extension().string()) == 0) {
          const std::string &file_path =
              fullpath ? p.string() : p.filename().string();
          files.push_back(file_path);
        }
      }
    }
  }
}

}  // namespace UTIL
}  // namespace SAOT
}  // namespace AOG_LIB
