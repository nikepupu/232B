#ifndef SAOT_UTIL_FILE_UTIL_HPP_
#define SAOT_UTIL_FILE_UTIL_HPP_

#include <string>
#include <vector>

/**
 * \author Zilong Zheng
*/
namespace AOG_LIB {
namespace SAOT {
namespace UTIL {

void CreateDir(const std::string &dir_name);
void GetFileList(std::vector<std::string> &files, const std::string &dir_name,
                 const std::string &ext, bool fullpath);

} // namespace UTIL
} // namespace SAOT
} // namespace AOG_LIB

#endif // SAOT_UTIL_FILE_UTIL_HPP_