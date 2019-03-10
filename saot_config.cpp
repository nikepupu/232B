#include "saot_config.hpp"
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "util/file_util.hpp"
#include "util/mat_util.hpp"

namespace AOG_LIB {
namespace SAOT {
namespace po = boost::program_options;

bool LoadConfigFile(const std::string &filename, SAOTConfig &config) {
  cv::FileStorage fs;
  fs.open(filename, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    BOOST_LOG_TRIVIAL(error) << boost::format("File %s not exist") % filename;
    return false;
  }

  fs["category"] >> config.category;
  config.num_resolution = static_cast<int>(fs["num_resolution"]);
  config.resize_factor = static_cast<double>(fs["resize_factor"]);
  config.template_size[0] = static_cast<int>(fs["template_size"]["x"]);
  config.template_size[1] = static_cast<int>(fs["template_size"]["y"]);
  
  config.epsilon = static_cast<double>(fs["epsilon"]);

  config.num_orient = static_cast<int>(fs["num_orient"]);
  config.orient_shift_limit = static_cast<int>(fs["orient_shift_limit"]);
  config.num_iteration = static_cast<int>(fs["num_iteration"]);

  config.part_size[0] = config.template_size[0] / 2;
  config.part_size[1] = config.template_size[1] / 2;

  std::vector<int>partloc_x0 = std::vector<int>();
  std::vector<int>partloc_y0 = std::vector<int>();
  UTIL::MatlabColonExpression(0, config.part_size[0], config.template_size[0] - config.part_size[0] + 1, partloc_x0);
  UTIL::MatlabColonExpression(0, config.part_size[0], config.template_size[0] - config.part_size[0] + 1, partloc_x0);
  config.numCandPart = partloc_x0.size() * partloc_y0.size();
  
  return true;
}

bool GetCmdOptions(int argc, char **argv, SAOTConfig &config) {
  std::string cfg_file, img_dir, img_ext, output_dir, bkg_dir;

  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message")(
      "config-file,c",
      po::value<std::string>(&cfg_file)->default_value("config.yml"),
      "config file path")(
      "image-dir,i",
      po::value<std::string>(&img_dir)->default_value("positiveImage"),
      "input image directory")(
      "image-ext,e", po::value<std::string>(&img_ext)->default_value(".jpg"),
      "image extension")(
      "output-dir,o",
      po::value<std::string>(&output_dir)->default_value("output"),
      "output directory")(
      "bkg-image-dir,b",
      po::value<std::string>(&bkg_dir)->default_value("BackgroundImage"),
      "background image directory");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return false;
  }

  if (!vm.count("config-file")) {
    std::cout << "No configuration file provided" << std::endl;
    std::cout << desc << std::endl;
    return false;
  }

  if (!LoadConfigFile(cfg_file, config)) {
    return false;
  }
  config.img_dir = img_dir;
  config.img_ext = img_ext;
  config.output_dir = output_dir;
  config.bkg_img_dir = bkg_dir;

  config.image_name = std::vector<std::string>();
  AOG_LIB::SAOT::UTIL::GetFileList(config.image_name, config.img_dir, config.img_ext, true);
  config.num_image = config.image_name.size();

  return true;
}

}  // namespace SAOT
}  // namespace AOG_LIB
