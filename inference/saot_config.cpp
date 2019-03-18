#include "util/meta_type.hpp"
#include "saot_config.hpp"
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "util/file_util.hpp"
#include "util/mat_util.hpp"
#include <fstream>
#include <unistd.h>
#define GetCurrentDir getcwd

namespace AOG_LIB {
namespace SAOT {
namespace po = boost::program_options;

std::string GetCurrentWorkingDir( void ) {
  char buff[FILENAME_MAX];
  GetCurrentDir( buff, FILENAME_MAX );
  std::string current_working_dir(buff);
  return current_working_dir;
}

void LoadMatCell2(std::string filename, MatCell_2<cv::Mat> & var)
{
  std::ifstream ifs;
  int sub_size1,sub_size2;
  int sz1, sz2,total;

  ifs.open(filename.c_str(),std::ifstream::in);
  assert(ifs.is_open()==true);
  ifs>>sz1 >> sz2;

  //MatCell_2<cv::Mat> temp=CreateMatCell2Dim<cv::Mat>(sz1,sz2);
 
  var.resize(boost::extents[sz1][sz2]);
  for(int i = 0; i < sz1; i++)
    for(int j = 0; j < sz2; j++)
    {
      ifs >> sub_size1 >> sub_size2;
      if(filename == "./inference/allFilter.txt")
        var[i][j].create(sub_size1, sub_size2, CV_64FC2);
      else
        var[i][j].create(sub_size1, sub_size2, CV_64F);

      double temp, temp2;
      for(int m = 0; m < sub_size1; m++)
        for(int n = 0; n < sub_size2; n++)
        {

          std::complex<double> c;
          ifs >> c;
          temp =  c.real();
          temp2 = c.imag();

          if(filename == "./inference/allFilter.txt")
          {
             std::cout << c <<std::endl;
            var[i][j].at<cv::Vec2d>(m,n)[0] = temp;
            var[i][j].at<cv::Vec2d>(m,n)[1] = temp2;
          }
          else
          {
            var[i][j].at<double>(m,n)  = temp;
          }
        }
    }

  ifs.close();

}

void LoadMatCell1(std::string filename, MatCell_1<cv::Mat> & var)
{
  std::ifstream ifs;
  int sub_size1,sub_size2;
  int sz1, sz2,total;

  ifs.open(filename.c_str(),std::ifstream::in);
  assert(ifs.is_open()==true);
  ifs>>sz1 >> sz2;

  assert(sz1 == 1 || sz2 == 1);
  if(sz2 == 1)
  var.resize(boost::extents[sz1]);
  else if(sz1 == 1)
  var.resize(boost::extents[sz2]);

  for(int i = 0; i < sz1; i++)
    {
      ifs >> sub_size1 >> sub_size2;
      if(filename == "./inference/allFilter.txt")
        var[i].create(sub_size1, sub_size2, CV_64FC2);
      else
        var[i].create(sub_size1, sub_size2, CV_64F);

      double temp, temp2;
      for(int m = 0; m < sub_size1; m++)
        for(int n = 0; n < sub_size2; n++)
        {
          std::complex<double> c;
          ifs >> c;
          temp =  c.real();
          temp2 = c.imag();

          if(filename == "./inference/allFilter.txt")
          {
            std::cout << c <<std::endl;
            var[i].at<cv::Vec2d>(m,n)[0] = temp;
            var[i].at<cv::Vec2d>(m,n)[1] = temp2;
          }
          else
          {
            var[i].at<double>(m,n)  = temp;
          }

        }
    }

  ifs.close();

}


void LoadMat(std::string filename , cv::Mat & var)
{
  std::ifstream ifs;
  int sz1, sz2,total;
  ifs.open(filename.c_str(),std::ifstream::in);
  assert(ifs.is_open()==true);
  ifs>>sz1 >> sz2;
  total = sz1*sz2;
  assert(total >= 0);

  var.create(sz1, sz2, CV_64F);
  for(int i = 0; i < sz1; i++)
    for(int j = 0; j < sz2; j++)
        ifs >> var.at<double>(i,j);

  ifs.close();
}

bool LoadConfigFile(const std::string &filename, SAOTConfig &config) {
  std::cout<<"start Load Config Files"<<std::endl;
  std::cout << GetCurrentWorkingDir() << std::endl;
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

  // [Deprecated] used template.part_loc_x0 and template.part_loc_y0 instead
  std::vector<int>partloc_x0 = std::vector<int>();
  std::vector<int>partloc_y0 = std::vector<int>();
  UTIL::MatlabColonExpression(0, config.part_size[0], config.template_size[0] - config.part_size[0] + 1, partloc_x0);
  UTIL::MatlabColonExpression(0, config.part_size[0], config.template_size[0] - config.part_size[0] + 1, partloc_x0);
  config.numCandPart = partloc_x0.size() * partloc_y0.size();

  config.part_loc_x = std::vector<int>();
  config.part_loc_y = std::vector<int>();
  for (int x : partloc_x0) {
    for (int y : partloc_y0) {
      config.part_loc_x.push_back(x);
      config.part_loc_y.push_back(y);
    }
  }

  int _part_rotate_start = static_cast<int>(fs["part_rotate"]["start"]);
  int _part_rotate_end   = static_cast<int>(fs["part_rotate"]["end"]);
  int _part_rotate_scale = static_cast<int>(fs["part_rotate"]["scale"]);

  config.num_part_rotation = _part_rotate_end - _part_rotate_start + 1;
  config.part_rotation_range.resize(config.num_part_rotation);
  for (int i = 0; i < config.num_part_rotation; i++)
    config.part_rotation_range[i] = (i + _part_rotate_start) * _part_rotate_scale;
    
  config.location_shift_limit = static_cast<int>(fs["location_shift_limit"]);
  config.orient_shift_limit = static_cast<int>(fs["orient_shift_limit"]);
  
  config.max_part_relative_rotation = 2;
  config.resolution_shift_limit = 1;
  config.min_rotation_dif = pow(sin(config.max_part_relative_rotation * PI / config.num_orient) - sin(0), 2) + pow(cos(config.max_part_relative_rotation  * PI / config.num_orient) - cos(0), 2) + 1e-10;

  int _rotate_start = static_cast<int>(fs["obj_rotate"]["start"]);
  int _rotate_end   = static_cast<int>(fs["obj_rotate"]["end"]);
  int _rotate_scale = static_cast<int>(fs["obj_rotate"]["scale"]);
  config.num_rotate = _rotate_end - _rotate_start + 1;
  config.rotation_range.resize(config.num_rotate);
  
  for (int i = 0; i < config.num_part_rotation; i++)
    config.rotation_range[i] = (i + _rotate_start) * _rotate_scale;


  config.startx = 1; config.starty = 1;
  config.endx = config.startx + config.template_size[0] - 1;
  config.endy = config.starty + config.template_size[1] - 1;
  /////////////////////////////////////////////////////////
  // reading from the thrid mat file --- object model
  std::ifstream ifs;
  ifs.open("./inference/PartOnOff.txt",std::ifstream::in);
  assert(ifs.is_open()==true);
  int sz1, sz2, total;
  ifs>>sz1 >> sz2;
  total = sz1*sz2;
  assert(total >= 0);
  config.PartOnOff = new int[total];
  for(int i = 0; i < total; i++)
    ifs >> config.PartOnOff[i];
  ifs.close();

  ///////////////////////////////////////////////////////
  ///// Pay Attention
  ///// this is hard coded to all one, since we want to
  ///// detect all object parts. Turn some of them off 
  ///// if we don't need some parts.
  //////////////////////////////////////////////////////
  config.selectedPart.create(total,1, CV_64F);

  for(int i = 0; i < total; i++ )
    config.selectedPart.at<double>(i,0)  = i+1;

  
  LoadMat("./inference/allS3SelectedRow.txt", config.allS3SelectedRow);
  LoadMat("./inference/allS3SelectedCol.txt", config.allS3SelectedCol);
  LoadMat("./inference/allS3SelectedOri.txt", config.allS3SelectedOri);

  ////////////////////////////////////////////
  ////read from the second mat file part model
  ////////////////////////////////////////////
  /// the following are in MatCell_2<cv::Mat>  format

  LoadMatCell2("./inference/allSelectedx.txt", config.allSelectedx);
  LoadMatCell2("./inference/allSelectedy.txt", config.allSelectedy);
  LoadMatCell2("./inference/allSelectedOrient.txt", config.allSelectedOrient);
  LoadMatCell1("./inference/selectedlambda.txt", config.selectedlambda);
  LoadMatCell1("./inference/selectedLogZ.txt", config.selectedLogZ);
  LoadMatCell1("./inference/allSymbol.txt", config.allSymbol);
  LoadMatCell1("./inference/allFilter.txt", config.allFilter);

  LoadMatCell2("./inference/largerAllSelectedx.txt", config.largerAllSelectedx);
  LoadMatCell2("./inference/largerAllSelectedy.txt", config.largerAllSelectedy);
  LoadMatCell2("./inference/largerAllSelectedOrient.txt", config.largerAllSelectedOrient);
  LoadMatCell1("./inference/largerSelectedlambda.txt", config.largerSelectedlambda);
  LoadMatCell1("./inference/largerSelectedLogZ.txt", config.largerSelectedLogZ);





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
