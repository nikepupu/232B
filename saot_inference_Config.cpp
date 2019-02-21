#include "saot_inference_Config.hpp"
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>


namespace AOG_LIB {
namespace SAOT {
namespace po = boost::program_options;

bool LoadSAOTInferenceConfigFile(const std::string &filename1,  const std::string &filename2, const std::string &filename3, SAOTInferenceConfig &config) 
{
  cv::FileStorage fs1;
  cv::FileStorage fs2;
  cv::FileStorage fs3;
  fs1.open(filename1, cv::FileStorage::READ);
  if (!fs1.isOpened()) {
    BOOST_LOG_TRIVIAL(error) << boost::format("File %s not exist") % filename1;
    return false;
  }

  // fs2.open(filename2, cv::FileStorage::READ);
  // if (!fs2.isOpened()) {
  //   BOOST_LOG_TRIVIAL(error) << boost::format("File %s not exist") % filename2;
  //   return false;
  // }

  // fs3.open(filename3, cv::FileStorage::READ);
  // if (!fs3.isOpened()) {
  //   BOOST_LOG_TRIVIAL(error) << boost::format("File %s not exist") % filename3;
  //   return false;
  // }

  
  //config.subsampleS2 = static_cast<int>(fs1["subsampleS2"]);
  //config.numOrient = static_cast<int>(fs1["numOrient"]);


  
  return true;
}





}  // namespace SAOT_inference
}  // namespace AOG_LIB
