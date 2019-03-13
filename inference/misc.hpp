#ifndef SAOT_MISC_HPP_
#define SAOT_MISC_HPP_
#include "./util/meta_type.hpp"
#include <string>
#include <vector>
#include "template.hpp"
#include "saot_config.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "exponential_model.hpp"

/**
 * \author Xiaofeng Gao
*/
namespace AOG_LIB {
namespace SAOT {
    
    struct CropInstanceReturnType{};

    void DrawElement(cv::Mat &Template, int mo, int mx, int my, double w, int sizex, 
    int sizey, int halfFilterSize, MatCell_1<cv::Mat> &allSymbol);

    double LocalMaximumPooling(int img, int orient, int x, int y, int *trace, 
    int numShift, cv::Mat &xShift, cv::Mat &yShift, cv::Mat &orientShifted, 
    MatCell_1<cv::Mat> &SUM1map, int sizex, int sizey, int numImage);

    void NonMaximumSuppression(int img, int mo, int mx, int my, const SAOTConfig& config, 
    MatCell_1<cv::Mat> &SUM1map, cv::Mat &MAX1map, cv::Mat &trackMap, MatCell_1<cv::Mat> &Correlation, 
    cv::Mat &pooledMax1map, cv::Mat &orientShifted, cv::Mat &xShift, cv::Mat &yShift, 
    int numShift, const cv::Size &img_size);

    void Sigmoid(const SAOTConfig& config, MatCell_2<cv::Mat> &map_sum1_find);

    void SharedSketch(const SAOTConfig& config, const cv::Size &img_size, MatCell_1<cv::Mat> &SUM1map, 
    MatCell_1<cv::Mat> &Correlation, MatCell_1<cv::Mat> &allSymbol, cv::Mat &commonTemplate, 
    MatCell_1<cv::Mat> &deformedTemplate, MatCell_1<ExpParam> &eParam, MatCell_1<BasisParam> &bParam);

    void LocalNormalize(const cv::Size &img_size, int num_filters,
                        int half_filter_size, const cv::Size &local_half,
                        double threshold_factor,
                        MatCell_1<cv::Mat> &filter_responses);

    void CropInstance(const SAOTConfig& config, const MatCell_1<cv::Mat> &src, MatCell_1<cv::Mat> &dest, 
        double rshift, double cshift, double rotation, double scaling, double reflection, 
        const double* transformedRow, const double* transformedCol, int nOri, int nScale, int destHeight, int destWidth);

    void CropInstance(const SAOTConfig& config, const cv::Mat &src, cv::Mat &dest, 
        double rshift, double cshift, double rotation, double scaling, double reflection, 
        const std::vector<double> transformedRow, const std::vector<double> transformedCol, int nOri, int nScale, int destHeight, int destWidth);

    void Instribe(cv::Mat &srcImage, cv::Mat &destImage, int val);

    void Histogram(MatCell_2<cv::Mat> &filteredImage, const SAOTConfig& config, const cv::Size &img_size, 
    double binSize, int numBin, cv::Mat& histog);

}  // namespace SAOT
}  // namespace AOG_LIB

#endif  // SAOT_MISC_HPP_
