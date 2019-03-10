#ifndef SAOT_MISC_HPP_
#define SAOT_MISC_HPP_

#include "saot_inference_Config.hpp"
#include "./util/meta_type.hpp"
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

/**
 * \author Xiaofeng Gao
*/
namespace AOG_LIB {
namespace SAOT {
    
    struct CropInstanceReturnType{};

    struct expParam {
        double storedlambda, storedExpectation, storedlogZ;
    };

    struct basisParam {
        double selectedOrient, selectedx, selectedy, selectedlambda, selectedLogZ;
    };

    void DrawElement(cv::Mat &Template, int mo, int mx, int my, double w, int sizex, 
    int sizey, int halfFilterSize, MatCell_1<cv::Mat> &allSymbol);

    double LocalMaximumPooling(int img, int orient, int x, int y, int *trace, 
    int numShift, cv::Mat &xShift, cv::Mat &yShift, cv::Mat &orientShifted, 
    MatCell_1<cv::Mat> &SUM1map, int sizex, int sizey, int numImage);

    void NonMaximumSuppression(int img, int mo, int mx, int my, const SAOTInferenceConfig& config, 
    MatCell_1<cv::Mat> &SUM1map, cv::Mat &MAX1map, cv::Mat &trackMap, MatCell_1<cv::Mat> &Correlation, 
    cv::Mat &pooledMax1map, cv::Mat &orientShifted, cv::Mat &xShift, cv::Mat &yShift, 
    int numShift, const cv::Size &img_size);

    void Sigmoid(const SAOTInferenceConfig& config, MatCell_2<cv::Mat> &map_sum1_find);

    void SharedSketch(const SAOTInferenceConfig& config, const cv::Size &img_size, MatCell_1<cv::Mat> &SUM1map, 
    MatCell_1<cv::Mat> &Correlation, MatCell_1<cv::Mat> &allSymbol, cv::Mat &commonTemplate, 
    MatCell_1<cv::Mat> &deformedTemplate, MatCell_1<expParam> &eParam, MatCell_1<basisParam> &bParam);

    void LocalNormalize(const cv::Size &img_size, int num_filters,
                        int half_filter_size, const cv::Size &local_half,
                        double threshold_factor,
                        MatCell_1<cv::Mat> &filter_responses);

    void CropInstance(const SAOTInferenceConfig& config, MatCell_1<cv::Mat> &src, MatCell_1<cv::Mat> &dest, 
        double rshift, double cshift, double rotation, double scaling, double reflection, 
        const double* transformedRow, const double* transformedCol, int destHeight, int destWidth);

    void Instribe(cv::Mat &srcImage, cv::Mat &destImage, int val);

    void Histogram(MatCell_2<cv::Mat> &filteredImage, const SAOTInferenceConfig& config, const cv::Size &img_size, 
    double binSize, int numBin, cv::Mat& histog);

}  // namespace SAOT
}  // namespace AOG_LIB

#endif  // SAOT_MISC_HPP_
