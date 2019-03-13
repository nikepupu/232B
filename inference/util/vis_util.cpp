#include "vis_util.hpp"
#include "meta_type.hpp"

namespace AOG_LIB {
namespace SAOT {
namespace UTIL {

void DisplayImages(const MatCell_1<cv::Mat>& images, cv::Mat & out_image, int ncol, int bx, int by, bool normalize) {

    int n = images.shape()[0];
    int nrow = ceil(n / ncol);
    int widthMargin = 5;
    int width = by * ncol + (ncol - 1) * widthMargin;
    int heightMargin = 5;
    int height = bx * nrow + (nrow - 1) * heightMargin;
    out_image = cv::Mat(height, width, CV_64F, cv::Scalar(255));

    for (int i = 0; i < n; i++) {
        int row = ceil(i / ncol);
        int col = i - (row - 1) * ncol;
        int startx = (row - 1) * (bx + heightMargin);
        int starty = (col - 1) * (by + widthMargin);
        cv::Size dsize = cv::Size(bx, by);
        cv::Mat towrite = cv::Mat(bx, by, CV_64F, cv::Scalar(0));
        cv::resize(images[i], towrite, dsize, 0, 0);
        if (normalize) {
            double minV, maxV;
            cv::minMaxLoc(towrite, &minV, &maxV);
            towrite = 255 * (towrite - minV) / (maxV - minV);
        }
        towrite.convertTo(towrite, CV_8U);
        for (int x = startx; x < startx + bx; x++) {
            for (int y = starty; y < starty + by; y++) {
                out_image.at<uint8_t>(x, y) = towrite.at<uint8_t>(x - startx, y - starty);
            }
        }
    }

    out_image.convertTo(out_image, CV_8U);
}

void DrawGaborSymbol(cv::Mat &im, const MatCell_1<cv::Mat> &allSymbol, double row, double col, double orientationIndex, int nGaborOri, double scaleIndex, double intensity) {
    int h = floor( (allSymbol[scaleIndex * nGaborOri + orientationIndex].size().width - 1) / 2.0);

    for (int r = row -h ; r < row + h; r++) {
        if (r < 0 || r >= im.size().width) continue;
        for (int c = col - h; c < col + h; c++) {
            if (c < 0 || c >= im.size().height) continue;
            double val = intensity * allSymbol[scaleIndex * nGaborOri + orientationIndex].at<double>(r - row + h, c - col + h);
            if (val > im.at<double>(r, c)) im.at<double>(r, c) = val;
        }
    }
}

void DisplayMatchedTemplate(const cv::Size latticeSize, const std::vector<double> &selectedRow, 
			const std::vector<double> &selectedCol, const std::vector<double> &selectedO, const std::vector<double> &selectedS, const std::vector<double> selectedMean, 
			const MatCell_1<cv::Mat> &allSymbol, const int nGaborOri, cv::Mat &sym) {
	double nGaborScale = allSymbol.shape()[0] / (double)nGaborOri;

	sym = cv::Mat(latticeSize, CV_64F, cv::Scalar(0));

	for (int k = 0; k < selectedRow.size(); k++) {
		double scale = selectedS[k];
		double ori = selectedO[k];
		double col = selectedCol[k];
		double row = selectedRow[k];
		if (scale < 0 || scale >= nGaborScale) {
            continue;
		}
        DrawGaborSymbol(sym, allSymbol, row, col, ori, nGaborOri, scale, sqrt(selectedMean[k]));
	}

    double minV, maxV;
    cv::minMaxLoc(sym, &minV, &maxV);
    sym = 255 * (sym - minV) / (maxV - minV);
    sym.convertTo(sym, CV_8U);
}


}  // namespace UTIL
}  // namespace SAOT
}  // namespace AOG_LIB
