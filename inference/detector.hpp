#ifndef SAOT_Inference_HPP_
#define SAOT_Inference_HPP_
#include "./util/meta_type.hpp"
#include <string>
#include <climits>
#include <cmath>
#include "saot_config.hpp"

namespace AOG_LIB {
namespace SAOT{

	struct mat
	{
		int num;
		std::vector<int> vec;
		std::string type;
		mat(){};
		~mat(){};
	};
	
	/// overload == operator between an number and a vector 
	template <class type>
	std::vector<int> operator==(const std::vector<type> tmp, const type t)
	{
		std::vector<int> res;
		for(int i = 0; i < tmp.size();i++)
		{
			if(tmp[i] - t == 0)
				res.emplace_back(1);
			else
				res.emplace_back(0);

		}
		return res;
	}

	

	// test if a element  equals to a vector logical array the same as above
	template <class type>
	std::vector<int> operator==(const type t, const std::vector<type> tmp)
	{
		std::vector<int> res;
		for(int i = 0; i < tmp.size();i++)
		{
			if(tmp[i] - t == 0)
				res.emplace_back(1);
			else
				res.emplace_back(0);

		}
		return res;
	}

	// vector divide by a vector
	template <class type>
	double operator/(const std::vector<type> tmp1,const std::vector<type> tmp2 )
	{
		assert(tmp1.size() == tmp2.size());
		double res =-1;
		for(int i = 0; i < tmp1.size();i++)
		{
			if (res == -1)
				res = tmp1[i]/tmp2[i];
			else
			{
				double tmp = tmp1[i]/tmp2[i];
				//if not equal the situation is undefined;
				assert(res == tmp);
			}

		}
		return res;
	}

	template<class type>
	std::vector<type> operator-(const std::vector<type> vec, const type num)
	{
		std::vector<type> res;
		for(auto it : vec)
			res.emplace_back(it-num);
		return res;
	}

	class SAOT_Inference 
	{

	public:
		SAOT_Inference(const SAOTConfig &config) : config_(config) {};
		void StartInference();
		void TraceBack(SAOTConfig &configs);

	private:
	void LoadConfig();

	/////////////
	void LoadImageAndFeature();
	/////////////
	/// size function
	template <class type>
	std::vector<int> size(std::vector<std::vector<type> > tmp, int dim);

	template <class type>
	std::vector<int> size(std::vector<std::vector<type> > tmp);

	template <class type>
	int size(std::vector<type> tmp);


	std::vector<int> size(MatCell_1<cv::Mat> tmp);

	template <class type>
	std::vector<int> size(MatCell_2<type> tmp)
	{
		std::vector<int> res;
		res.emplace_back(tmp.shape()[0]);
		res.emplace_back(tmp.shape()[1]);

		return res;
	}
	


	template <class type>
	int size(MatCell_2<type> tmp, int dim)
	{
		assert(dim==1 || dim == 2);
		if(dim == 1)
			return tmp.shape()[0];
		else return tmp.shape()[1];

	}

	int size(MatCell_3<cv::Mat> tmp, int dim)
	{
		assert(dim==1 || dim == 2 || dim == 3);
		if(dim == 1)
			return tmp.shape()[0];
		else  if (dim == 2)
			return tmp.shape()[1];
		else 
			return tmp.shape()[2];

	}

	
	std::vector<int> size(cv::Mat tmp);

	int size(cv::Mat tmp, int dim);

	// min function for mat
	template <class type>
	type min(type a, type b)
	{
		if (a < b)
			return a;
		else return b;
	}

	double min(cv::Mat tmp)
	{
		int nRow = tmp.rows;
		int nCol = tmp.cols;
		double mi = INT_MAX;
		for (int r=0; r<nRow; ++r)
			for (int c=0; c<nCol; ++c)
			{
				mi = min(tmp.at<double>(r,c), mi);
			}
		return mi;

	}	


	/// this verison of find returns the non zero indices of a vector function
	template <class type>
	mat find(std::vector<type> tmp);
	void LoadImages(std::vector<cv::Mat> &images);

	void Compute();


	/////////////
	SAOTConfig config_;

	std::vector<std::string> img_list_;
	 
	std::string modelFolder;
	std::string configFile;
	std::string dotFile;
	std::string imageFolder;
	std::string imageName;
	std::string featureFolder;

	std::string featurefile;
	bool reLoadModel;
	int  it;
	bool doMorphBackS1map = true;
	bool doCropBackImage = true;
	bool showMatchedTemplate = true;
	bool showPartBoundingBox = true;
	bool showObjectBoundingBox = true;


	int bestRotInd;
	std::vector<int> imageSizeAtBestObjectResolution;
	MatCell_2<cv::Mat> map_sum1_find;
	MatCell_1<cv::Mat> SUM1map;
	MatCell_1<cv::Mat> ImageMultiResolution;



	int objRotation = 0;
	int objResolution = 0;
	int objLocation[2] = {0,0};
	int objScore = 0; //record MAX3 score

	int therex;
	int therey;
	int bestRes;


	std::vector<double> gaborResponses;
	std::vector<double> partScores;

	cv::Mat M1RowShift;
	cv::Mat M1ColShift;
	cv::Mat M1OriShifted;
	MatCell_1<cv::Mat> morphedSUM1map;
	MatCell_2<cv::Mat> MAX1map;
	MatCell_2<cv::Mat> largerSUM2map;

	MatCell_2<cv::Mat> M1Trace;

	MatCell_3<cv::Mat> MAX2map;
	MatCell_3<cv::Mat> SUM2map;
	MatCell_3<cv::Mat> MAX2ResolutionTrace;
	MatCell_3<cv::Mat> largerMAX2LocTrace;
	MatCell_3<cv::Mat> largerMAX2TransformTrace;
	MatCell_3<cv::Mat> largerMAX2map;


	cv::Mat morphedPatch;
	cv::Mat M2RowColShift;
	cv::Mat matchedSym;


	std::vector<std::vector<int> > partAbsoluteLocation;
	std::vector<int> partAbsoluteRotation;
	std::vector<int> partAbsoluteResolution;
	std::vector<std::vector<int> > gaborAbsoluteLocation;
	std::vector<int > gaborAbsoluteRotation;
	std::vector<int> gaborAbsoluteResolution;

	//////////// configuration file variables


	};
}
}



#endif