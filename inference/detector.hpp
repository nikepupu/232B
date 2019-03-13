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
		SAOT_Inference();
		void TraceBack(SAOTConfig &configs);

	private:
	void LoadConfig();
	/////////////
	/// size function
	template <class type>
	std::vector<int> size(std::vector<std::vector<type> > tmp, int dim);

	template <class type>
	std::vector<int> size(std::vector<std::vector<type> > tmp);

	template <class type>
	int size(std::vector<type> tmp);

	
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

	cv::Mat displayMatchedTemplate(std::vector<int> &latticeSize, std::vector<int> &selectedRow, 
	 std::vector<int> &selectedCol, std::vector<int> &selectedO, std::vector<int> &selectedS, 
	 std::vector<int> &selectedMean, MatCell_1<cv::Mat> &allsymbol, int &nGaborOri);

	cv::Mat drawGaborSymbol(cv::Mat im, MatCell_1<cv::Mat> &allsymbol, int row, int col, int orientationIndex, int nGaborOri, 
	int scaleIndex, double intensity );



	/////////////
	SAOTConfig config;
	 
	std::string modelFolder;
	std::string configFile;
	std::string dotFile;
	std::string imageFolder;
	std::string imageName;
	std::string featureFolder;

	std::string featurefile;
	bool reLoadModel;
	int  it;
	bool doMorphBackS1map;
	bool doCropBackImage;
	bool showMatchedTemplate;
	bool showPartBoundingBox;
	bool showObjectBoundingBox;


	int bestRotInd;

	std::vector<int> imageSizeAtBestObjectResolution;

	MatCell_2<cv::Mat> map_sum1_find;
	MatCell_1<cv::Mat> SUM1map;

	MatCell_1<cv::Mat> ImageMultiResolution;





	int therex;
	int therey;
	int bestRes;

	MatCell_1<cv::Mat> M1RowShift;
	MatCell_1<cv::Mat> M1ColShift;
	MatCell_1<cv::Mat> M1OriShifted;
	MatCell_1<cv::Mat> gaborResponses;
	MatCell_1<cv::Mat> morphedSUM1map;

	MatCell_2<cv::Mat> M1Trace;

	MatCell_3<cv::Mat> MAX2map;
	MatCell_3<cv::Mat> MAX2ResolutionTrace;
	MatCell_3<cv::Mat> largerMAX2LocTrace;
	MatCell_3<cv::Mat> largerMAX2TransformTrace;




	cv::Mat morphedPatch;

	cv::Mat M2RowColShift;
	cv::Mat matchedSym;



	std::vector<std::vector<int> > partAbsoluteLocation = std::vector<std::vector<int> >(config.numCandPart, std::vector<int>(2) );
	std::vector<int> partAbsoluteRotation = std::vector<int >(config.numCandPart );
	std::vector<int> partAbsoluteResolution = std::vector<int>(config.numCandPart );
	std::vector<std::vector<int> > gaborAbsoluteLocation = std::vector<std::vector<int> >(config.num_element, std::vector<int>(2) );
	std::vector<int > gaborAbsoluteRotation = std::vector<int>(config.num_element);
	std::vector<int> gaborAbsoluteResolution = std::vector<int>(config.num_element );


	//////////// configuration file variables


	};
}
}



#endif