#ifndef SAOT_Inference_HPP_
#define SAOT_Inference_HPP_

#include <string>
#include "saot_inference_Config.hpp"
#include "saot_inference_Config.hpp"
namespace AOG_LIB {
namespace SAOT{

	union mat
	{
		int num;
		std::vector<int> vec;
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

	class SAOT_Inference 
	{

	public:
		SAOT_Inference();
		void TraceBack();

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

	/// this verison of find returns the non zero indices of a vector function
	template <class type>
	mat find(std::vector<type> tmp);






	/////////////
	SAOTInferenceConfig config;
	 
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


	//////////// configuration file variables


	};
}
}



#endif