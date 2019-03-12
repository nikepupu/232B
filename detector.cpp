// ToDo:
// implement: size(a,num), size(a) , min
#include "detector.hpp"
#include "saot_inference_Config.hpp"
#include "template.hpp"
#include "misc.hpp"
#include <assert.h>
#include <cmath>
#include <algorithm> 
namespace AOG_LIB {
namespace SAOT {

SAOT_Inference::SAOT_Inference()
{


	modelFolder = "working";
	configFile = "Config.mat";
	dotFile = "inferenceResult.dot";
	imageFolder = "positiveImage";
	imageName = "l1.jpg";
	featureFolder = "working";

	reLoadModel = false;
	it = 25;
	doMorphBackS1map = true;
	doCropBackImage = true;
	showMatchedTemplate = true;
	showPartBoundingBox = true;
	showObjectBoundingBox = true;

	featurefile   = featureFolder+"/"+imageName+".mat";
	morphedPatch = ( cv::Mat_<double>(config.template_size[0], config.template_size[1]) );
	LoadConfig();
}

void SAOT_Inference::LoadConfig()
{
	AOG_LIB::SAOT::LoadSAOTInferenceConfigFile( configFile, "./"+modelFolder+"/partModel_iter"+std::to_string(it)+".mat" , 
		"./"+modelFolder+"/objectModel_iter"+std::to_string(it)+".mat" ,config);
}

template <class type>
int SAOT_Inference::size(std::vector<type> tmp)
{
	return tmp.size();
}

template <class type>
std::vector<int> SAOT_Inference::size(std::vector<std::vector<type> > tmp)
{
	std::vector<int> res(2);
	res[0] =  tmp.size();
	assert(tmp.size() != 0);
	res[1] = tmp[0].size();
	return res;
}


template <class type>
std::vector<int> SAOT_Inference::size(std::vector<std::vector<type> > tmp, int dim)
{
	assert (dim == 1 || dim == 2);
	if(dim==1)
		return tmp.size();
	else
		return tmp[0].size();


}


std::vector<int> SAOT_Inference::size(cv::Mat tmp)
{

	std::vector<int> res(2);
	res[0] = tmp.rows;
	res[1] = tmp.cols;

	return res;
}


int SAOT_Inference::size(cv::Mat tmp, int dim)
{
	assert (dim == 1 || dim == 2);
	if(dim==1)
		return tmp.rows;
	else
		return tmp.cols;


}

/// this verison of find returns the non zero indices of a vector function
template <class type>
mat SAOT_Inference::find(std::vector<type> tmp)
{
	mat t;
	std::vector<int> temp;
	for(int i = 0; i < tmp.size(); i++)
		if(tmp[i] > 0)
		temp.push_back(i);

	if( temp.size() == 1){
		t.num=temp[0];
		t.type = "num";
	}
	else
	{
		t.vec = temp;
		t.type = "vec";
	}

	return t;
	
}
// void SAOT_Inference::displayMatchedTemplate(std::vector<int> latticeSize, std::vector<int> selectedRow, 
//  std::vector<int> selectedCol, std::vector<int> selectedO, std::vector<int> selectedS, 
//  std::vector<int>selectedMean, MatCell_1<int> allsymbol, int nGaborOri)

cv::Mat SAOT_Inference::drawGaborSymbol(cv::Mat im, MatCell_1<cv::Mat> &allsymbol, int row, int col, int orientationIndex, int nGaborOri, 
	int scaleIndex, double intensity )
{
	int h = floor( (size(allsymbol[(scaleIndex-1)*nGaborOri + orientationIndex], 1)-1)/2 );
	for(int r = row-h; r <= row+h; r++)
	{
		if(r <0 || r > size(im,1))
			continue;

		for(int c = col-h; c <= col+h; c++)
		{
			if( c < 0 || c > size(im, 2))
				continue;

			double val = intensity * allsymbol[(scaleIndex-1)*nGaborOri + orientationIndex].at<double>(r-row+h+1,c-col+h+1);
			if(val > im.at<double>(r,c) )
				im.at<double>(r,c) = val;
		}

	}
	return im;

}

cv::Mat SAOT_Inference::displayMatchedTemplate(std::vector<int> &latticeSize, std::vector<int> &selectedRow, 
 std::vector<int> &selectedCol, std::vector<int> &selectedO, std::vector<int> &selectedS, 
 std::vector<int> &selectedMean, MatCell_1<cv::Mat> &allsymbol, int &nGaborOri)
{
  int nGaborScale = allsymbol.shape()[0] * allsymbol.shape()[1] / nGaborOri;
  cv::Mat sym= ( cv::Mat_<double>(latticeSize[0], latticeSize[1]) );
  int nRow = sym.rows;
  int nCol = sym.cols;

  for(int i=0; i < nRow; i++ )
  	for(int j = 0; j < nCol; j++)
  		sym.at<double>(i,j)=0;



  for(int k = 0; k < selectedRow.size(); k++)
  {
  	double scale = selectedS[k]+1;
  	int ori = selectedO[k]+1;
  	int col = selectedCol[k];
  	int row = selectedRow[k];
  	if(scale < 1 || scale > nGaborScale)
  		continue;
  	sym = drawGaborSymbol( sym, allsymbol, row, col, ori, nGaborOri, scale, sqrt(selectedMean[k]) );

  }

  return sym;
}

void SAOT_Inference::TraceBack(SAOTInferenceConfig &configs)
{
	 int gaborCount = 0;
	 int count;
	 double val,actualPartRotation ;
	 int r, actualPartRotationInd ;
	 for(int iPart = 0; iPart < configs.numCandPart; iPart++)
	 {
	 	mat temp = find( config.allS3SelectedOri.at<int>(bestRotInd,iPart) == config.part_rotation_range );
	 	assert(temp.type == "num");
		r = int(temp.num); // the index of part rotation

		double Fx = therex + floor(.5+configs.allS3SelectedOri.at<int>(bestRotInd,iPart)/configs.subsample_M2/ configs.subsample_S2);
		double Fy = therey + floor(.5+configs.allS3SelectedCol.at<int>(bestRotInd,iPart)/configs.subsample_M2/configs.subsample_S2); // sub-sampled position

		// size need to return a 2d array
		std::vector<int> imagesize = size(MAX2map[r][iPart][bestRes]); // subsampled image size 
		
		// set default values of some output variables
		int bestPartRes = bestRes;
		std::vector<double> partScores= std::vector<double>(configs.numCandPart,0);
		partScores[0] = min(MAX2map[r][iPart][bestRes]);
		int actualPartRotationInd;
		if (Fx >= 1 && Fx <= imagesize[0] && Fy >= 1 && Fy <= imagesize[1])
		{
			////////////////////////////////////////////////////
			//need to declare the type for tmp might need to do 
			//a deep copy instead of a shallow copy
			///////////////////////////////////////////////////
			cv::Mat tmp = MAX2map[r][iPart][bestRes];
			partScores[iPart] = tmp.at<double>(Fx,Fy);
			
			tmp = MAX2ResolutionTrace[r][iPart][bestRes];
			bestPartRes = tmp.at<int>(Fx,Fy) + 1; // best part resolution
			// current_size is a 2d array, size need to return a pointer
			std::vector<int> current_size = size(tmp);
			
			tmp = largerMAX2LocTrace[r][iPart][bestPartRes];
			std::vector<int> new_size = size(tmp);
			Fx = floor(.5+Fx*(new_size/current_size));
			Fy = floor(.5+Fy*(new_size/current_size));
			
			int translationInd, transformInd;
			if(Fx >= 1 && Fx <= size(tmp,1) && Fy >= 1 && Fy <= size(tmp,2))
				translationInd = tmp.at<int>(Fx,Fy) + 1;
			else
				translationInd = floor(size(M2RowColShift,1)/2);
			
			
			tmp = largerMAX2TransformTrace[r][iPart][bestPartRes];
			if(Fx >= 1 && Fx <= size(tmp,1) && Fy >= 1 && Fy <= size(tmp,2))
				transformInd = tmp.at<int>(Fx,Fy) + 1;
			else
				transformInd = floor(configs.num_part_rotation/2) + 1;
			
			
			actualPartRotationInd = transformInd - configs.num_part_rotation*(ceil(double(transformInd)/configs.num_part_rotation)-1);
			Fx = floor( Fx + M2RowColShift.at<int>(translationInd,0) * configs.partSizeX/configs.subsample_S2 );
			Fy = floor( Fy + M2RowColShift.at<int>(translationInd,1)  * configs.partSizeY/configs.subsample_S2 );
		}
		else
		{
			actualPartRotationInd = r;
		}
		
		actualPartRotation = configs.part_rotation_range[actualPartRotationInd];

		// find the part location at the higher resolution
		Fx = (Fx-1 + .5) * configs.subsample_S2 * configs.subsample_M2;
		Fy = (Fy-1 + .5) * configs.subsample_S2 * configs.subsample_M2;
		partAbsoluteLocation[iPart][0] = Fx;
		partAbsoluteLocation[iPart][1] = Fy;
		partAbsoluteResolution[iPart] = (int)bestPartRes;
		partAbsoluteRotation[iPart] = (int)actualPartRotation;

		std::vector<int> denseX,denseY;
		denseX.clear(); 
		denseY.clear();

		if(doMorphBackS1map)
	    {
	        //some precomputation
	        for(int i = 0; i < configs.partSizeX; i++)
	        {
	        	denseX.emplace_back(-floor(configs.partSizeX/2) + i+1);
	        }

	       	for(int i = 0; i < configs.partSizeY; i++)
	        {
	        	denseY.emplace_back(-floor(configs.partSizeY/2) + i+1);
	        }
	        count = 0;
	        ///////////////////////////////////////////////////
	        ////// need to modify this part
	        ////// based on the data type Shu Wang defined
	        ///////////////////////////////////////////////////

	        std::vector<double> inRow= std::vector<double>(denseX.size() * denseY.size(), 0.0);
	        std::vector<double> inCol= std::vector<double>(denseX.size() * denseY.size(), 0.0);
	        for(auto y : denseY)
	            for (auto x : denseX)
	            {
	                count = count+1;
	                inRow[count] = x;
	                inCol[count] = y;
	            }
	          
	     
	        //////////////////////////////////////////////////////
	        int tScale = 0, rScale = 1, cScale = 1; 
	       
	        std::vector<double> inO = inRow;
	        std::vector<double> inS = inCol;


	        std::vector<double> outRow, outCol;

	        //////////////////////////////////////////////////////
	        ////TemplateAffineTransform is not done
	        ///////////////////////////////////////////////////

	        // [outRow, outCol] = ...
	        //     TemplateAffineTransform(tScale,rScale,cScale,
	        //         actualPartRotation,inRow,inCol,inO,inS,numOrient);
	        
			// crop the feature patch that is registered to the part template
	        
	        MatCell_1<cv::Mat> tmpCropped;
	        //////////////////////////////////////////////////////
	        /// Some bug in CropInstance needs to be fixed
	        /////////////////////////////////////////////////////

			// CropInstance(configs , map_sum1_find[bestPartRes] , tmpCropped ,Fx,Fy,
			// 	actualPartRotation,tScale,1,
			// 	outRow.data(),outCol.data(),
			// 	configs.num_orient,1,configs.partSizeX,configs.partSizeY
			// 	);

			for(int o = 0; o < configs.num_orient; o++)
				morphedSUM1map[o](cv::Rect_<int>(configs.PartLocX[iPart], configs.PartLocX[iPart]+configs.partSizeX,
				 configs.PartLocY[iPart], configs.PartLocY[iPart]+configs.partSizeY )) = tmpCropped[o];
				
			
			// also crop the corresponding image patch (for each part)

			// tmpCropped = mexc_CropInstance(ImageMultiResolution(bestPartRes),Fx,Fy,...
			// 	actualPartRotation,tScale,1,...
			// 	outRow.data(),outCol.data(),...
			// 	1,1,partSizeX,partSizeY);

			morphedPatch(cv::Rect_<int>(configs.PartLocX[iPart], configs.PartLocX[iPart]+configs.partSizeX,
				 configs.PartLocY[iPart], configs.PartLocY[iPart]+configs.partSizeY )) = tmpCropped[0];
		}
	 
		
		// ==== continue to trace back Gabor elements based on the part localization ====

		std::vector<int> gaborXX;
		std::vector<int> gaborYY;
		std::vector<int> gaborOO;
		std::vector<int> gaborMM;

		// //Gabor basis elements locations

		
		for(int j = 0; j <size(configs.selectedlambda[iPart],1); j++ )
		{
			double gaborX = floor(Fx + configs.allSelectedx[iPart][actualPartRotationInd].at<double>(j,0) );
			double gaborY = floor(Fy + configs.allSelectedy[iPart][actualPartRotationInd].at<double>(j,0) );
			double gaborO = configs.allSelectedOrient[iPart][actualPartRotationInd].at<double>(j,0);
			if(gaborX > 0 && gaborX <= size(M1Trace[bestPartRes][1],1) && gaborY > 0 && gaborY <= size(M1Trace[bestPartRes][1],2) )
			{
				int trace = M1Trace[bestPartRes][gaborO+1].at<int>(gaborX,gaborY) + 1;
				int dx = M1RowShift[gaborO+1].at<int>(trace,1);
				int dy = M1ColShift[gaborO+1].at<int>(trace,1);
				int shiftedo = M1OriShifted[gaborO+1].at<int>(trace,1);
				double gaborX = floor(.5 + gaborX + dx);
				double gaborY = floor(.5 + gaborY + dy);
				double gaborO = double(shiftedo);
			}
			
			gaborXX.push_back(gaborX);
			gaborYY.push_back(gaborY);
			gaborOO.push_back(gaborO);
			
			gaborCount = gaborCount + 1;
			gaborAbsoluteLocation[gaborCount][1] = gaborX;
			gaborAbsoluteLocation[gaborCount][2] = gaborY;
			gaborAbsoluteRotation[gaborCount] = gaborO; // start from 0
			gaborAbsoluteResolution[gaborCount] = bestPartRes;

			if(gaborX > 0 && gaborX <= size(M1Trace[bestPartRes][1],1) && gaborY > 0 && gaborY <= size(M1Trace[bestPartRes][1],2) )
				val = map_sum1_find[bestPartRes][gaborO+1].at<double>(gaborX,gaborY);
			else
				val = 0;
			
			gaborMM.push_back(std::max(0.0,sqrt(val)-.2));
			gaborResponses[gaborCount] = val;
		}
		cv::Mat tmpMatchedSym;
		if (showMatchedTemplate)
		{
			//render the template for each part separately, then overlay the rendered images
			auto sz = size(ImageMultiResolution[bestPartRes]);
			auto selectedS = std::vector<int>(gaborXX.size(), 0);
			tmpMatchedSym = displayMatchedTemplate(sz,gaborXX,
				gaborYY,gaborOO, selectedS,gaborMM, configs.allSymbol, configs.num_orient);

			cv::resize(tmpMatchedSym,tmpMatchedSym , 
				cv::Size(imageSizeAtBestObjectResolution[0],imageSizeAtBestObjectResolution[1] )) ;

			matchedSym = cv::max(matchedSym,tmpMatchedSym);
			if(showPartBoundingBox)
			{
	            int margin = 3;
	            //xx = repmat((1:partSizeX),1,margin*2);
	            cv::Mat src= cv::Mat::zeros(1,configs.partSizeX, cv::CV_64F);
	            for(int i = 0; i < configs.partSizeX;i++)
	            	src.at<double>(1,i) = i+1;
	            

	            

	           	// cv::Mat xx = cv::repeat( src,1,margin*2,xx);
	            // std::vector<int> yt;

	            // for(int y = 1; y <= magin; y++ )
	            // 	yt.push_back(y);

	            // for(int y = partSizeY-margin+1; y <= partSizeY; y++)
	            // 	yt.push_back(y);


	            // for y = [1:margin partSizeY-margin+1:partSizeY]
	            //     yt = [yy,ones(1,partSizeX)*y];

	           
	           	

	            // yy = [yy,repmat((1:partSizeY),1,margin*2)];

	            // for x = [1:margin partSizeX-margin+1:partSizeX]
	            //     xx = [xx,ones(1,partSizeY)*x];
	            // end
	            // inRow = single(xx-floor(partSizeX/2)); inCol = single(yy-floor(partSizeY/2));
	            // tScale = 0; rScale = 1; cScale = 1; inO = zeros(numel(inRow),1,'single'); inS = zeros(numel(inRow),1,'single');
	            // [outRow, outCol] = ...
	            //     mexc_TemplateAffineTransform(tScale,rScale,cScale,...
	            //         actualPartRotation,inRow,inCol,inO,inS,numOrient);
	                
	            // //directly overwrite the corresponding pixels
	            // matchedBoundingBox = imresize(matchedBoundingBox,size(ImageMultiResolution{bestPartRes}),'nearest');
	            // for p = 1:length(outRow)
	            //     x = floor(.5 + outRow(p) + Fx); y = floor(.5 + outCol(p) + Fy);
	            //     if x > 0 && x <= size(matchedBoundingBox,1) && y > 0 && y <= size(matchedBoundingBox,2)
	            //         matchedBoundingBox(x,y,:) = [1 .5 .3];
	            //     end
	            // end
	            // matchedBoundingBox = imresize(matchedBoundingBox,size(ImageMultiResolution{bestRes}),'nearest');
			} //if(showPartBoundingBox)
		}// if showMatchedTemplate
		
	}//for iPart
	}//TraceBack function
 


 }//SAOT
 }//AOG_LIB

