#include "misc.hpp"
#include "math.h"
#include "./util/meta_type.hpp"
#include "saot_config.hpp"
#include <boost/multi_array.hpp>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
# define PI 3.1415926
# define ROUND(x) (floor((x)+.5))
# define NEGMAX -1e10
# define ABS(x) ((x)>0? (x):(-(x)))

int px(int x, int y, int lengthx, int lengthy)  /* the image is lengthx*lengthy */
{            
   return (x + (y-1)*lengthx - 1); 
}

namespace AOG_LIB {
namespace SAOT {

void DrawElement(cv::Mat &Template, int mo, int mx, int my, double w, int sizex, 
	int sizey, int halfFilterSize, MatCell_1<cv::Mat> &allSymbol) 
{
  int x, y, here; 
  double a; 
		  
  for (x=mx-halfFilterSize; x<=mx+halfFilterSize; x++)
	 for (y=my-halfFilterSize; y<=my+halfFilterSize; y++)
	  if ((x>=0)&&(x<sizex)&&(y>=0)&&(y<sizey)) 
		{
		 a = w*allSymbol[mo].at<double>(x-mx+halfFilterSize, y-my+halfFilterSize);
		 
		 if (Template.at<double>(x,y) < a)
			 Template.at<double>(x,y) = a;
		}
}

/* local maximum pooling at (x, y, orient) */
double LocalMaximumPooling(int img, int orient, int x, int y, int *trace, 
	int numShift, cv::Mat &xShift, cv::Mat &yShift, cv::Mat &orientShifted, 
	MatCell_1<cv::Mat> &SUM1map, int sizex, int sizey, int numImage) 
{
   double maxResponse, r;
   int shift, x1, y1, orient1, mshift; 
 
   maxResponse = NEGMAX;  
   mshift = 0; 
   for (shift=0; shift<numShift; shift++)
   {
	   x1 = x + xShift.at<int>(orient, shift); 
	   y1 = y + yShift.at<int>(orient, shift); 
	   orient1 = orientShifted.at<int>(orient, shift);
	   if ((x1>=0)&&(x1<sizex)&&(y1>=0)&&(y1<sizey)) 
		{            
		   int idx = orient1*numImage+img;
		   r = SUM1map[idx].at<double>(x1, y1);
		   if (r>maxResponse)
				{
				   maxResponse = r;   
				   mshift = shift; 
				}
		 }
	 }
   trace[0] = mshift;  /* return the shift in local maximum pooling */
   return(maxResponse); 
}

/* the local maximal Gabor inhibits overlapping Gabors */
void NonMaximumSuppression(int img, int mo, int mx, int my, const SAOTConfig& config, 
	MatCell_1<cv::Mat> &SUM1map, cv::Mat &MAX1map, cv::Mat &trackMap, MatCell_1<cv::Mat> &Correlation, 
	cv::Mat &pooledMax1map, cv::Mat &orientShifted, cv::Mat &xShift, cv::Mat &yShift, 
	int numShift, int startx, int starty, int endx, int endy, int sizexSubsample, 
	int sizeySubsample, const cv::Size &img_size) 
{
   int x, y, orient, x1, y1, orient1, i, here, shift, startx0, endx0, starty0, endy0, trace[2]; 
   double maxResponse, maxResponse1; 
   int numImage = config.num_image;
   int numOrient = config.num_orient;
   int halfFilterSize = config.half_filter_size;
   int subsample = config.subsample;
   int sizex = ROUND(img_size.height);
   int sizey = ROUND(img_size.width);
   int locationShiftLimit = config.location_shift_limit;
   cv::Size newSize(4*halfFilterSize+1, 4*halfFilterSize+1);

   /* inhibit on the SUM1 maps */
   for (orient=0; orient<numOrient; orient++)   
	 {
	   cv::Mat &f = SUM1map[orient*numImage+img];   
	   cv::Mat &fc = Correlation[mo+orient*numOrient];   
	   for (x=MAX(1, mx-2*halfFilterSize)-1; x<MIN(sizex, mx+2*halfFilterSize); x++)
		 for (y=MAX(1, my-2*halfFilterSize)-1; y<MIN(sizey, my+2*halfFilterSize); y++)
		 {
		  f.at<double>(x, y) *= fc.at<double>(x-mx+2*halfFilterSize+1, y-my+2*halfFilterSize+1);
		 }
	  }
   /* update the MAX1 maps */
   startx0 = floor((mx-2*halfFilterSize)/subsample)-locationShiftLimit+1; 
   starty0 = floor((my-2*halfFilterSize)/subsample)-locationShiftLimit+1; 
   endx0 = floor((mx+2*halfFilterSize)/subsample)+locationShiftLimit; 
   endy0 = floor((my+2*halfFilterSize)/subsample)+locationShiftLimit; 
   for (orient=0; orient<numOrient; orient++)   
	 {
	  i = orient*numImage+img; 
	  for (x=MAX(startx, startx0); x<=MIN(endx, endx0); x++)
		 for (y=MAX(starty, starty0); y<=MIN(endy, endy0); y++)
		 { /* go over the locations that may be affected */         
		   here = px(x, y, sizexSubsample, sizeySubsample);        
		   maxResponse = MAX1map.at<double>(i, here); 
		   shift = trackMap.at<int>(i, here);
		   orient1 = orientShifted.at<int>(orient, shift);    
		   x1 = x*subsample + xShift.at<int>(orient, shift); 
		   y1 = y*subsample + yShift.at<int>(orient, shift);            
		   if ((x1-mx>=-2*halfFilterSize)&&(x1-mx<=2*halfFilterSize)&&
			   (y1-my>=-2*halfFilterSize)&&(y1-my<=2*halfFilterSize)) 
			 { /* if the previous local maximum is within the inhibition range */
			  
			  cv::Mat &fc = Correlation[mo+orient1*numOrient];
			  if(fc.at<double>(x1-mx+2*halfFilterSize+1,y1-my+2*halfFilterSize+1)==0.) 
				 {   /* if it is indeed inhibited */
					 maxResponse1 = LocalMaximumPooling(img, orient, x*subsample, y*subsample, trace, numShift, 
						xShift, yShift, orientShifted, SUM1map, sizex, sizey, numImage);
					 trackMap.at<int>(i, here) = trace[0];  
					 MAX1map.at<double>(i, here) = maxResponse1;
					 pooledMax1map.at<double>(orient, here) += (maxResponse1-maxResponse);                  
				 }
			 }         
		 }        
	  }        
}


void Sigmoid(const SAOTConfig& config, MatCell_2<cv::Mat> &map_sum1_find)
{
	double satur = config.saturation;
	for (int i=0; i<map_sum1_find.shape()[0]; i++)
		for (int j=0; j<map_sum1_find.shape()[1]; j++)
		{
			int nRow = map_sum1_find[i][j].rows;
			int nCol = map_sum1_find[i][j].cols;
			for (int r=0; r<nRow; ++r)
				for (int c=0; c<nCol; ++c)
				{
					map_sum1_find[i][j].at<double>(r,c) = 
					satur*(2./(1.+exp(-2.*map_sum1_find[i][j].at<double>(r,c)/satur))-1.);
				}
		}	

}

// LocalNormalize
void LocalNormalize(const cv::Size &img_size, int num_filters,
                        int half_filter_size, const cv::Size &local_half,
                        double threshold_factor,
                        MatCell_1<cv::Mat> &filter_responses)
{

	int x, y, leftx, rightx, upy, lowy, k, startx[8], endx[8], starty[8], endy[8], copyx[8], copyy[8], fx, fy;  
	double maxAverage, averageDivide; 
	int numOrient, halfFilterSize, localHalfx, localHalfy, sizex, sizey; 
	double thresholdFactor;
	int orient;

	// init variables
	thresholdFactor = threshold_factor;
	halfFilterSize = half_filter_size;
	numOrient = num_filters;
	sizex = ROUND(img_size.height);
	sizey = ROUND(img_size.width);
	localHalfx = ROUND(local_half.height);
	localHalfy = ROUND(local_half.width);

	// original global variables
	cv::Mat SUM1mapAll, integralMap, averageMap; 

	/* compute the sum over all the orientations at each pixel */
	SUM1mapAll = cv::Mat::zeros(img_size, CV_64F);
	for (int x=0; x<sizex; x++)
	   for (int y=0; y<sizey; y++)
	   {
		for (orient=0; orient<numOrient; orient++)
		   {    
				SUM1mapAll.at<double>(x, y) += filter_responses[orient].at<double>(x, y); 
			}
	   }

	/* compute the integral map */
	integralMap = cv::Mat::zeros(img_size, CV_64F);
	integralMap.at<double>(0, 0) = SUM1mapAll.at<double>(0, 0);
	for (x=1; x<sizex; x++)
		integralMap.at<double>(x, 0) = integralMap.at<double>(x-1, 0)+SUM1mapAll.at<double>(x, 0);
	for (y=1; y<sizey; y++)
		integralMap.at<double>(0, y) = integralMap.at<double>(0, y-1)+SUM1mapAll.at<double>(0, y); 
	for (x=1; x<sizex; x++)
	   for (y=1; y<sizey; y++)
		integralMap.at<double>(x, y) = integralMap.at<double>(x, y-1)+integralMap.at<double>(x-1, y)-integralMap.at<double>(x-1, y-1)+SUM1mapAll.at<double>(x, y); 
	
	/* compute the local average around each pixel */
	averageMap = cv::Mat::zeros(img_size, CV_64F);
	leftx = halfFilterSize+localHalfx; rightx = sizex-halfFilterSize-localHalfx; 
	upy = halfFilterSize+localHalfy; lowy = sizey-halfFilterSize-localHalfy; 
	maxAverage = NEGMAX; 
	if ((leftx<rightx) && (upy<lowy))
	{
	for (x=leftx; x<rightx; x++)
	   for (y=upy; y<lowy; y++)
	   {
			averageMap.at<double>(x,y) = (integralMap.at<double>(x+localHalfx,y+localHalfy) 
				- integralMap.at<double>(x-localHalfx-1,y+localHalfy) - integralMap.at<double>(x+localHalfx,y-localHalfy-1)
				+ integralMap.at<double>(x-localHalfx-1,y-localHalfy-1))/(2.*localHalfx+1.)/(2.*localHalfy+1.)/numOrient; 
			if (maxAverage < averageMap.at<double>(x,y))
				maxAverage = averageMap.at<double>(x,y); 
	   }

	/* take care of the boundaries */
	k = 0;  
	/* four corners */
	startx[k] = 0; endx[k] = leftx; starty[k] = 0; endy[k] = upy; copyx[k] = leftx; copyy[k] = upy; k++; 
	startx[k] = 0; endx[k] = leftx; starty[k] = lowy; endy[k] = sizey; copyx[k] = leftx; copyy[k] = lowy-1; k++;  
	startx[k] = rightx; endx[k] = sizex; starty[k] = 0; endy[k] = upy; copyx[k] = rightx-1; copyy[k] = upy; k++; 
	startx[k] = rightx; endx[k] = sizex; starty[k] = lowy; endy[k] = sizey; copyx[k] = rightx-1; copyy[k] = lowy-1; k++; 
	/* four sides */
	startx[k] = 0; endx[k] = leftx; starty[k] = upy; endy[k] = lowy; copyx[k] = leftx; copyy[k] = -1; k++; 
	startx[k] = rightx; endx[k] = sizex; starty[k] = upy; endy[k] = lowy; copyx[k] = rightx-1; copyy[k] = -1; k++; 
	startx[k] = leftx; endx[k] = rightx; starty[k] = 0; endy[k] = upy; copyx[k] = -1; copyy[k] = upy; k++; 
	startx[k] = leftx; endx[k] = rightx; starty[k] = lowy; endy[k] = sizey; copyx[k] = -1; copyy[k] = lowy-1; k++; 

	/* propagate the average to the boundaries */
	for (k=0; k<8; k++) 
		for (x=startx[k]; x<endx[k]; x++)
			for (y=starty[k]; y<endy[k]; y++)
			{
				if (copyx[k]<0)
					fx = x; 
				else 
					fx = copyx[k]; 
				if (copyy[k]<0)
					fy = y; 
				else 
					fy = copyy[k]; 
				averageMap.at<double>(x, y) = averageMap.at<double>(fx, fy); 
			}

	 /* normalize the responses by local averages */
	 for (x=0; x<sizex; x++)
	   for (y=0; y<sizey; y++)
	   {      
		averageDivide = MAX(averageMap.at<double>(x, y), maxAverage*thresholdFactor); 
		for (orient=0; orient<numOrient; orient++)
				filter_responses[orient].at<double>(x, y) /= averageDivide; 
	   }
	}

}

void CropInstance(const SAOTConfig& config, const MatCell_1<cv::Mat> &src, MatCell_1<cv::Mat> &dest, 
		double rshift, double cshift, double rotation, double scaling, double reflection, 
		const double* transformedRow, const double* transformedCol, int nOri, int nScale, int destHeight, int destWidth)
{

	int nPixel;
	nPixel = destWidth * destHeight;

	int i, s, o, destR, destC; /* destination image: scale, orientation */
	int srcS, srcO, r, c; /* src image: scale, orientation */
	int indSrc, indDest;
	int srcHeight = src[0].rows;
	int srcWidth = src[0].cols;

	/* The following two FOR loops go over destimation images/feature maps
	 * at different scales and orientations.
	 */	
	for( s = 0; s < nScale; ++s )
	{
		srcS = s + scaling;
		if( srcS < 0 || srcS >= nScale )
		{
			continue;
		}

		for( o = 0; o < nOri; ++o )
		{
			srcO = o + rotation;
			while( srcO < 0 )
			{
				srcO += nOri;
			}
			while( srcO >= nOri )
			{
				srcO -= nOri;
			}
			if( reflection < 0 ) // deal with reflection
			{
				 srcO = nOri - srcO;
			}
			while( srcO < 0 )
			{
				srcO += nOri;
			}
			while( srcO >= nOri )
			{
				srcO -= nOri;
			}
			indDest = s*nOri + o;
			indSrc = srcS*nOri + srcO;
			for( i = 0; i < nPixel; ++i ) /* location */
			{
				destR = (int)transformedRow[i];
				destC = (int)transformedCol[i];
				r = rshift + destR;
				c = cshift + destC;
				if( r < 0 || r >= srcHeight || c < 0 || c >= srcWidth )
				{
					/* do nothing */
				}
				else
				{
					dest[indDest].at<double>(destR, destC) = src[indSrc].at<double>(r,c);
				}
			}
		}
	}

}

void CropInstance(const SAOTConfig& config, const cv::Mat &src, cv::Mat &dest, 
	double rshift, double cshift, double rotation, double scaling, double reflection, 
	const std::vector<double> transformedRow, const std::vector<double> transformedCol, int nOri, int nScale, int destHeight, int destWidth)
{

	int nPixel;
	nPixel = destWidth * destHeight;

	int i, s, o, destR, destC; /* destination image: scale, orientation */
	int srcS, srcO, r, c; /* src image: scale, orientation */
	int indSrc, indDest;
	int srcHeight = src.rows;
	int srcWidth = src.cols;

	/* The following two FOR loops go over destimation images/feature maps
	 * at different scales and orientations.
	 */	
	for( s = 0; s < nScale; ++s )
	{
		srcS = s + scaling;
		if( srcS < 0 || srcS >= nScale )
		{
			continue;
		}

		for( o = 0; o < nOri; ++o )
		{
			srcO = o + rotation;
			while( srcO < 0 )
			{
				srcO += nOri;
			}
			while( srcO >= nOri )
			{
				srcO -= nOri;
			}
			if( reflection < 0 ) // deal with reflection
			{
				 srcO = nOri - srcO;
			}
			while( srcO < 0 )
			{
				srcO += nOri;
			}
			while( srcO >= nOri )
			{
				srcO -= nOri;
			}
			indDest = s*nOri + o;
			indSrc = srcS*nOri + srcO;
			for( i = 0; i < nPixel; ++i ) /* location */
			{
				destR = (int)transformedRow[i];
				destC = (int)transformedCol[i];
				r = rshift + destR;
				c = cshift + destC;
				if( r < 0 || r >= srcHeight || c < 0 || c >= srcWidth )
				{
					/* do nothing */
				}
				else
				{
					dest.at<double>(destR, destC) = src.at<double>(r,c);
				}
			}
		}
	}

}


void Instribe(cv::Mat &srcImage, cv::Mat &destImage, int val) 
{
	destImage = val * cv::Mat::ones(destImage.size(), CV_64F);
	int i, j, sx, sy, dx, dy;
	sx = srcImage.rows;
	sy = srcImage.cols;
	dx = destImage.rows;
	dy = destImage.cols;

	if (sx * dy >= sy * dx)
	{
		int newSx = dx;
		int newSy = ROUND(newSx*sy/sx);
		cv::Mat tmpImage;
		cv::Size newSize(newSy, newSx);
		cv::resize(srcImage, tmpImage, newSize);
		for (i=0; i<newSx; i++)
			for (j=0; j<newSy; j++)
			{

				destImage.at<double>(i, j+floor((dy-newSy)/2)) = tmpImage.at<double>(i,j);
			}
	}
	else
	{
		int newSy = dy; 
		int newSx = ROUND(newSy*sx/sy);
		cv::Mat tmpImage;
		cv::Size newSize(newSy, newSx);
		cv::resize(srcImage, tmpImage, newSize);
		for (i=0; i<newSx; i++)
			for (j=0; j<newSy; j++)
			{
				destImage.at<double>(i+floor((dx-newSx)/2),j) = tmpImage.at<double>(i,j);
			}

	}

}

void Histogram(MatCell_2<cv::Mat> &filteredImage, const SAOTConfig& config, const cv::Size &img_size, 
	double binSize, int numBin, cv::Mat& histog) 
{
	int sizex, sizey;  
	int orient, img, b, tot;
	int numImage = config.num_image;
	int numOrient = config.num_orient;
	int halfFilterSize = config.half_filter_size;
	double saturation = config.saturation;
	int x,y;

	sizex = ROUND(img_size.height);
	sizey = ROUND(img_size.width);
	Sigmoid(config, filteredImage);

	tot = 0; 
	for (x=halfFilterSize; x<sizex-halfFilterSize; x++)
	   for (y=halfFilterSize; y<sizey-halfFilterSize; y++)
	   {
			for (int r=0; r<numOrient; ++r)
				for (int c=0; c<numImage; ++c)
				{
					b = MIN(floor(filteredImage[r][c].at<double>(x, y)/binSize), numBin-1); 
					histog.at<double>(b, 0) += 1.; 
					tot ++; 
				}
			
	   }

	for (b=0; b<numBin; b++)
		histog.at<double>(b, 0) /= (binSize*tot);

}

// SharedSketch
void SharedSketch(const SAOTConfig& config, const cv::Size &img_size, MatCell_1<cv::Mat> &SUM1map, 
	MatCell_1<cv::Mat> &Correlation, MatCell_1<cv::Mat> &allSymbol, cv::Mat &commonTemplate, 
	MatCell_1<cv::Mat> &deformedTemplate, MatCell_1<ExpParam> &eParam, MatCell_1<BasisParam> &bParam) 
{
	int numOrient = config.num_orient;
	int locationShiftLimit = config.location_shift_limit;
	int orientShiftLimit = config.orient_shift_limit;
	int subsample = config.subsample;
	int numElement = config.num_element;
	int numImage = config.num_image;
	int halfFilterSize = config.half_filter_size;
	int numStoredPoint = config.numStoredPoint;
	int sizex, sizey;
	sizex = ROUND(img_size.height);
	sizey = ROUND(img_size.width);
	int sizexSubsample, sizeySubsample;
	sizexSubsample = floor((double)sizex/subsample); 
 	sizeySubsample = floor((double)sizey/subsample); 

	commonTemplate = cv::Mat::zeros(img_size, CV_64F);

	for (int ii=0; ii<numImage; ii++)
	{
		deformedTemplate[ii] = cv::Mat::zeros(img_size, CV_64F);
	}

	// store shift
	int orient, i, j, shift, ci, cj, si, sj, os;
	double alpha; 
	/* store all the possible shifts for all orientations */
	int numShift = (locationShiftLimit*2+1)*(orientShiftLimit*2+1); 
	cv::Mat xShift, yShift, orientShifted;
	xShift = cv::Mat(numOrient, numShift, CV_32S);
	yShift = cv::Mat(numOrient, numShift, CV_32S);
	orientShifted = cv::Mat(numOrient, numShift, CV_32S);
	/* order the shifts from small to large */
	for (orient=0; orient<numOrient; orient++)        
	{
		alpha = PI*orient/numOrient;
		ci = 0; 
		for (i=0; i<=locationShiftLimit; i++)
		   for (si=0; si<=1; si++)
			{
			 cj = 0;    
			 for (j = 0; j<=orientShiftLimit; j++)
				 for (sj=0; sj<=1; sj++)
				  {
					shift = ci*(2*orientShiftLimit+1) + cj; 
					xShift.at<int>(orient, shift) = ROUND(i*(si*2-1)*subsample*cos(alpha)); 
					yShift.at<int>(orient, shift) = ROUND(i*(si*2-1)*subsample*sin(alpha)); 
					os = orient + j*(sj*2-1);
					if (os<0)
						os += numOrient; 
					else if (os>=numOrient)
						os -= numOrient; 
					orientShifted.at<int>(orient, shift) = os; 
					
					if (j>0) 
						cj ++; 
					else if (sj==1)
						cj++; 
				 }
			 if (i>0)
				 ci ++; 
			 else if (si==1)
				 ci++;                     
			}
	}


	/* Initialize MAX1 maps */
   int img, x, y, trace[2];  
   int startx, endx, starty, endy, here; /* search within the interior of the image */
   /* calculate MAX1 maps and record the shifts */
   cv::Mat pooledMax1map, MAX1map, trackMap;
   pooledMax1map = cv::Mat::zeros(numOrient, sizexSubsample*sizeySubsample, CV_64F);
   MAX1map = cv::Mat::zeros(numImage*numOrient, sizexSubsample*sizeySubsample, CV_64F);  
   trackMap = cv::Mat::zeros(numImage*numOrient, sizexSubsample*sizeySubsample, CV_32S);

   startx = floor((halfFilterSize+1)/subsample)+1+locationShiftLimit; 
   endx = floor((sizex-halfFilterSize)/subsample)-1-locationShiftLimit; 
   starty = floor((halfFilterSize+1)/subsample)+1+locationShiftLimit; 
   endy = floor((sizey-halfFilterSize)/subsample)-1-locationShiftLimit; 
   for (x=startx; x<=endx; x++)
	  for (y=starty; y<=endy; y++)
	   {
		here = px(x, y, sizexSubsample, sizeySubsample); 
		for (orient=0; orient<numOrient; orient++)
		   {    
			 pooledMax1map.at<double>(orient, here) = 0.; 
			 for (img=0; img<numImage; img++)
			 {
				i = orient*numImage+img; 

				MAX1map.at<double>(i, here) = LocalMaximumPooling(img, orient, x*subsample, y*subsample, trace, 
					numShift, xShift, yShift, orientShifted, SUM1map, sizex, sizey, numImage);    
				trackMap.at<int>(i, here) = trace[0]; 
				pooledMax1map.at<double>(orient, here) += MAX1map.at<double>(i, here);     
			  }
			}
	   }


	// Pursue Element
	int besto, bestx, besty, mo, mx, my, t; 
	double r, maxPooled, maxResponse, average, overShoot; 
	t = 0; /* t is for iterations */
	do  
	{ /* select the next Gabor */
	 maxPooled = NEGMAX;  
	 for (x=startx; x<=endx; x++)
	  for (y=starty; y<=endy; y++)
	   {
		here = px(x, y, sizexSubsample, sizeySubsample); 
		for (orient=0; orient<numOrient; orient++)
		   {    
			r = pooledMax1map.at<double>(orient, here); 
			if (maxPooled<r)
			 {
			   maxPooled = r; 
			   besto = orient; 
			   bestx = x; 
			   besty = y; 
			  }
		   }
	   }
	 /* estimate the parameter of exponential model */
	 bParam[t].selectedOrient = besto; 
	 bParam[t].selectedx = bestx; 
	 bParam[t].selectedy = besty; 
	 average = maxPooled/numImage;    
	 j = numStoredPoint-1; 
	 while (eParam[j].storedExpectation>average)
		 j--; 
	 if (j==numStoredPoint-1)
	  {
		bParam[t].selectedlambda = eParam[j].storedlambda; 
		bParam[t].selectedLogZ = eParam[j].storedlogZ; 
	   }
	 else 
	  { /* linear interpolation */
		overShoot = (average-eParam[j].storedExpectation)/(eParam[j+1].storedExpectation-eParam[j].storedExpectation); 
		bParam[t].selectedlambda = eParam[j].storedlambda+(eParam[j+1].storedlambda-eParam[j].storedlambda)*overShoot; 
		bParam[t].selectedLogZ = eParam[j].storedlogZ+(eParam[j+1].storedlogZ-eParam[j].storedlogZ)*overShoot; 
	  }  
	 /* plot selected and perturbed Gabor and inhibit nearby Gabors */
	 DrawElement(commonTemplate, besto, bestx*subsample, besty*subsample, sqrt(average), 
	 	sizex, sizey, halfFilterSize, allSymbol);     
	 
	 here = px(bestx, besty, sizexSubsample, sizeySubsample); 
	 for (img=0; img<numImage; img++)
		{    
		  i = besto*numImage+img; 
		  maxResponse = MAX1map.at<double>(i, here); 
		  shift = trackMap.at<int>(i, here); 
		  mo = orientShifted.at<int>(besto, shift);      
		  mx = bestx*subsample + xShift.at<int>(besto, shift); 
		  my = besty*subsample + yShift.at<int>(besto, shift);

		 if (maxResponse>0.)
			{  
			   DrawElement(deformedTemplate[img], mo, mx, my, sqrt(maxResponse), sizex, sizey, 
			   halfFilterSize, allSymbol); 
			   NonMaximumSuppression(img, mo, mx, my, config, SUM1map, MAX1map, trackMap,
			   	Correlation, pooledMax1map, orientShifted, xShift, yShift, numShift,
			   	startx, starty, endx, endy, sizexSubsample, sizeySubsample, img_size);  

			} 
		 } 
	  t++; 
	}
	while (t<numElement);

}



}  // namespace SAOT
}  // namespace AOG_LIB
