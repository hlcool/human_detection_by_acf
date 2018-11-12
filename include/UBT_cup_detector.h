#ifndef _UBT_CUP_DETECTOR_H_
#define _UBT_CUP_DETECTOR_H_

#include <string>
#include <opencv2/opencv.hpp>

typedef struct tagCUP
{
	int x;
	int y;
	int width;
	int height;
	float score;
}CUP;

bool createCupDetector(const std::string modle);


int applyDetector(const cv::Mat frame, std::vector<CUP> &cups);


bool destoryDetector();


#endif




