#include <iostream>
#include <opencv2/opencv.hpp>
#include "UBT_cup_detector.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		std::cout << "number of arguments is not right..." << std::endl;
		return -1;
	}

	createCupDetector(argv[1]);

	cv::VideoCapture cap(-1);
	if (!cap.isOpened())
	{
		std::cout << "load camera error..." << std::endl;
		return -1;
	}

	cv::Mat frameBGR, frameDetector, mergeImg;
	std::vector<CUP> cups;
	for (;;)
	{
		cap >> frameBGR;
		if (!frameBGR.data)
		{
			std::cout << "load image error..." << std::endl;
			return -1;
		}
		
		frameBGR.copyTo(frameDetector);
		applyDetector(frameDetector, cups);

		for (int d = 0; d < cups.size(); d++)
		{
			int x = cups[d].x;
			int y = cups[d].y;
			int width = cups[d].width;
			int height = cups[d].height;
			float score = cups[d].score;

			cv::Rect myROI(x, y, width, height);
			cv::rectangle(frameDetector, myROI, cv::Scalar(0, 0, 255), 2);
			std::cout << "x=" << x << " y=" << y << " width=" << cups[d].width << " height=" << cups[d].height << " Score=" << score << std::endl;
		}
		cv::hconcat(frameBGR, frameDetector, mergeImg);
		cv::imshow("Detection", mergeImg);
		if(cv::waitKey(10)>0)
			break;

	}

	destoryDetector();
	return 0;
}