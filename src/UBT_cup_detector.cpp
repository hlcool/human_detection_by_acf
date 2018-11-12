#include "UBT_cup_detector.h"
#include "ACFDetector.h"
#include <iostream>

ACFDetector *cupDetector = NULL;

bool createCupDetector(const std::string modle)
{
	if (cupDetector != NULL)
	{
		delete cupDetector;
		cupDetector = NULL;
	}
	if (modle.empty())
	{
		std::cout << "model file must provied..." << std::endl;
		return false;
	}
	cupDetector = new ACFDetector;
	cupDetector->loadModel(modle);

	return true;
}


int applyDetector(const cv::Mat frame, std::vector<CUP> &cups)
{
	cups.clear();
	if (!frame.data)
	{
		std::cout << "image data is empty..." << std::endl;
		return 0;
	}
	cv::Mat frameRGB;
	cv::cvtColor(frame, frameRGB, CV_BGR2RGB);
	std::vector<OBJECT> objects;
	objects.clear();
	objects = cupDetector->applyDetector(frameRGB);
	if (objects.size()>0)
	{
		CUP temp;
		for (size_t i = 0; i < objects.size(); i++)
		{
			temp.x		= objects[i].x;
			temp.y		= objects[i].y;
			temp.width  = objects[i].width;
			temp.height = objects[i].height;
			temp.score  = objects[i].score;
			cups.push_back(temp);
		}

		return (int)objects.size();
	}

	return 0;
}


bool destoryDetector()
{
	if (cupDetector != NULL)
	{
		delete cupDetector;
		cupDetector = NULL;
	}
	return true;
}
