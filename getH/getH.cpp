#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>


using namespace std;
using namespace cv;

//#define SQUARE_SIZE 40
//#define TEST_H



int main()
{
	Mat image1, image2;
	image1 = imread("left3.jpg");
	imshow("image1", image1);
	image2 = imread("left4.jpg");
	imshow("image2", image2);

	int width = 6, length = 9;
	Size boardSize(width, length);
	vector<Point2f> cornerPoints1;
	vector<Point2f> cornerPoints2;
	
	if (findChessboardCorners(image1, boardSize, cornerPoints1, CV_CALIB_CB_ADAPTIVE_THRESH))// CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS | CV_CALIB_CB_NORMALIZE_IMAGE))
	{
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < length; j++)
			{
				circle(image1, cornerPoints1.at(i * width + j), 4, Scalar(255, 0, 255, 0), 2);
				imshow("image1", image1);
				
				if (i == 0 && j == 0)
				{
					waitKey(0);
					cout << "该点像素坐标为:" << cornerPoints1.at(0) << endl;
				}
				
				waitKey(0);
			}
		}
	}
	else
	{
		cout << "findChessboardCorners Error" << endl;
		cout << cornerPoints1.size() << endl;
		waitKey(0);
		return 0;
	}


	if (findChessboardCorners(image2, boardSize, cornerPoints2, CV_CALIB_CB_ADAPTIVE_THRESH))// CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS | CV_CALIB_CB_NORMALIZE_IMAGE))
	{
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < length; j++)
			{
				circle(image2, cornerPoints2.at(i * width + j), 4, Scalar(255, 0, 255, 0), 2);
				imshow("image2", image2);
				
				if (i == 0 && j == 0)
				{
					waitKey(0);
					cout << "该点像素坐标为:" << cornerPoints2.at(0) << endl;
				}
				
				waitKey(0);
			}
		}
	}
	else
	{
		cout << "findChessboardCorners Error" << endl;
		cout << cornerPoints2.size() << endl;
		waitKey(0);
		return 0;
	}

	//错误检测
	if (cornerPoints1.size() != cornerPoints2.size())
	{
		cout << "Error : cornerPoints1.size() != cornerPoints2.size()" << endl;
		return 0;
	}
		

	//计算H
	Mat H = findHomography(cornerPoints1, cornerPoints2, CV_RANSAC);
	cout << "H = " << endl << H << endl;


	waitKey(0);
	return 0;
}
