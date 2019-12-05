#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>  
#include <string>
#include <fstream>
  
using namespace cv;
using namespace std;

#define imageNumber 13   //need change
int main()
{
	Mat frame, frame_gray, frame_color;
	vector<Point2f> pointbuf;
	vector< vector<Point2f> > imagePoints;
	Size imageSize;//the size of a frame
	float squareSize = 30.0;//the size of a small square
	vector<Mat> Re, Te;
	int i = 0, sucNumber = 0;
	int flags = 0;
	bool found = false;
	bool calirated = false;
	char  filename[50];
	char filename1[50];
	int count;
	Size boardSize(6, 9);   //the size of the board
	for (count = 1; count <= imageNumber; count++)    //need change
	{

		sprintf(filename, "/home/jack/Desktop/C++/calibration/images/left%d.jpg", count);   //need change
		frame = imread(filename, 1);
		imshow("1", frame);
		imageSize = frame.size();
		frame_color = frame.clone();
		cvtColor(frame, frame_gray, CV_RGB2GRAY);
		found = findChessboardCorners(frame_gray, boardSize, pointbuf, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);//| CV_CALIB_CB_FAST_CHECK| CV_CALIB_CB_NORMALIZE_IMAGE);
		if (!found)
		{
			std::cout << "the corner of the image" << count << " is not found" << endl;
		}
		else
		{
			drawChessboardCorners(frame_color, boardSize, pointbuf, found);
			sprintf(filename1, "/home/jack/Desktop/C++/calibration/images/left%d.jpg", count);   //need change
			//imwrite(filename1, frame_color);
			imshow("corner", frame_color);
			//cvWaitKey();
			cvWaitKey(30);
			cornerSubPix(frame_gray, pointbuf, boardSize, Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			imagePoints.push_back(pointbuf);
			sucNumber++;
			std::cout << sucNumber << endl;
		}
	}
	/**************************************************************************************************************/
	vector< vector<Point3f> >  object_Points;        /****  定义世界真实坐标   ****/
	vector<int>  point_counts;
	for (int t = 0; t < sucNumber; t++)
	{
		vector<Point3f> tempPointSet;
		for (int i = 0; i < boardSize.height; i++)
		{
			for (int j = 0; j < boardSize.width; j++)
			{
				Point3f tempPoint;
				tempPoint.x = i*squareSize;
				tempPoint.y = j*squareSize;
				tempPoint.z = 0;
				tempPointSet.push_back(tempPoint);
			}
		}
		object_Points.push_back(tempPointSet);
	}
	for (int i = 0; i < sucNumber; i++)
	{
		point_counts.push_back(boardSize.width*boardSize.height);
	}
	/* 计算内外参数矩阵 */
	cv::Matx33d intrinsic_matrix;    /*****内参矩阵****/
	cv::Vec4d distortion_coeffs;     /* 畸变向量k1,k2,k3,k4*/
	std::vector<cv::Vec3d> rotation_vectors;                           /* 旋转矩阵 */
	std::vector<cv::Vec3d> translation_vectors;                        /* 平移矩阵*/
	flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	//flags |= cv::fisheye::CALIB_CHECK_COND;
	flags |= cv::fisheye::CALIB_FIX_SKEW;
	fisheye::calibrate(object_Points, imagePoints, imageSize, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, flags, cv::TermCriteria(3, 20, 1e-6));

	cout << "intrinsic_matrix=" << intrinsic_matrix << endl;
	cout << "distortion_coeffs=" << distortion_coeffs << endl;
	//cout << "rotation_vectors=" << rotation_vectors << endl;
	//cout << "translation_vectors=" << translation_vectors << endl;

	cvWaitKey(0);
}

