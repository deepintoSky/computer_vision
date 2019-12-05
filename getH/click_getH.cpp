#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

//#define SQUARE_SIZE 40
Mat image1, image2;
vector<Point2f> cornerPoints1, cornerPoints2;

void on_mouse1(int event, int x, int y, int flags, void *ustc)
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		circle(image1, Point2f(x, y), 4, Scalar(255, 0, 255), 0);
		cornerPoints1.push_back(Point2f(x, y));
		cout << "(x,y) :" << endl << x << "," << y << endl;
		imshow("image1", image1);
	}
}

void on_mouse2(int event, int x, int y, int flags, void *ustc)
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		circle(image2, Point2f(x, y), 4, Scalar(255, 0, 255), 0);
		cornerPoints2.push_back(Point2f(x, y));
		cout << "(x,y) :" << endl << x << "," << y << endl;
		imshow("image2", image2);
	}
}


int main()
{
	image1 = imread("left3.jpg");
	namedWindow("image1", 1);
	setMouseCallback("image1", on_mouse1);
	imshow("image1", image1);
	waitKey(0);

	image2 = imread("left4.jpg");
	namedWindow("image2", 2);
	setMouseCallback("image2", on_mouse2);
	imshow("image2", image2);
	waitKey(0);

	cout << "cornerPoints1:"<<endl<<cornerPoints1 << endl;
	cout << "cornerPoints2:" << endl<<cornerPoints2<<endl;
	cout << endl;

	if (cornerPoints1.size() != cornerPoints2.size())
	{
		cout << "Error : cornerPoints1.size() != cornerPoints2.size()" << endl;
		return 0;
	}

	//计算H
	Mat H = findHomography(cornerPoints1, cornerPoints2, CV_RANSAC);
	cout << "H = " << endl << H << endl;

	//waitKey(0);
	return 0;

}
