#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <vector>

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
	Mat image_left, image_right, tempL, tempR;
	image_left = imread("/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/left cam/4-1.bmp");
	//imshow("image_left", image_left);
    //image_right = imread("Image2.png");
	//imshow("image_right", image_right);

    resize(image_left, image_left, Size(640,480));
    cvtColor(image_left, tempL, CV_RGB2GRAY);
    GaussianBlur(tempL, tempL, Size(5, 5), 1.5, 1.5);
    imshow("image_left", tempL);

    equalizeHist(tempL, tempL);
	imshow("equalizeHist", tempL);

	
    vector<KeyPoint> keypoints;  
	SimpleBlobDetector::Params params;  
	params.filterByArea = true;
	params.minArea = 250;
	params.maxArea = 1000;  
 
    params.filterByCircularity = true;
    params.minCircularity = 0.9;
    params.maxCircularity = 1.0;

    params.filterByCircularity = true;
    params.minCircularity = 0.5;
    params.maxCircularity = 1.0;

	// Set up detector with params
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

    // Detect blobs
    detector->detect(tempL, keypoints); 
	drawKeypoints(tempL, keypoints, tempL, Scalar(255,0,0)); 
	 
    imshow("result", tempL);
  
    cvWaitKey(0);
    return 0;
}