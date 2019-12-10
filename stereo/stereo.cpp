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
    char  filename[50];


    for (int count = 1; count <= 8; count++)    //need change
	{

		sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/left cam/%d-1.bmp", count);   //need change
		image_left = imread(filename, 1);
    
        //imshow("image_left", image_left);
        //image_right = imread("Image2.png");
        //imshow("image_right", image_right);

        resize(image_left, image_left, Size(640,480));
        cvtColor(image_left, tempL, CV_RGB2GRAY);

        GaussianBlur(tempL, tempL, Size(5, 5), 1.5, 1.5);
        //imshow("image_left", tempL);

        threshold(tempL, tempL, 100, 255, THRESH_BINARY);
        imshow("BINARY", tempL);

        //equalizeHist(tempL, tempL);
        //imshow("equalizeHist", tempL);

        //Canny(tempL, tempL, 100, 200, 3);
        //imshow("canny", tempL);


        vector<KeyPoint> keypoints;  
        SimpleBlobDetector::Params params;  
        params.filterByArea = true;
        params.minArea = 300;
        params.maxArea = 1100;  
    
        params.filterByCircularity = true;
        params.minCircularity = 0.9;
        params.maxCircularity = 1.0;

        params.filterByCircularity = true;
        params.minCircularity = 0.5;
        params.maxCircularity = 1.0;

        // Set up detector with params
        Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

        // Detect blobs
        Mat blobL;
        blobL = tempL.clone();
        detector->detect(blobL, keypoints); 
        drawKeypoints(image_left, keypoints, image_left, Scalar(255,0,0)); 

        //bool isFound = findCirclesGrid(image, patternSize, centers, CALIB_CB_ASYMMETRIC_GRID, blobDetector); 
        //bool patternfound = findCirclesGrid(gray, patternsize, centers);

        imshow("result_blobL", image_left);
        for (int i=0; i<keypoints.size(); i++){
            float X = keypoints[i].pt.x; 
            float Y = keypoints[i].pt.y;
            cout << X << "\n" << Y << endl;
            cout << endl;
        }
        cvWaitKey(0);
        
    }


    for (int count = 1; count <= 8; count++)    //need change
	{

		sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/right cam/%d-1.bmp", count);   //need change
		image_right = imread(filename, 1);
    

        resize(image_right, image_right, Size(640,480));
        cvtColor(image_right, tempR, CV_RGB2GRAY);

        equalizeHist(tempR, tempR);
        //imshow("equalizeHist", tempR);

        Mat BlurR = tempR.clone();
        GaussianBlur(tempR, BlurR, Size(11, 11), 1.3, 1.3);
        //imshow("image_right", tempR);
        tempR =  tempR - BlurR;
        //imshow("edge", tempR);

        threshold(tempR, tempR, 5, 255, THRESH_BINARY);
        imshow("BINARY", tempR);

        //Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        //erode(tempR, tempR, kernel);
        //dilate(tempR, tempR, kernel);
        //erode(tempR, tempR, kernel);
        //imshow("BINARY", tempR);

 /*       Mat contoursR = tempR.clone();
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(contoursR, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

        int area = 0;
        for (int i = 0; i < contours.size(); i++)
        {
            Scalar color = Scalar(255,0,0);
            drawContours(contoursR, contours, -1, color, 3, 8, hierarchy, INT_MAX, Point());
            if (area < contourArea(contours[i]))	area = contourArea(contours[i]);
        }
        imshow("contoursR", contoursR);

        //Canny(tempR, tempR, 80, 200, 3);
        //imshow("canny", tempR);
*/

        vector<KeyPoint> keypointsR;  
        SimpleBlobDetector::Params paramsR;  
        paramsR.filterByArea = true;
        paramsR.minArea = 250;
        paramsR.maxArea = 1000;  
    
        paramsR.filterByCircularity = true;
        paramsR.minCircularity = 0.98;
        paramsR.maxCircularity = 1.0;

        paramsR.filterByCircularity = true;
        paramsR.minCircularity = 0.5;
        paramsR.maxCircularity = 1.0;

        // Set up detector with params
        Ptr<SimpleBlobDetector> detectorR = SimpleBlobDetector::create(paramsR);

        // Detect blobs
        Mat blobR;
        blobR = tempR.clone();
        detectorR->detect(blobR, keypointsR); 
        drawKeypoints(image_right, keypointsR, image_right, Scalar(255,0,0)); 

        //bool isFound = findCirclesGrid(image, patternSize, centers, CALIB_CB_ASYMMETRIC_GRID, blobDetector); 
        //bool patternfound = findCirclesGrid(gray, patternsize, centers);

        imshow("result_blobR", image_right);
        for (int i=0; i<keypointsR.size(); i++){
            float X_R = keypointsR[i].pt.x; 
            float Y_R = keypointsR[i].pt.y;
            cout << X_R << "\n" << Y_R << endl;
            cout << endl;
        }
        cvWaitKey(0);
        
    }




   /*
    Mat  circleL = tempL.clone();
    vector<Vec3f> circles;
	//霍夫圆变换找圆
	HoughCircles(circleL, circles, HOUGH_GRADIENT, 1, 80, 100, 40, 1, 40);
	//GaussianBlur(circleL, circleL, Size(9, 9), 2, 2);
	int maxR = 0;
	//逐个画圆
	for (size_t i = 0; i < circles.size(); i++)
	{

		//声明圆心与半径
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		
		//储存最大半径
		if (radius > maxR)	maxR = radius;

		//画圆
		circle(circleL, center, radius, Scalar(255, 0, 0), 3, 8, 0);
	}
    imshow("result_circleL", circleL);
    */

    //cvWaitKey(0);
    return 0;
}