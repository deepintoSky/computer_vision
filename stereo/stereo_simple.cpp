#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <vector>

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
	Mat image_left, image_right, tempL, tempR, image_left_visial, image_right_visial;

    Mat cameraMatrix_L = (Mat_<double>(3, 3) << 7491.20599927464,	0,	0,
                        65.8894515063429,	7491.97168906742,	0,
                        1942.52279709139,	1497.45072963980,	1);

    Mat cameraMatrix_R = (Mat_<double>(3, 3) << 7351.97823087610,	0,	0,
                        53.1795796695527,	7352.02889760308,	0,
                        1637.89205830372,	1474.16645815975,	1);

    Mat rotation = (Mat_<double>(3, 3) << 0.998842, -0.0078428, -0.0474674,
                        0.00941882, 0.99940863, 0.03307073,
                        0.04717992, -0.0334795, 0.99832519);

    Mat translation = (Mat_<double>(3, 1) << -624.71864, 7.77449929, -68.122574);
    Mat R1, R2, P1, P2, Q;
    Mat distCoeffs_L = (Mat_<double>(4, 1) << 0.196, -13, 0.02, 0.02);
    Mat distCoeffs_R = (Mat_<double>(4, 1) << 0.3, -8.59, 0.02, 0.02);

    stereoRectify(cameraMatrix_L, distCoeffs_L, cameraMatrix_R, distCoeffs_R, Size(2448, 2048), rotation, translation, R1, R2, P1, P2, Q);

    cout << P1 << "\n" << P2 << "\n" << endl;






    char  filename[50];

    for (int count = 1; count <= 8; count++)    //need change
	{
		sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/left cam/%d-1.bmp", count); 
		image_left = imread(filename, 1);
        cout << image_left.size() << endl;
        cvtColor(image_left, tempL, CV_RGB2GRAY);

        sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/right cam/%d-1.bmp", count);   //need change
		image_right = imread(filename, 1);   
        cvtColor(image_right, tempR, CV_RGB2GRAY);

        GaussianBlur(tempL, tempL, Size(15, 15), 3, 3);
        GaussianBlur(tempR, tempR, Size(15, 15), 3, 3);

        threshold(tempL, tempL, 120, 255, THRESH_BINARY);
        threshold(tempR, tempR, 120, 255, THRESH_BINARY);

        vector<KeyPoint> keypointsL, keypointsR;  
        SimpleBlobDetector::Params params;

        params.thresholdStep = 10;
        params.maxThreshold = 50;
        params.maxThreshold = 220;
        params.minRepeatability = 2;
        params.minDistBetweenBlobs = 10;

        params.filterByArea = true;
        params.minArea = 5000;
        params.maxArea = 20000;  
    
        params.filterByCircularity = true;
        params.minCircularity = 0.8;
        params.maxCircularity = 1.0;

        params.filterByConvexity = true;
        params.minConvexity = 0.5;
        params.maxConvexity = 1.0;

        // Set up detector with params
        Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

        // Detect blobs
        Mat blobL = tempL.clone();
        detector->detect(blobL, keypointsL); 
        drawKeypoints(blobL, keypointsL, blobL, Scalar(255,0,0));
        // Detect blobs
        Mat blobR = tempR.clone();
        detector->detect(blobR, keypointsR); 
        drawKeypoints(blobR, keypointsR, blobR, Scalar(255,0,0)); 

        Mat worldPointEst;
        for (int i=0; i<keypointsL.size(); i++){
            float X = keypointsL[i].pt.x; 
            float Y = keypointsL[i].pt.y;
            cout << X << "\n" << Y << endl;
            cout << endl;
            circle(blobL, Point(X, Y), 40, cv::Scalar(255, 0, 0), 3);

            float X_R = keypointsR[i].pt.x; 
            float Y_R = keypointsR[i].pt.y;
            cout << X_R << "\n" << Y_R << endl;
            cout << endl;
            circle(blobR, Point(X_R, Y_R), 40, cv::Scalar(255, 0, 0), 3);

            triangulatePoints(P1, P2, Mat(Point(X, Y)), Mat(Point(X_R, Y_R)), worldPointEst);
            //float X_G = worldPointEst; 
            //float Y_G = worldPointEst.y;
            //float Z_G = worldPointEst.z;
            cout << worldPointEst << endl;
            cout << endl;

        }

        resize(blobL, image_left_visial, Size(612, 512)); 
        imshow("result_blobL", image_left_visial);

		resize(blobR, image_right_visial, Size(612, 512)); 
        imshow("result_blobR", image_right_visial);
        cvWaitKey(0);
    }

    cvWaitKey(0);

    return 0;    
}