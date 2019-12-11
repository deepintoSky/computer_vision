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

    char  filename[50];

    for (int count = 1; count <= 8; count++)    //need change
	{
		sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/left cam/%d-1.bmp", count); 
		image_left = imread(filename, 1);
        cout << image_left.size() << endl;
        resize(image_left, image_left, Size(612,512));
        cvtColor(image_left, tempL, CV_RGB2GRAY);

        GaussianBlur(tempL, tempL, Size(5, 5), 1.5, 1.5);
        threshold(tempL, tempL, 100, 255, THRESH_BINARY);

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
        //resize(image_left, image_left, Size(2448, 2048)); 
        //imshow("result_blobL", image_left);
        for (int i=0; i<keypoints.size(); i++){
            float X = keypoints[i].pt.x; 
            float Y = keypoints[i].pt.y;
            cout << X << "\n" << Y << endl;
            cout << endl;
        }

		sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/right cam/%d-1.bmp", count);   //need change
		image_right = imread(filename, 1);
    

        resize(image_right, image_right, Size(612, 512));
        cvtColor(image_right, tempR, CV_RGB2GRAY);

        equalizeHist(tempR, tempR);
        //imshow("equalizeHist", tempR);

        Mat BlurR = tempR.clone();
        GaussianBlur(tempR, BlurR, Size(9, 9), 1.5, 1.5);
        //imshow("image_right", tempR);
        tempR =  tempR - BlurR;
        //imshow("edge", tempR);
        threshold(tempR, tempR, 8, 255, THRESH_BINARY);
        imshow("BINARY", tempR);


        vector<KeyPoint> keypointsR;  
        SimpleBlobDetector::Params paramsR;  
        paramsR.filterByArea = true;
        paramsR.minArea = 300;
        paramsR.maxArea = 1000;  
    
        paramsR.filterByCircularity = true;
        paramsR.minCircularity = 0.95;
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
        //resize(image_left, image_left, Size(2448, 2048)); 
        imshow("result_blobR", image_right);
        for (int i=0; i<keypointsR.size(); i++){
            float X_R = keypointsR[i].pt.x; 
            float Y_R = keypointsR[i].pt.y;
            cout << X_R << "\n" << Y_R << endl;
            cout << endl;
        }
       
    


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

        vector< Point3d > worldPointEst;
       for(int i =0; i<keypoints.size(); i++)
        {
            //triangulatePoints(P1, P2, Mat(keypoints), Mat(keypointsR), worldPointEst);
        }
/* 
        for(int i=0; i<keypointsR.size(); i++){
                float X_G = worldPointEst[i].x; 
                float Y_G = worldPointEst[i].y;
                float Z_G = worldPointEst[i].z;
                cout << X_G << "\n" << Y_G << "\n" << Z_G << endl;
                cout << endl;
        }
*/
         cvWaitKey(0);
    }


    cvWaitKey(0);

    return 0;    
}