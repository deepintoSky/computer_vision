#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <vector>

using namespace std;
using namespace cv;

//像素坐标转为相机坐标
Point2f pixel2cam(const Point2d& p, const Mat& K)
{
	return Point2f
	(
		(p.x - K.at<double>(0, 2) - K.at<double>(0, 1)*(p.y - K.at<double>(1, 2)) / K.at<double>(1, 1) ) / K.at<double>(0, 0),
		(p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
	);
}

//三角化测量得到世界坐标
void triangulation(
    bool mismatch,
	const vector< KeyPoint >& keypoint_1,
	const vector< KeyPoint >& keypoint_2,
	const Mat& K1, const Mat& K2,
	const Mat& R, const Mat& t,
	vector< Point3d >& points)
{
    //由相机的旋转平移获得投影变换矩阵T1, T2
	Mat T1 = (Mat_<float>(3, 4) <<
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0);
	Mat T2 = (Mat_<float>(3, 4) <<
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
		);

	vector<Point2f> pts_1, pts_2;
	for (int i = 0; i< 2; i++)
	{
		// 像素坐标转为相机坐标
        if(mismatch){//误匹配则把匹配点调换一下
            pts_1.push_back(pixel2cam(keypoint_1[i].pt, K1));
		    pts_2.push_back(pixel2cam(keypoint_2[1 - i].pt, K2));
        }
        else{
            pts_1.push_back(pixel2cam(keypoint_1[i].pt, K1));
		    pts_2.push_back(pixel2cam(keypoint_2[i].pt, K2));
        }	
	}

	Mat pts_4d;
	cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

	// 转为非齐次坐标
	for (int i = 0; i<pts_4d.cols; i++)
	{
		Mat x = pts_4d.col(i);
		x /= x.at<float>(3, 0); // 归一化
		Point3d p(
			x.at<float>(0, 0),
			x.at<float>(1, 0),
			x.at<float>(2, 0)
		);
		points.push_back(p);
	}
}




int main(int argc, char** argv)
{
	Mat image_left, image_right, tempL, tempR, ROI_tempL, ROI_tempR, image_left_visial, image_right_visial;

    //相机标定结果
    Mat cameraMatrix_L = (Mat_<double>(3, 3) << 7464.9352329183,	40.4091065701,	    1818.5330168797,
                                                0,	                7465.6081382331,	1383.8527966054,
                                                0,	                0,	                   1);

    Mat cameraMatrix_R = (Mat_<double>(3, 3) << 7344.34068362847,	38.5756634932,	    1582.6595267309,
                                                0,	                7342.6530413198,	1399.9973531205,
                                                0,	                0,	                1);

    Mat rotation = (Mat_<double>(3, 3) << 0.9992262817,	-0.0083102752,	-0.0384418682,
                                        0.0097792163,	0.999222904,	0.0381831777,
                                        0.0380946825,	-0.038529566,	0.998531055);

    Mat translation = (Mat_<double>(3, 1) << -625.2470674697,	 7.6217840851,	-61.3736062666);
    Mat R1, R2, P1, P2, Q;
    Mat distCoeffs_L = (Mat_<double>(4, 1) << -0.037, -2.023, 0.017, 0.015);
    Mat distCoeffs_R = (Mat_<double>(4, 1) << 0.07, -1.59, 0.015, 0.01);

    //stereoRectify(cameraMatrix_L, distCoeffs_L, cameraMatrix_R, distCoeffs_R, Size(2448, 2048), rotation, translation, R1, R2, P1, P2, Q);
    //cout << P1 << "\n" << P2 << "\n" << endl;






    char  filename[50];

    for (int count = 1; count <= 8; count++)    //need change
	{
        //读入图片并灰度化
		sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/left cam/%d-1.bmp", count); 
		image_left = imread(filename, 1);
        cout << image_left.size() << endl;
        cvtColor(image_left, tempL, CV_RGB2GRAY);

        sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/right cam/%d-1.bmp", count);   //need change
		image_right = imread(filename, 1);   
        cvtColor(image_right, tempR, CV_RGB2GRAY);

        //高斯滤波平滑
        GaussianBlur(tempL, tempL, Size(15, 15), 3, 3);
        GaussianBlur(tempR, tempR, Size(15, 15), 3, 3);

        //二值化使特征突出，去除干扰
        threshold(tempL, tempL, 120, 255, THRESH_BINARY);
        threshold(tempR, tempR, 120, 255, THRESH_BINARY);

        //定义BLOB分析参数
        vector<KeyPoint> keypointsL, keypointsR;  
        SimpleBlobDetector::Params params;

        //二值化阈值与步长
        params.thresholdStep = 10;
        params.maxThreshold = 50;
        params.maxThreshold = 220;
        params.minRepeatability = 2;
        params.minDistBetweenBlobs = 10;

        //面积阈值
        params.filterByArea = true;
        params.minArea = 5000;
        params.maxArea = 20000;  
    
        //圆度阈值
        params.filterByCircularity = true;
        params.minCircularity = 0.8;
        params.maxCircularity = 1.0;

        //凸度阈值
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

        bool mismatch = false;
        for (int i=0; i<keypointsL.size(); i++){//打印特征点并画圆
            float X = keypointsL[i].pt.x; 
            float Y = keypointsL[i].pt.y;
            cout << Mat(Point(X, Y)) << endl;
            cout << endl;
            //circle(image_left, Point(X, Y), 30, cv::Scalar(255, 0, 0), 3);

            
            float X_R = keypointsR[i].pt.x; 
            float Y_R = keypointsR[i].pt.y;
            cout << Mat(Point(X_R, Y_R)) << endl;
            cout << endl;
            //circle(image_right, Point(X_R, Y_R), 30, cv::Scalar(255, 0, 0), 3);

            if(abs(X - X_R) >= 1000 ) mismatch = true;//若像素值差别太大则误匹配
            else mismatch = false;
            cout << endl;

            //寻找像素几何中心
            ROI_tempL = image_left(Rect(X-75, Y-80, 150, 160));
            cvtColor(ROI_tempL, ROI_tempL, CV_RGB2GRAY);
            
            //SimpleBlobDetector::Params params_blob;
            //params_blob.minArea = 10e3;
            //Ptr<SimpleBlobDetector> detector_blob = SimpleBlobDetector::create(params_blob);

            //vector<Point2f> centerL;
            //findCirclesGrid(ROI_tempL, Size(1, 1), centerL, CALIB_CB_ASYMMETRIC_GRID, detector_blob);
            //cout << centerL << endl;
            //cout << endl;
            imshow("ROI_tempL", ROI_tempL);


            ROI_tempR = image_right(Rect(X_R-75, Y_R-80, 150, 160));
            cvtColor(ROI_tempL, ROI_tempL, CV_RGB2GRAY);
            imshow("ROI_tempR", ROI_tempR);
            //vector<Point2f> centerR;
            //findCirclesGrid(ROI_tempR, Size(1, 1), centerR);
            //cout << centerR << endl;
            //cout << endl;

            

        }
        //三角化获得世界坐标
        vector< Point3d > worldPointEst;
        triangulation(mismatch, keypointsL, keypointsR, cameraMatrix_L, cameraMatrix_R, rotation.inv(), translation, worldPointEst);
        cout << worldPointEst << "\n" << endl;

        //球门横向距离计算
        float d_x = pow( (worldPointEst[0].x - worldPointEst[1].x), 2);
        float d_y = pow( (worldPointEst[0].y - worldPointEst[1].y), 2);
        float d_z = pow( (worldPointEst[0].z - worldPointEst[1].z), 2);
        float d = sqrt( d_x + d_y + d_z );
        cout << "d = " << d << "\n" << endl;

        //图像尺寸缩小副本用于可视化
        resize(image_left, image_left_visial, Size(612, 512)); 
        imshow("result_blobL", image_left_visial);

		resize(image_right, image_right_visial, Size(612, 512)); 
        imshow("result_blobR", image_right_visial);
        cvWaitKey(0);
    }

    cvWaitKey(0);

    return 0;    
}