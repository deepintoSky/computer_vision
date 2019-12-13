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
		(p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
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

void Blob_analysis( const Mat& tempL, const Mat& tempR, vector<KeyPoint>& keypointsL, vector<KeyPoint>& keypointsR){

        SimpleBlobDetector::Params params;

        //二值化阈值与步长
        params.thresholdStep = 10;
        params.maxThreshold = 50;
        params.maxThreshold = 220;
        params.minRepeatability = 2;
        params.minDistBetweenBlobs = 10;

        //面积阈值
        params.filterByArea = true;
        params.minArea = 4000;
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

}

void Pose_analysis( const vector< Point3d >& worldPointEst){
    
    //球门法向量
    float n_x = worldPointEst[0].x - worldPointEst[1].x;
    float n_y = worldPointEst[0].y - worldPointEst[1].y;
    float n_z = worldPointEst[0].z - worldPointEst[1].z;
    Point3d n(n_x, n_y, n_z);
    Point3d ny(0, 1, 0);
    Point3d n_goal = n.cross(ny);
    cout << "normal vetor of goal = " << n_goal << "\n" << endl;
    //球门姿态角
    float theta = atan(n_goal.x / n_goal.z);
    cout << "theta of goal = " << theta*180/3.14159 << "\n" << endl;


    //球门横向距离计算
    float d_x = pow( n_x, 2);
    //float d_y = pow( n_y, 2);
    float d_y = 0;
    float d_z = pow( n_z, 2);
    float d = sqrt( d_x + d_y + d_z );
    cout << "distance between A and B = " << d << "\n" << endl;

    //球门横梁中心与相机坐标系距离计算
    float g_x = pow( (worldPointEst[0].x + worldPointEst[1].x)/2, 2);
    float g_y = pow( (worldPointEst[0].y + worldPointEst[1].y)/2, 2);
    float g_z = pow( (worldPointEst[0].z + worldPointEst[1].z)/2, 2);
    float dg = sqrt( g_x + g_y + g_z );
    cout << "distance between goal and camera = " << dg << "\n" << endl;

}



int main(int argc, char** argv)
{
	Mat image_left, image_right, tempL, tempR, image_left_visial, image_right_visial;

/*
    //相机标定结果
    Mat cameraMatrix_L = (Mat_<double>(3, 3) << 7439.1449067652,	0,	                0,
                                                0,	                7437.6255516057,	0,
                                                1405.886966076,	    937.3362811242,	    1);

    Mat cameraMatrix_R = (Mat_<double>(3, 3) << 7342.704388258,	    0,	                0,
                                                0,	                7343.6607746441,	0,
                                                1372.6677103634,	1081.5986501337,	1);

    Mat rotation = (Mat_<double>(3, 3) << 0.9998598444,	-0.0124996949,	-0.0111377335,
                                        0.0130880528,	0.998432338,	0.054420302,
                                        0.0104400362,	-0.0545584459,	0.9984559988);

    Mat translation = (Mat_<double>(3, 1) << -626.5672011742,	7.0094879625,	-41.344391405);
*/

    //新相机标定结果3coeff 剔除误差
    Mat cameraMatrix_L = (Mat_<double>(3, 3) << 7841.0384836509,	0,	0,
                                                0,	7812.0266528615,	0,
                                                1294.0280524123,	1181.0841489902,	1);

    Mat cameraMatrix_R = (Mat_<double>(3, 3) << 7763.8253620761,	0,	0,
                                                0,	7721.7607676568,	0,
                                                1154.177708236,	945.6251049212,	1);

    Mat rotation = (Mat_<double>(3, 3) << 0.999634448,	-0.0129989519,	-0.0237064908,
                                            0.013086313,	0.9999081265,	0.0035337015,
                                            0.0236583784,	-0.0038426403,	0.9997127164);

    Mat translation = (Mat_<double>(3, 1) << -622.7266416748,	2.0965186942,	3.1009756712);




    char  filename[50];

    for (int count = 1; count <= 11; count++)    //need change
	{
        //读入图片并灰度化
        sprintf(filename, "/home/jack/Desktop/cv_project/1213/left/%d-1.bmp", count); 
		//sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/left cam/%d-1.bmp", count); 
		image_left = imread(filename, 1);
        cout << image_left.size() << endl;
        cvtColor(image_left, tempL, CV_RGB2GRAY);

        sprintf(filename, "/home/jack/Desktop/cv_project/1213/right/%d-1.bmp", count); 
        //sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/right cam/%d-1.bmp", count);   //need change
		image_right = imread(filename, 1);   
        cvtColor(image_right, tempR, CV_RGB2GRAY);

        //equalizeHist(tempL, tempL);
        //equalizeHist(tempR, tempR);

        //CANNY边缘检测
        Canny(tempL, tempL, 30, 70);
        Canny(tempR, tempR, 30, 70);
        resize(tempL, image_left_visial, Size(612, 512));
        resize(tempR, image_right_visial, Size(612, 512));
        imshow("edgeL", image_left_visial);
        imshow("edgeR", image_right_visial);

        //sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/result/left/canny/edgeL_%d.png", count);
        //imwrite(filename, image_left_visial);
        //sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/result/right/canny/edgeR_%d.png", count);
        //imwrite(filename, image_right_visial);

        //二值化使特征突出，去除干扰
        //threshold(tempL, tempL, 5, 255, THRESH_BINARY);
        //threshold(tempR, tempR, 5, 255, THRESH_BINARY);
        //resize(tempL, image_left_visial, Size(612, 512));
        //resize(tempR, image_right_visial, Size(612, 512));
        //imshow("BINARYL", image_left_visial);
        //imshow("BINARYR", image_right_visial);

        //定义BLOB分析参数
        vector<KeyPoint> keypointsL, keypointsR;  
        Blob_analysis(tempL, tempR, keypointsL, keypointsR);

        bool mismatch = false;
        for (int i=0; i<keypointsL.size(); i++){//打印特征点并画圆
            float X = keypointsL[i].pt.x; 
            float Y = keypointsL[i].pt.y;
            cout << Mat(Point(X, Y)) << endl;
            cout << endl;
            circle(image_left, Point(X, Y), 40, cv::Scalar(255, 0, 0), 3);

            float X_R = keypointsR[i].pt.x; 
            float Y_R = keypointsR[i].pt.y;
            cout << Mat(Point(X_R, Y_R)) << endl;
            cout << endl;
            circle(image_right, Point(X_R, Y_R), 40, cv::Scalar(255, 0, 0), 3);

            if(abs(X - X_R) >= 1000 ) mismatch = true;//若像素值差别太大则误匹配
            else mismatch = false;
            
            cout << endl;

        }

        //图像尺寸缩小副本用于可视化
        resize(image_left, image_left_visial, Size(612, 512)); 
        imshow("result_blobL", image_left_visial);

		resize(image_right, image_right_visial, Size(612, 512)); 
        imshow("result_blobR", image_right_visial);


        //三角化获得世界坐标
        vector< Point3d > worldPointEst;
        triangulation(mismatch, keypointsL, keypointsR, cameraMatrix_L.t(), cameraMatrix_R.t(), rotation.inv(), -translation, worldPointEst);
        cout << worldPointEst << "\n" << endl;

        Pose_analysis(worldPointEst);
        
        cvWaitKey(0);
    }

    cvWaitKey(0);

    return 0;    
}