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
    double n_x = worldPointEst[0].x - worldPointEst[1].x;
    double n_y = worldPointEst[0].y - worldPointEst[1].y;
    double n_z = worldPointEst[0].z - worldPointEst[1].z;
    Point3d n(n_x, n_y, n_z);
    Point3d ny(0, 1, 0);
    Point3d n_goal = n.cross(ny);
    cout << "normal vetor of goal = " << n_goal << "\n" << endl;
    //球门姿态角
    double theta = atan(n_goal.x / n_goal.z);
    cout << "theta of goal = " << theta*180/3.14159 << "\n" << endl;


    //球门横向距离计算
    double d_x = pow( n_x, 2);
    //double d_y = pow( n_y, 2);
    double d_y = 0;
    double d_z = pow( n_z, 2);
    double d = sqrt( d_x + d_y + d_z );
    cout << "distance between A and B = " << d << "\n" << endl;

    //球门横梁中心与相机坐标系距离计算
    double g_x = pow( (worldPointEst[0].x + worldPointEst[1].x)/2, 2);
    double g_y = pow( (worldPointEst[0].y + worldPointEst[1].y)/2, 2);
    double g_z = pow( (worldPointEst[0].z + worldPointEst[1].z)/2, 2);
    double dg = sqrt( g_x + g_y + g_z );
    cout << "distance between goal and camera = " << dg << "\n" << endl;

}


void ellipse_fitting(
    const Mat& image_left, const Mat& image_right,
    const int& count,
    double& X,  double& Y,
    double& X_R, double& Y_R)
{

    //寻找像素几何中心
    vector< vector<Point> > contoursL, contoursR;
    Mat ROI_tempL, ROI_tempR;
    double dX, dY, dX_R, dY_R;

    ROI_tempL = image_left(Rect(X-75, Y-80, 150, 160));
    cvtColor(ROI_tempL, ROI_tempL, CV_RGB2GRAY);
    Canny(ROI_tempL, ROI_tempL, 30, 70);
    //imshow("canny边缘L", ROI_tempL);



    ROI_tempR = image_right(Rect(X_R-75, Y_R-80, 150, 160));
    cvtColor(ROI_tempR, ROI_tempR, CV_RGB2GRAY);;
    Canny(ROI_tempR, ROI_tempR, 30, 70);
    //imshow("canny边缘R", ROI_tempR);


    //************************************椭圆拟合
	findContours(ROI_tempL, contoursL, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	Mat cimageL = Mat::zeros(ROI_tempL.size(), CV_8UC3);

	//DoG和椭圆拟合公共部分——椭圆检测
	for (size_t i = 0; i < contoursL.size(); i++)
	{
				
		//对拟合轮廓点进行筛选，椭圆检测至少6个点
		if (contoursL[i].size() < 6)    continue;

		//对轮廓面积进行筛选
		if (contourArea(contoursL[i]) < 10|| contourArea(contoursL[i]) > 140*170)   continue;

		//椭圆拟合
		RotatedRect boxL = fitEllipse(contoursL[i]);

		//如果椭圆不在中心点附近，则排除
		if (boxL.center.x >80 || boxL.center.x < 70||boxL.center.y >85|| boxL.center.y<75)  continue;

		//画出追踪出的轮廓
		drawContours(ROI_tempL, contoursL, (int)i, Scalar::all(255), 1, 8);

		//画出拟合的椭圆
		ellipse(ROI_tempL, boxL, Scalar(100, 0, 0), 1, CV_AA);
		//cout << boxL.center.x<< endl;
		//cout << boxL.center.y<< endl;
        dX = boxL.center.x;
        dY = boxL.center.y;
	}

	imshow("left_ellipse", ROI_tempL);
    //sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/result/left/ellipse_roi/left_ellipse_%d.png", count);
    //imwrite(filename, cimageL);

    //椭圆拟合
	findContours(ROI_tempR, contoursR, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	Mat cimageR = Mat::zeros(ROI_tempR.size(), CV_8UC3);

	//DoG和椭圆拟合公共部分——椭圆检测
	for (size_t i = 0; i < contoursR.size(); i++)
	{
				
		//对拟合轮廓点进行筛选，椭圆检测至少6个点
		if (contoursR[i].size() < 6)    continue;

		//对轮廓面积进行筛选
		if (contourArea(contoursR[i]) < 10|| contourArea(contoursR[i]) > 140*170)   continue;

		//椭圆拟合
		RotatedRect boxR = fitEllipse(contoursR[i]);

		//如果椭圆不在中心点附近，则排除
		if (boxR.center.x >80 || boxR.center.x < 70||boxR.center.y >85|| boxR.center.y<75)  continue;

		//画出追踪出的轮廓
		drawContours(ROI_tempR, contoursR, (int)i, Scalar::all(255), 1, 8);

		//画出拟合的椭圆
		ellipse(ROI_tempR, boxR, Scalar(100, 0, 0), 5, CV_AA);
		//cout << boxR.center.x<< endl;
		//cout << boxR.center.y<< endl;
        dX_R = boxR.center.x;
        dY_R = boxR.center.y;


	}

	imshow("right_ellipse", ROI_tempR);
    //sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/result/right/ellipse_roi/right_ellipse_%d.png", count);
    //imwrite(filename, cimageR);

    //坐标矫正
    X = X -75 + dX;
    X_R = X_R - 75 + dX_R;
    Y = Y - 80 + dY;
    Y_R = Y_R - 80 + dY_R;
          
}


void pixel_average(
    const Mat& image_left, const Mat& image_right,
    const int& count,
    double& X,  double& Y,
    double& X_R, double& Y_R)
{

    //寻找像素几何中心
    vector< vector<Point> > contoursL, contoursR;
    vector<Point> marker_contoursL, marker_contoursR;
    Mat ROI_tempL, ROI_tempR;
    double dX, dY, dX_R, dY_R, width, height, width_R, height_R;

    ROI_tempL = image_left(Rect(X-75, Y-80, 150, 160));
    cvtColor(ROI_tempL, ROI_tempL, CV_RGB2GRAY);
    Canny(ROI_tempL, ROI_tempL, 30, 70);
   

    ROI_tempR = image_right(Rect(X_R-75, Y_R-80, 150, 160));
    cvtColor(ROI_tempR, ROI_tempR, CV_RGB2GRAY);
    Canny(ROI_tempR, ROI_tempR, 30, 70);


    //************************************椭圆拟合
	findContours(ROI_tempL, contoursL, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	Mat cimageL = Mat::zeros(ROI_tempL.size(), CV_8UC3);
	//DoG和椭圆拟合公共部分——椭圆检测
	for (size_t i = 0; i < contoursL.size(); i++)
	{
				
		//对拟合轮廓点进行筛选，椭圆检测至少6个点
		if (contoursL[i].size() < 6)    continue;

		//对轮廓面积进行筛选
		if (contourArea(contoursL[i]) < 10|| contourArea(contoursL[i]) > 140*170)   continue;

		//椭圆拟合
		RotatedRect boxL = fitEllipse(contoursL[i]);

		//如果椭圆不在中心点附近，则排除
		if (boxL.center.x >80 || boxL.center.x < 70||boxL.center.y >85|| boxL.center.y<75)  continue;

		//画出追踪出的轮廓
		drawContours(ROI_tempL, contoursL, (int)i, Scalar::all(255), 1, 8);

		//画出拟合的椭圆
		ellipse(ROI_tempL, boxL, Scalar(100, 0, 0), 5, CV_AA);
		//cout << boxL.center.x<< endl;
		//cout << boxL.center.y<< endl;
        //dX = boxL.center.x;
        //dY = boxL.center.y;

        //像素坐标平均准备
        width = boxL.size.width+10;
        height = boxL.size.height+10;
        ROI_tempL = image_left(Rect(X - 0.5*width, Y-0.5*height, width, height));
        cvtColor(ROI_tempL, ROI_tempL, CV_RGB2GRAY);
        //threshold(ROI_tempL, ROI_tempL, 100, 255, THRESH_BINARY);
	}

	imshow("left_ellipse", ROI_tempL);
    //sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/result/left/ellipse_roi/left_ellipse_%d.png", count);
    //imwrite(filename, cimageL);

    //椭圆拟合
	findContours(ROI_tempR, contoursR, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	Mat cimageR = Mat::zeros(ROI_tempR.size(), CV_8UC3);

	//DoG和椭圆拟合公共部分——椭圆检测
	for (size_t i = 0; i < contoursR.size(); i++)
	{
				
		//对拟合轮廓点进行筛选，椭圆检测至少6个点
		if (contoursR[i].size() < 6)    continue;

		//对轮廓面积进行筛选
		if (contourArea(contoursR[i]) < 10|| contourArea(contoursR[i]) > 140*170)   continue;

		//椭圆拟合
		RotatedRect boxR = fitEllipse(contoursR[i]);

		//如果椭圆不在中心点附近，则排除
		if (boxR.center.x >80 || boxR.center.x < 70||boxR.center.y >85|| boxR.center.y<75)  continue;

		//画出追踪出的轮廓
		drawContours(ROI_tempR, contoursR, (int)i, Scalar::all(255), 1, 8);

		//画出拟合的椭圆
		ellipse(ROI_tempR, boxR, Scalar(100, 0, 0), 5, CV_AA);
		//cout << boxR.center.x<< endl;
		//cout << boxR.center.y<< endl;
        //dX_R = boxR.center.x;
        //dY_R = boxR.center.y;

        //像素坐标平均准备
        width_R = boxR.size.width+10;
        height_R = boxR.size.height+10;
        ROI_tempR = image_right(Rect(X_R - 0.5*width_R, Y_R-0.5*height_R, width_R, height_R));
        cvtColor(ROI_tempR, ROI_tempR, CV_RGB2GRAY);
        //threshold(ROI_tempR, ROI_tempR, 100, 255, THRESH_BINARY);
	}

	imshow("right_ellipse", ROI_tempR);
    //sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/result/right/ellipse_roi/right_ellipse_%d.png", count);
    //imwrite(filename, cimageR);

    //******************************************/

    //像素坐标平均
    double sum_widthL = 0;
    double sum_heightL = 0;
    double number = 0;
    for (int i = 0; i < ROI_tempL.size().width; i++)
	{
			uchar* pdata = ROI_tempL.ptr<uchar>(i);
			for (int j = 0; j < ROI_tempL.size().height; j++)
			{
				if( pdata[j] <= 100){
                    sum_widthL += i;
                    sum_heightL += j;
                    number++;
                }
			}
	}
    dX = sum_widthL / number;
    dY = sum_heightL / number;
    cout << dX << endl;
    cout << dY << endl;

    double sum_widthR = 0;
    double sum_heightR = 0;
    double numberR = 0;
    for (int i = 0; i < ROI_tempR.size().width; i++)
	{
			uchar* pdata = ROI_tempR.ptr<uchar>(i);
			for (int j = 0; j < ROI_tempR.size().height; j++)
			{
				if( pdata[j] <= 100){
                    sum_widthR += i;
                    sum_heightR += j;
                    numberR++;
                }
			}
	}
    dX_R = sum_widthR / numberR;
    dY_R = sum_heightR / numberR;
    cout << dX_R << endl;
    cout << dY_R << endl;

    //坐标矫正
    X = X -0.5*width + dX;
    X_R = X_R - 0.5*width_R + dX_R;
    Y = Y - 0.5*height + dY;
    Y_R = Y_R - 0.5*height_R + dY_R;

}

int main(int argc, char** argv)
{
	Mat image_left, image_right, tempL, tempR, image_left_visial, image_right_visial;


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

     //新相机标定结果2coeff 剔除误差
    Mat rotation11 = (Mat_<double>(3, 3) << 0.999585426780172, -0.012918895796858, -0.025730851134213,
		0.013320644153222, 0.999791075039552, 0.0155037644009375,
		0.0255251838004028, -0.0158400884671288, 0.999548676448180);

	Mat translation11 = (Mat_<double>(3, 1) << -626.282183743998, 2.75695252294966, 11.6702170037552);

	Mat cameraMatrix_L11 = (Mat_<double>(3, 3) << 7.659415501692802e+03, 0, 1.247834598440359e+03, 0, 7.657339635627016e+03, 1.232085832287226e+03, 0, 0, 1);
	Mat cameraMatrix_R11 = (Mat_<double>(3, 3) << 7.594464903367622e+03, 0, 1.101734656332642e+03, 0, 7.579663788497434e+03, 1.086257026142670e+03, 0, 0, 1);


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

 
//******************处理一**************************
        //处理一：CANNY边缘检测
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
//**************************************/

/*****************处理二*********************
        //处理二：高斯滤波平滑
        GaussianBlur(tempL, tempL, Size(15, 15), 3, 3);
        GaussianBlur(tempR, tempR, Size(15, 15), 3, 3);
        resize(tempL, image_left_visial, Size(612, 512));
        resize(tempR, image_right_visial, Size(612, 512));
        imshow("BlurL", image_left_visial);
        imshow("BlurR", image_right_visial);

        //sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/result/left/Gaussian/BlurL_%d.png", count);
        //imwrite(filename, image_left_visial);
        //sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/result/right/Gaussian/BlurR_%d.png", count);
        //imwrite(filename, image_right_visial);

        //处理二：二值化使特征突出，去除干扰
        threshold(tempL, tempL, 120, 255, THRESH_BINARY);
        threshold(tempR, tempR, 120, 255, THRESH_BINARY);
        resize(tempL, image_left_visial, Size(612, 512));
        resize(tempR, image_right_visial, Size(612, 512));
        imshow("BINARYL", image_left_visial);
        imshow("BINARYR", image_right_visial);

        //sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/result/left/binary/BINARYL_%d.png", count);
        //imwrite(filename, image_left_visial);
        //sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/result/right/binary/BINARYR_%d.png", count);
        //imwrite(filename, image_right_visial);
//********************************************/

        //定义BLOB分析参数
        vector<KeyPoint> keypointsL, keypointsR;  
        Blob_analysis(tempL, tempR, keypointsL, keypointsR);

        bool mismatch = false;
        for (int i=0; i<keypointsL.size(); i++){//打印特征点并画圆
            double X = keypointsL[i].pt.x; 
            double Y = keypointsL[i].pt.y;
            cout << "x, y is " << X << ", " <<  Y << endl;
            cout << endl;
            //circle(image_left, Point(X, Y), 40, cv::Scalar(255, 0, 0), 3);

            double X_R = keypointsR[i].pt.x; 
            double Y_R = keypointsR[i].pt.y;
            cout << "xR, yR is "  <<  X_R << ", " <<  Y_R  << endl;
            cout << endl;
            //circle(image_right, Point(X_R, Y_R), 40, cv::Scalar(255, 0, 0), 3);

            if(abs(X - X_R) >= 1000 ) mismatch = true;//若像素值差别太大则误匹配
            else mismatch = false;
            

            //ellipse_fitting(image_left, image_right, count, X, Y, X_R, Y_R);
            //pixel_average(image_left, image_right, count, X, Y, X_R, Y_R);
            cout << "real x, y is " << X << ", " <<  Y << endl;
            cout << "real xR, yR is "  <<  X_R << ", " <<  Y_R  << endl;
            cout << endl;

            keypointsL[i].pt.x = X;
            keypointsL[i].pt.y = Y; 
            keypointsR[i].pt.x = X_R;
            keypointsR[i].pt.y = Y_R;

        }

        //图像尺寸缩小副本用于可视化
        resize(image_left, image_left_visial, Size(612, 512)); 
        imshow("result_blobL", image_left_visial);

		resize(image_right, image_right_visial, Size(612, 512)); 
        imshow("result_blobR", image_right_visial);


        //三角化获得世界坐标
        vector< Point3d > worldPointEst;
        if(count == 11) triangulation(mismatch, keypointsL, keypointsR, cameraMatrix_L11, cameraMatrix_R11, rotation11.inv(), translation11, worldPointEst);
        else triangulation(mismatch, keypointsL, keypointsR, cameraMatrix_L.t(), cameraMatrix_R.t(), rotation.inv(), translation, worldPointEst);
        cout << "world x,y,z of B is "  << worldPointEst[0] << "\n" << endl;
        cout << "world x,y,z of A is "  << worldPointEst[1] << "\n" << endl;

        Pose_analysis(worldPointEst);
        
        cvWaitKey(0);
    }

    cvWaitKey(0);

    return 0;    
}