#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <vector>

using namespace std;
using namespace cv;


char  filename[50];


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


void findRealcenters(
    const Mat& image_left, const Mat& image_right,
    const int& count,
    double& X,  double& Y,
    double& X_R, double& Y_R)
{

    //寻找像素几何中心
    vector< vector<Point> > contoursL, contoursR;
    vector<Point> marker_contoursL, marker_contoursR;
    Mat ROI_tempL, ROI_tempR;
    double dX, dY, dX_R, dY_R;

    ROI_tempL = image_left(Rect(X-75, Y-80, 150, 160));
    cvtColor(ROI_tempL, ROI_tempL, CV_RGB2GRAY);
    //threshold(ROI_tempL, ROI_tempL, 120, 255, THRESH_BINARY);
    Canny(ROI_tempL, ROI_tempL, 30, 70);
    //imshow("canny边缘L", ROI_tempL);

    ROI_tempR = image_right(Rect(X_R-75, Y_R-80, 150, 160));
    cvtColor(ROI_tempR, ROI_tempR, CV_RGB2GRAY);
    //threshold(ROI_tempR, ROI_tempR, 120, 255, THRESH_BINARY);
    Canny(ROI_tempR, ROI_tempR, 30, 70);
    //imshow("canny边缘R", ROI_tempR);

    //椭圆拟合
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
		drawContours(cimageL, contoursL, (int)i, Scalar::all(255), 1, 8);

		//画出拟合的椭圆
		ellipse(cimageL, boxL, Scalar(0, 0, 255), 1, CV_AA);
		//cout << boxL.center.x<< endl;
		//cout << boxL.center.y<< endl;
        dX = boxL.center.x;
        dY = boxL.center.y;
	}

	imshow("left_ellipse", cimageL);
    sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/result/left/ellipse_roi/left_ellipse_%d.png", count);
    imwrite(filename, cimageL);

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
		drawContours(cimageR, contoursR, (int)i, Scalar::all(255), 1, 8);

		//画出拟合的椭圆
		ellipse(cimageR, boxR, Scalar(0, 0, 255), 1, CV_AA);
		//cout << boxR.center.x<< endl;
		//cout << boxR.center.y<< endl;
        dX_R = boxR.center.x;
        dY_R = boxR.center.y;
	}

	imshow("right_ellipse", cimageR);
    sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/result/right/ellipse_roi/right_ellipse_%d.png", count);
    imwrite(filename, cimageR);

/******************findCirclesGrid测试代码
            vector<Point2f> centersL, centersR; //this will be filled by the detected centers
            Mat patttern = imread("/home/jack/Desktop/C++/computer_vision/stereo/acircles_pattern.png", 1);
            bool isfound = findCirclesGrid(patttern, Size(4, 6), centersL, CALIB_CB_SYMMETRIC_GRID);
            cout << isfound << endl;
            drawChessboardCorners(patttern, Size(4, 6), Mat(centersL), isfound);
            bool flag = findChessboardCorners(patttern, Size(4, 6), centersL, CV_CALIB_CB_ADAPTIVE_THRESH);
	        drawChessboardCorners(patttern, Size(4, 6), centersL, flag);
            cout << flag << endl;
            resize(patttern, patttern, Size(612, 512)); 
            imshow("ROI_tempL", patttern);
            waitKey(0);
******************************************/

/******************findContours测试代码
            vector<Vec4i>  hierarchy;
            findContours(ROI_tempL, contoursL, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);//找轮廓
            Mat imageContoursL=Mat::zeros(ROI_tempL.size(),CV_8UC1);
            for(int i=0;i<contoursL.size();i++)
	        {
                drawContours(imageContoursL, contoursL, i, Scalar(255),1,8,hierarchy);//遍历每一个轮廓，并且绘制轮廓。
                double area = contourArea(contoursL.at(i));
                cout << "area is " << area << endl;
                if (area > 4900) marker_contoursL = contoursL.at(i);
            }
            //计算矩来计算中心
            //Moments mu = moments(marker_contoursL, false);
            //float dX = mu.m10/mu.m00;
            //float dY = mu.m01/mu.m00;
            //cout << "x, y is " << Mat(Point(dX, dY)) << endl;
            //椭圆拟合算中心
            RotatedRect ellipseL = fitEllipse(marker_contoursL);
            Mat image_ellipseL = Mat::zeros(ROI_tempL.size(), CV_8UC3);
            ellipse(image_ellipseL, ellipseL, Scalar(255),1, CV_AA);
            float dX = ellipseL.center.x;
            float dY = ellipseL.center.y;
            cout << "x, y is " << Mat(Point(dX, dY)) << endl;
            imshow("ROI_tempL", image_ellipseL);
            findContours(ROI_tempR, contoursR, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);//找轮廓
            Mat imageContoursR=Mat::zeros(ROI_tempR.size(), CV_8UC1);
            for(int i=0;i<contoursR.size();i++)
	        {
                drawContours(imageContoursR, contoursR, i, Scalar(255),1,8,hierarchy);//遍历每一个轮廓，并且绘制轮廓。
                double area = contourArea(contoursR.at(i));
                cout << "area is " << area << endl;
                if (area > 4900) marker_contoursR = contoursR.at(i);//阈值不对，注意
            }
            //Moments muR = moments(marker_contoursR, false);
            //float dX_R  = muR.m10/muR.m00;
            //float dY_R = muR.m01/muR.m00;
            //cout << "xR, yR is " << Mat(Point(dX_R, dY_R)) << endl;
            //RotatedRect ellipseR = fitEllipse(marker_contoursR);
            //Mat image_ellipseR = Mat::zeros(ROI_tempR.size(), CV_8UC3);
            //ellipse(image_ellipseR, ellipseR, Scalar(255),1, CV_AA);
            //imshow("ROI_tempR", image_ellipseR);
//******************************************/
    //坐标矫正
    X = X -75 + dX;
    X_R = X_R -75 + dX_R;
    Y = Y -80 + dY;
    Y_R = Y_R -80 + dY_R;
          
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
    cout << "theta of goal = " << theta*180/3.14159 << "度\n" << endl;


    //球门横向距离计算
    double d_x = pow( n_x, 2);
    //double d_y = pow( n_y, 2);
    double d_y = 0;
    double d_z = pow( n_z, 2);
    double d = sqrt( d_x + d_y + d_z );
    cout << "distance between A and B = " << d*0.001 << "m\n" << endl;

    //球门横梁中心与相机坐标系距离计算
    double g_x = pow( (worldPointEst[0].x + worldPointEst[1].x)/2, 2);
    double g_y = pow( (worldPointEst[0].y + worldPointEst[1].y)/2, 2);
    double g_z = pow( (worldPointEst[0].z + worldPointEst[1].z)/2, 2);
    double dg = sqrt( g_x + g_y + g_z );
    cout << "distance between goal and camera = " << dg*0.001 << "m\n" << endl;

}


int main(int argc, char** argv)
{
	Mat image_left, image_right, tempL, tempR, image_left_visial, image_right_visial;
 /*
    //相机标定结果2coeff
   Mat cameraMatrix_L = (Mat_<double>(3, 3) << 7428.2467268254,	0,              	0,
                                                0,              	7427.535411212, 	0,
                                                1401.9402721135,    966.2408611018,	    1);
    Mat cameraMatrix_R = (Mat_<double>(3, 3) << 7329.90147076187,	0,         	        0,
                                                0,	                7330.9969844233,	0,
                                                1364.908983579, 	1096.3907663372,   	1);
    Mat rotation = (Mat_<double>(3, 3) << 0.9998543546,	-0.0124307944,	-0.0116937953,
                                        0.0130298952,	0.9985288631,	0.0526339371,
                                        0.0110223105,	-0.0527786401,	0.998545404);
    Mat translation = (Mat_<double>(3, 1) << -626.4863650518,	6.9826191261,	-38.2149064922);
*/
    //相机标定结果3coeff
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


/*
    //相机标定结果2coeff_tang
    Mat cameraMatrix_L = (Mat_<double>(3, 3) << 7419.6344032975,	0,	0,
                                                0,	7426.0948752216,	0,
                                                1503.9200203138,	1276.9401955205,	1);
    Mat cameraMatrix_R = (Mat_<double>(3, 3) << 7304.4590546088,	0,	0,
                                                0,	7310.6487560616,	0,
                                                1371.535135567,	1318.9848091805,	1);
    Mat rotation = (Mat_<double>(3, 3) << 0.9996439996,	-0.0101688916,	-0.0246671387,
                                            0.011183332,	0.9990824642,	0.0413420234,
                                            0.0242241032,	-0.0416031664,	0.9988405125);
    Mat translation = (Mat_<double>(3, 1) << -625.0329294683,	7.7090590207,	-42.2618772565);
    //相机标定结果2coeff_tang+skew
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
*/





    

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
        resize(tempL, image_left_visial, Size(612, 512));
        resize(tempR, image_right_visial, Size(612, 512));
        imshow("BlurL", image_left_visial);
        imshow("BlurR", image_right_visial);

        sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/result/left/Gaussian/BlurL_%d.png", count);
        imwrite(filename, image_left_visial);
        sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/result/right/Gaussian/BlurR_%d.png", count);
        imwrite(filename, image_right_visial);

        //二值化使特征突出，去除干扰
        threshold(tempL, tempL, 120, 255, THRESH_BINARY);
        threshold(tempR, tempR, 120, 255, THRESH_BINARY);
        resize(tempL, image_left_visial, Size(612, 512));
        resize(tempR, image_right_visial, Size(612, 512));
        imshow("BINARYL", image_left_visial);
        imshow("BINARYR", image_right_visial);

        sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/result/left/binary/BINARYL_%d.png", count);
        imwrite(filename, image_left_visial);
        sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/result/right/binary/BINARYR_%d.png", count);
        imwrite(filename, image_right_visial);

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
            double X = keypointsL[i].pt.x; 
            double Y = keypointsL[i].pt.y;
            cout << "x, y is " << X << ", " <<  Y << endl;
            cout << endl;
            circle(image_left, Point(X, Y), 30, cv::Scalar(255, 0, 0), 3);

            
            double X_R = keypointsR[i].pt.x; 
            double Y_R = keypointsR[i].pt.y;
            cout << "xR, yR is "  <<  X_R << ", " <<  Y_R  << endl;
            cout << endl;
            circle(image_right, Point(X_R, Y_R), 30, cv::Scalar(255, 0, 0), 3);

            if(abs(X - X_R) >= 1000 ) mismatch = true;//若像素值差别太大则误匹配
            else mismatch = false;
            

            findRealcenters(image_left, image_right, count, X, Y, X_R, Y_R);
            cout << "real x, y is " << X << ", " <<  Y << endl;
            cout << "real xR, yR is "  <<  X_R << ", " <<  Y_R  << endl;
            cout << endl;
/*            if(count == 1 && i==1){
                keypointsL[1].pt.x = X+0.772308;
                keypointsL[1].pt.y = Y+0.83093; 
                keypointsR[1].pt.x = X_R+0.868469;
                keypointsR[1].pt.y = Y_R+1.07196;
            }
            //keypointsL[i].pt.x = X+1.5;
            //keypointsL[i].pt.y = Y+1.5; 
            //keypointsR[i].pt.x = X_R-1.5; 
            //keypointsR[i].pt.y = Y_R-1.5;  
*/

        }
        //三角化获得世界坐标
        vector< Point3d > worldPointEst;
        triangulation(mismatch, keypointsL, keypointsR, cameraMatrix_L.t(), cameraMatrix_R.t(), rotation.inv(), translation, worldPointEst);
        cout << "world x,y,z of B is "  << worldPointEst[0] << "\n" << endl;
         cout << "world x,y,z of A is "  << worldPointEst[1] << "\n" << endl;
        Pose_analysis(worldPointEst);

        //图像尺寸缩小副本用于可视化
        resize(image_left, image_left_visial, Size(612, 512)); 
        imshow("result_blobL", image_left_visial);

		resize(image_right, image_right_visial, Size(612, 512)); 
        imshow("result_blobR", image_right_visial);

        sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/result/left/blob/result_blobL_%d.png", count);
        imwrite(filename, image_left_visial);
        sprintf(filename, "/home/jack/Desktop/C++/computer_vision/stereo/Cv_Project3_Photos/result/right/blob/result_blobR_%d.png", count);
        imwrite(filename, image_right_visial);

        cvWaitKey(0);
    }

    cvWaitKey(0);

    return 0;    
}
