#include <opencv2/opencv.hpp>
#include <iostream>
#include <Eigen/Dense> 

using namespace std;
using namespace Eigen;
using namespace cv;




int main()
{
	Mat image1, image2;
	image1 = imread("hw4_1.png");
	imshow("image1", image1);
	image2 = imread("hw4_2.png");
	imshow("image2", image2);

	int width = 8, length = 10;
	Size boardSize(width, length);
	vector<Point2f> cornerPoints1, cornerPoints2;
	
	//寻找角点
	bool flag = findChessboardCorners(image2, boardSize, cornerPoints2, CV_CALIB_CB_ADAPTIVE_THRESH);
	drawChessboardCorners(image2, boardSize, cornerPoints2, flag);
	//findChessboardCorners(image2, boardSize, cornerPoints2, CV_CALIB_CB_ADAPTIVE_THRESH);
	imshow("image2", image2);
	//可视化角点
	/*if (findChessboardCorners(image1, boardSize, cornerPoints1, CV_CALIB_CB_ADAPTIVE_THRESH))// CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS | CV_CALIB_CB_NORMALIZE_IMAGE))
	{
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < length; j++)
			{
				circle(image1, cornerPoints1.at(i * width + j), 4, Scalar(255, 0, 255, 0), 2);
				imshow("image1", image1);
				
				if (i == 0 && j == 0)
				{
					waitKey(0);
					cout << "该点像素坐标为:" << cornerPoints1.at(0) << endl;
				}
				
				waitKey(0);
			}
		}
	}
	else
	{
		cout << "findChessboardCorners Error" << endl;
		cout << cornerPoints1.size() << endl;
		waitKey(0);
		return 0;
	}*/
	
	int num_points = cornerPoints2.size();
	MatrixXd U(2*num_points, 1), m(11, 1), K(2*num_points, 11);
	//定义世界坐标,构造矩阵方程
	int i=0, j=0, k=0, index=0;
	for(j = 0; j <= 4; j++){
		for(k = 0; k <= 7; k++){
			index = 8*j + k;
			U(2 * index) = cornerPoints2.at(index).x;
			U(2 * index + 1) = cornerPoints2.at(index).y;
			K.row(2 * index) << i*100, j*100, k*100, 1, 0, 0, 0, 0, -i*100 * U(2 * index), -j*100 * U(2 * index), -k*100 * U(2 * index);
			K.row(2 * index + 1) << 0, 0, 0, 0, i*100, j*100, k*100, 1, -i*100 * U(2 * index + 1), -j*100 * U(2 * index + 1), -k*100 * U(2 * index + 1);
		}
	}
	j = 4;
	for(i = 0; i <= 4; i++){

		for(k = 0; k <= 7; k++){
			int index = 8*i + k + 40;
			U(2 * index) = cornerPoints2.at(index).x;
			U(2 * index + 1) = cornerPoints2.at(index).y;
			K.row(2 * index) << i*100, j*100, k*100, 1, 0, 0, 0, 0, -i*100 * U(2 * index), -j*100 * U(2 * index), -k*100 * U(2 * index);
			K.row(2 * index + 1) << 0, 0, 0, 0, i*100, j*100, k*100, 1, -i*100 * U(2 * index + 1), -j*100 * U(2 * index + 1), -k*100 * U(2 * index + 1);
		}
	}
	cout << "U is " << endl << U << endl;
	cout << "K is " << endl << K << endl;
	
	//得到m最小二乘解
	m = ( K.transpose() * K ).inverse() * K.transpose() * U;
	cout << "m is " << endl << m << endl;


	//求解内参
	Vector3d m1, m2, m3, r1, r2, r3;
	double m14, m24, m34, t_x, t_y, t_z, f_x, f_y, u0, v0;

	m1 = m.block(0,0, 3,1);
	m2 = m.block(4,0, 3,1);
	m3 = m.block(8,0, 3,1);
	m14 = m(3);
	m24 = m(7);

	m34 = 1 / m3.norm();
	r3 = m34 * m3;
	u0 = (m34*m34 * m1.transpose() * m3)(0);
	v0 = (m34*m34 * m2.transpose() * m3)(0);
	f_x = m34*m34 * (m1.cross(m3)).norm();
	f_y = m34*m34 * (m2.cross(m3)).norm();

	//求解外参
	r1 = m34/f_x * (m1 - u0*m3);
	r2 = m34/f_y * (m2 - v0*m3);
	t_z = m34;
	t_x = m34/f_x * (m14 - u0);
	t_y = m34/f_y * (m24 - v0);
	
	cout << "u0 is " << u0 << endl;
	cout << "v0 is " << v0 << endl;
	cout << "f_x is " << f_x << endl;
	cout << "f_y is " << f_y << endl;
	cout << "r1 is " << endl << r1 << endl;
	cout << "r2 is " << endl << r2 << endl;
	cout << "r3 is " << endl << r3 << endl;
	cout << "t_x is " << t_x << endl;
	cout << "t_y is " << t_y << endl;
	cout << "t_z is " << t_z << endl;

	waitKey(0);
	return 0;

}
