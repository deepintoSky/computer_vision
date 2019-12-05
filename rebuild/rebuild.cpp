#include <opencv2/opencv.hpp>
#include <iostream> 
#include <fstream>
#include "Eigen/Eigen"
#include <Eigen/Core>

using namespace cv;
using namespace std;
using namespace Eigen;

int main()
{
	Mat image1, image2, image3, image4,  StdDev;
	image1 = imread("(0.2 0 -1).png");
	cvtColor(image1, image1, COLOR_RGB2GRAY);
	image2 = imread("(0 0.2 -1).png");
	cvtColor(image2, image2, COLOR_RGB2GRAY);
	image3 = imread("(0 0 -1).png");
	cvtColor(image3, image3, COLOR_RGB2GRAY);
	image4 = imread("(0 -0.2 -1).png");
	cvtColor(image4, image4, COLOR_RGB2GRAY);
	/**************************************************************************************************************/

	
	Size size = image1.size();
	int count = size.width * size.height;
	MatrixXd I(4, count), s(4, 3), n(3, count);
	s << 0.2, 0, -1, 0, 0.2, -1, 0, 0, -1, 0, -0.2, -1;
	//cout << "s=" << s << endl;
	ofstream file("n.txt", ios::out);
	for (int i = 0; i < size.width; i++)
		{
			uchar* pdata1= image1.ptr<uchar>(i);
			uchar* pdata2= image2.ptr<uchar>(i);
			uchar* pdata3= image3.ptr<uchar>(i);
			uchar* pdata4= image4.ptr<uchar>(i);
			for (int j = 0; j < size.height; j++)
			{
				uchar data1=pdata1[j];
				uchar data2=pdata2[j];
				uchar data3=pdata3[j];
				uchar data4=pdata4[j];
				MatrixXd I_temp(4, 1), n_temp(3, 1);
				I_temp(0) = double(data1);
				I_temp(1) = double(data2);
				I_temp(2) = double(data3);
				I_temp(3) = double(data4);
				//cout << "I=" << I_temp << endl;
				n_temp = (s.transpose() * s ).inverse() * s.transpose() * I_temp;
				float pho = n_temp.norm();
				n_temp = n_temp / pho;
				n_temp = n_temp / n_temp(2); 
				//n_temp(0) = - n_temp(0)/n_temp(2, 0);
				cout << "n=" << n_temp << endl;
				I.col(i*size.height + j) = I_temp;
				n.col(i*size.height + j) = n_temp;

				file << n_temp << endl;		//将像素值转化为整数后写入文件
				cvWaitKey(0);
			}
			cout << endl ;			//打印一行后进行换行
			file << endl ;	
		}
	file.close();//关闭文件
	cout << endl;  
	cout << "size.height=" << size.height << endl;
	cout << "size.width=" << size.width << endl;
	cvWaitKey(0);
}

