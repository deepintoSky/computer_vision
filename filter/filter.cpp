#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <math.h>

using namespace std;
using namespace cv;

//#define SQUARE_SIZE 40
//#define TEST_H

void my_meanBlur(Mat gray, Mat out, int size_kernel)
{
	out = gray.clone();
	Size size = out.size();
	Mat temp;
	out.convertTo(temp, CV_64FC1);

	if(size_kernel == 3)
	{
		double kernel_init[3][3] ={ 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9}; 
		Mat kernel; 
		kernel = Mat(3, 3, CV_64FC1, kernel_init);//创建矩阵
		//cout << "kernel = \n" << kernel << endl;

		for (int i = 1; i < size.width - 1; i++)
		{
			uchar* pdata = out.ptr<uchar>(i);
			for (int j = 1; j < size.height - 1; j++)
			{
				//Rect(x,y,width,height)//矩形框
				//Range(start,end)//感兴趣行列范围
				Mat roi = temp(Rect(j-1,i-1,3,3));
				pdata[j] = sum(roi.mul(kernel))(0);
				//cout << "ROI = \n" << roi << endl;
				//cvWaitKey(0);
			}	
		}
	}
	else if(size_kernel == 5)
	{
		Mat kernel = Mat::ones(5, 5, CV_64FC1); 
		kernel = kernel/25;//创建矩阵
		//cout << "kernel = \n" << kernel << endl;
		
		for (int i = 2; i < size.width - 2; i++)
		{
			uchar* pdata = out.ptr<uchar>(i);
			for (int j = 2; j < size.height - 2; j++)
			{
				//Rect(x,y,width,height)//矩形框
				//Range(start,end)//感兴趣行列范围
				Mat roi = temp(Rect(j-2,i-2,5,5));
				pdata[j] = sum(roi.mul(kernel))(0);
				//cout << "ROI = \n" << roi << endl;
				//cvWaitKey(0);
			}	
		}
	}
};


void my_medianBlur(Mat gray, Mat out, int size_kernel)
{
	out = gray.clone();
	Size size = out.size();
	Mat temp;
	out.convertTo(temp, CV_64FC1);

	if(size_kernel == 3)
	{
		for (int i = 1; i < size.width - 1; i++)
		{
			uchar* pdata = out.ptr<uchar>(i);
			for (int j = 1; j < size.height - 1; j++)
			{
				Mat roi = temp(Rect(j-1,i-1,3,3));
				//cout << "ROI = \n" << roi << endl;
				Mat rortroi = roi.clone();//不连续变连续
				rortroi = rortroi.reshape(1, 9);
				cv::sort(rortroi, rortroi, SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
				//cout << "roi: \n " << rortroi << endl;
				//cout << "roi: \n " << rortroi.ptr<double>(0)[4] << endl;
				pdata[j]  = rortroi.ptr<double>(0)[4];
				//waitKey(0);
			}	
		}
	}
	else if(size_kernel == 5)
	{
		for (int i = 2; i < size.width - 2; i++)
		{
			uchar* pdata = out.ptr<uchar>(i);
			for (int j = 2; j < size.height - 2; j++)
			{
				Mat roi = temp(Rect(j-2,i-2,5,5));
				//cout << "ROI = \n" << roi << endl;
				Mat rortroi = roi.clone();//不连续变连续
				rortroi = rortroi.reshape(1, 25);
				cv::sort(rortroi, rortroi, SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
				//cout << "roi: \n " << rortroi << endl;
				//cout << "roi: \n " << rortroi.ptr<double>(0)[4] << endl;
				pdata[j]  = rortroi.ptr<double>(0)[12];
				//waitKey(0);
			}	
		}
	}
};

Mat my_getGaussian(int size_kernel, float sigma)
{
	Mat kernel;
	if(size_kernel == 3)
	{
		kernel = Mat::zeros(Size(3, 3), CV_64FC1);
		for (int i = -1; i < 2; i++)
		{
			double* pdata = kernel.ptr<double>(1+i);
			for (int j = -1; j < 2; j++)
			{
				double a, b;
				a = 1./(2*sigma*sigma*3.1415926);
				b = (i*i +j*j)/(2*sigma*sigma);
				pdata[1+j] = a*exp(-b);
			}
		}
		kernel = kernel/sum(kernel)(0);//归一化
	}
	else if(size_kernel == 5)
	{
		kernel = Mat::zeros(Size(5, 5), CV_64FC1);
		for (int i = -2; i < 3; i++)
		{
			double* pdata = kernel.ptr<double>(2+i);
			for (int j = -2; j < 3; j++)
			{
				double a, b;
				a = 1./(2*sigma*sigma*3.1415926);
				b = (i*i +j*j)/(2*sigma*sigma);
				pdata[2+j] = a*exp(-b);
			}
		}
		kernel = kernel/sum(kernel)(0);//归一化
	}
	return kernel;
};

void my_GaussianBlur(Mat gray, Mat out, int size_kernel, float sigma)
{
	out = gray.clone();
	Size size = out.size();
	Mat temp, kernel;
	out.convertTo(temp, CV_64FC1);

	if(size_kernel == 3)
	{ 
		kernel = my_getGaussian(3, sigma);//创建矩阵
		//cout << "kernel = \n" << kernel << endl;
		//cout << "sum of kernel = \n" << sum(kernel)(0) << endl;

		for (int i = 1; i < size.width - 1; i++)
		{
			uchar* pdata = out.ptr<uchar>(i);
			for (int j = 1; j < size.height - 1; j++)
			{
				Mat roi = temp(Rect(j-1,i-1,3,3));
				pdata[j] = sum(roi.mul(kernel))(0);
			}
		}
	}
	else if(size_kernel == 5)
	{
		kernel = my_getGaussian(5, sigma);//创建矩阵
		//cout << "kernel = \n" << kernel << endl;
		//cout << "sum of kernel = \n" << sum(kernel)(0) << endl;

		for (int i = 2; i < size.width - 2; i++)
		{
			uchar* pdata = out.ptr<uchar>(i);
			for (int j = 2; j < size.height - 2; j++)
			{
				Mat roi = temp(Rect(j-2,i-2,5,5));
				pdata[j] = sum(roi.mul(kernel))(0);
			}	
		}
	}
};



int main()
{
	Mat image, grayimg, temp;
	image = imread("noiseimg.bmp");
	//imshow("image", image);
	cvtColor(image, grayimg, COLOR_RGB2GRAY);
	imshow("gray", grayimg); 

	blur(grayimg, temp, Size(3, 3));
	//imshow("blur3x3", temp);
	imwrite("blur3x3.png", temp);
	blur(grayimg, temp, Size(5, 5));
	//imshow("blur5x5", temp);
	imwrite("blur5x5.png", temp);
	medianBlur(grayimg, temp, 3);
	//imshow("medianBlur3x3", temp);
	imwrite("medianBlur3x3.png", temp);
  	medianBlur(grayimg, temp, 5);
	//imshow("medianBlur5x5", temp);
	imwrite("medianBlur5x5.png", temp);
	GaussianBlur(grayimg, temp, Size(3, 3), 1.5, 1.5);
	//imshow("GaussianBlur3x3", temp);
	imwrite("GaussianBlur3x3.png", temp);
	GaussianBlur(grayimg, temp, Size(5, 5), 1.5, 1.5);
	//imshow("GaussianBlur5x5", temp);
	imwrite("GaussianBlur5x5.png", temp);

	my_meanBlur(grayimg, temp, 3);
	//imshow("my_blur3x3", temp);
	imwrite("blur3x3.png", temp);
	my_meanBlur(grayimg, temp, 5);
	//imshow("my_blur5x5", temp);
	imwrite("blur5x5.png", temp);
	my_medianBlur(grayimg, temp, 3);
	//imshow("my_medianBlur3x3", temp);
	imwrite("my_medianBlur3x3.png", temp);
  	my_medianBlur(grayimg, temp, 5);
	//imshow("my_medianBlur5x5", temp);
	imwrite("my_medianBlur5x5.png", temp);
	my_GaussianBlur(grayimg, temp, 3, 1.5);
	//imshow("my_GaussianBlur3x3", temp);
	imwrite("my_GaussianBlur3x3.png", temp);
	my_GaussianBlur(grayimg, temp, 5, 1.5);
	//imshow("my_GaussianBlur5x5", temp);
	imwrite("my_GaussianBlur5x5.png", temp);
	
	//imshow("gray_after", grayimg); 
	waitKey(0);
	return 0;
}
