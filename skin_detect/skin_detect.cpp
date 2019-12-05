#include <opencv2/opencv.hpp>
#include <iostream>
//#include <algorithm>
//#include <math.h>

using namespace std;
using namespace cv;

void skin_detect_rgb(Mat src, Mat out)
{
	for (int i = 0; i<out.rows; i++)
	{
		for (int j = 0; j<out.cols; j++)
		{
            //b,g,r
            bool img1 = (out.at<Vec3b>(i, j)[0] > 90) && (out.at<Vec3b>(i, j)[0] < 150);
            bool img2 = (out.at<Vec3b>(i, j)[1] > 120) && (out.at<Vec3b>(i, j)[1] < 180);
            bool img3 = (out.at<Vec3b>(i, j)[2] > 180) && (out.at<Vec3b>(i, j)[2] < 240);
			if(img1 && img2 && img3)
            {
                out.at<Vec3b>(i, j)[0] = 255;
                out.at<Vec3b>(i, j)[1] = 255;
                out.at<Vec3b>(i, j)[2] = 255;
            }  
            else
            {
                out.at<Vec3b>(i, j)[0] = 0;
                out.at<Vec3b>(i, j)[1] = 0;
                out.at<Vec3b>(i, j)[2] = 0;
            }
        }
    }
};

void skin_detect_hsv(Mat src, Mat out)
{
	for (int i = 0; i<out.rows; i++)
	{
		for (int j = 0; j<out.cols; j++)
		{
            //b,g,r
            uchar b, g, r, max, min;
            max = b;
            min = b;
            b = src.at<Vec3b>(i, j)[0];
            g = src.at<Vec3b>(i, j)[1];
            r = src.at<Vec3b>(i, j)[2];
            if (g > max) max = g;
            if (r > max) max = r;
            if (min > g) min = g;
            if (min > r) min = r;

            float h_channel;
            if(max == min)              h_channel = 0;
            else if(max == r && g >= b) h_channel = 0 + 60 * (g-b) / (max-min);
            else if(max == r && g < b)  h_channel = 360 + 60 * (g-b) / (max-min);
            else if(max == g)           h_channel = 120 + 60 * (g-b) / (max-min);
            else                        h_channel = 240 + 60 * (g-b) / (max-min);

			if(h_channel > 0 && h_channel < 50)
            {
                out.at<Vec3b>(i, j)[0] = 255;
                out.at<Vec3b>(i, j)[1] = 255;
                out.at<Vec3b>(i, j)[2] = 255;
            }  
            else
            {
                out.at<Vec3b>(i, j)[0] = 0;
                out.at<Vec3b>(i, j)[1] = 0;
                out.at<Vec3b>(i, j)[2] = 0;
            }
        }
    }
};

void myErode(Mat Src, Mat kernel, Mat Dst)
{
	for (int i = 1; i < Src.rows - 1; i++)//i、j的范围保证结构元始终在扩展后的图像内部
	{
		for (int j = 1; j < Src.cols - 1; j++)
		{
			Mat roi= Src(Rect(j-1,i-1,3,3));
            
			double count = 0;
			count = sum(roi.mul(kernel))(0);//矩阵对应位置相乘后求和
            //cout << "count = \n" << count << endl;
            //cvWaitKey(0);
			if (count >= 5*255)//结构元的9个元素均为1，和为9才能保证结构元完全包含于相应集合
				Dst.at<uchar>(i, j) = 255;
			else
				Dst.at<uchar>(i, j) = 0;
		}
	}
};

//定义膨胀函数
void myDilate(Mat Src, Mat kernel, Mat Dst)
{
	for (int i = 1; i < Src.rows - 1; i++)//i、j的范围保证结构元始终在扩展后的图像内部
	{
		for (int j = 1; j < Src.cols - 1; j++)
		{
			Mat roi = Src(Rect(j-1,i-1,3,3));
            //cout << "roi = \n" << roi << endl;
			double count = 0;
			count = sum(roi.mul(kernel))(0);//矩阵对应位置相乘后求和
			if (count != 0)//结构元的9个元素均为1，只要和不为0，就能说明结构元与相应集合有交集
				Dst.at<uchar>(i, j) = 255;
			else
				Dst.at<uchar>(i, j) = 0;
		}
	}
};

void myClose(Mat Src, Mat kernel, Mat Dst)
{
	Mat temp2 = Src.clone();
    myDilate(Src, kernel, temp2);
    //imshow("temp", temp2);
    //cvWaitKey(0);

    myErode(temp2, kernel, Dst);
	//imshow("result1", Dst);
    //cvWaitKey(0);
};

void myOpen(Mat Src, Mat kernel, Mat Dst)
{
	Mat temp2 = Src.clone();
    myErode(Src, kernel, temp2);
    myDilate(temp2, kernel, Dst);
	
};


int main(int argc, char** argv)
{
   
	
	Mat image1, image2, binary, temp1, temp2;
	image1 = imread("Image1.png");
	imshow("image1", image1);
	image2 = imread("Image2.png");
	imshow("image2", image2);

	//Mat kernel = Mat::ones(3, 3, CV_8UC1);
    uchar kernel_init[3][3] ={ 0, 1, 0, 1, 1, 1, 0, 1, 0}; 
	Mat kernel; 
	kernel = Mat(3, 3, CV_8UC1, kernel_init);//创建矩阵
    cout << "kernel = \n" << kernel << endl;
    //cvWaitKey(0);

    temp1 = image1.clone();
    skin_detect_rgb(image1, temp1);
    cvtColor(temp1, binary, CV_BGR2GRAY);
    //imshow("temp", binary);
    //cvWaitKey(0);

    temp2 = binary.clone();
    myOpen(binary, kernel, temp2);
    myClose(temp2, kernel, binary);    
    imshow("result1", binary);
    cvWaitKey(0);

    temp1 = image2.clone();
    skin_detect_rgb(image2, temp1);
    cvtColor(temp1, binary, CV_BGR2GRAY);
    //imshow("temp", binary);

    temp2 = binary.clone();
    myOpen(binary, kernel, temp2);
    myClose(temp2, kernel, binary);    
    imshow("result2", binary);
    cvWaitKey(0);
   

    VideoCapture capture;
    if(!capture.open("Liyifeng.avi"))
        return 0;

    //VideoWriter writer_color;
	//writer_color.open("result.avi", CV_FOURCC('F', 'L', 'V', '1'), 20, Size(640,480), true);

    for(;;)
    {
        Mat frame;
        capture >> frame;
        if( frame.empty() ) break; // end of video stream
        imshow("original", frame);

        temp1 = frame.clone();
        skin_detect_rgb(frame, temp1);
        cvtColor(temp1, binary, CV_BGR2GRAY);

        //temp2 = binary.clone();
        //myClose(binary, kernel, temp2);
        //myOpen(temp2, kernel, binary);    
	    imshow("result_video", binary);

        //writer_color << temp;
        if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC 

    }
    // the camera will be closed automatically upon exit
    capture.release();


    cvWaitKey(0);
    return 0;
}


