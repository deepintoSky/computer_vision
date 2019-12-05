#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
    VideoCapture capture;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if(!capture.open(0))
        return 0;
	

    VideoWriter writer_color, writer_gray;
	writer_color.open("color.avi", CV_FOURCC('F', 'L', 'V', '1'), 20, Size(640,480), true);
	writer_gray.open("gray.avi", CV_FOURCC('F', 'L', 'V', '1'), 20, Size(640,480), false);

    for(;;)
    {
        Mat frame;
        capture >> frame;
        if( frame.empty() ) break; // end of video stream
        imshow("original", frame);

		Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
		dilate(frame, frame, kernel);
		imshow("dilate", frame);

	  	Mat grayframe;
	 	cvtColor(frame, grayframe, COLOR_BGR2GRAY);
	  	imshow("gray", grayframe);

	  	Mat edges;
     	blur(grayframe, edges, Size(3,3) );
	  	Canny(edges, edges, 3, 8, 3);
        imshow("canny edges", edges);

	  
	  	writer_color << frame;
	  	writer_gray << grayframe;


        if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC 

    }

    // the camera will be closed automatically upon exit
    // cap.close();
    return 0;
}


