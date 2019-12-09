#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
    VideoCapture capture1;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if(!capture1.open(1))
        return 0;
	
	
    for(;;)
    {
        Mat view0, m_mView;
		capture1 >> view0;

        view0.copyTo(m_mView);
 
			Mat mLeftView = m_mView(cv::Rect(0, 0, 640/2, 480));
			Mat mRightView = m_mView(cv::Rect(640/2, 0, 640/2, 480));
			imshow("cameraR", mRightView);
			imshow("cameraL", mLeftView);



        if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC 

    }

    return 0;
}