#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

int main(int /*argc*/, char** /*argv*/) {

    string image_path;
    image_path = "../../image.jpeg";
    
    Mat img = imread(image_path, IMREAD_COLOR);

    if (img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    imshow("Display window", img);


    int k = waitKey(0);
    if (k== 's')
    {
        imwrite("../../image.jpeg", img);
    }


    // code for kmeans
    const int N = 5;
    Scalar colorTab[] =
    {
        Scalar(0, 0, 255),
        Scalar(0,255,0),
        Scalar(255,100,100),
        Scalar(255,0,255),
        Scalar(0,255,255)
    };

    // for (;;) 
    // {
    //     int k
    // }



    return 0;
}
/*

*/

