/**
* samples/cpp/kmeans.cpp
* https://docs.opencv.org/4.13.0/d9/dde/samples_2cpp_2kmeans_8cpp-example.html
*/
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
 
using namespace cv;
using namespace std;

 
int main( int argc, char** argv )
{
    const int MAX_CLUSTERS = 5;
    Scalar colorTab[] =
    {
        Scalar(0, 0, 255),
        Scalar(0,255,0),
        Scalar(255,100,100),
        Scalar(255,0,255),
        Scalar(0,255,255)
    };
 
    Mat img(500, 500, CV_8UC3);
    RNG rng(12345);
 
    for(;;)
    {
        int k, clusterCount = rng.uniform(2, MAX_CLUSTERS+1); // returns 1 value between [2, MAX_CLUSTERS+1)
        int i, sampleCount = rng.uniform(1, 1001);
        cout << clusterCount << endl;
        cout << sampleCount << endl;

        Mat points(sampleCount, 1, CV_32FC2), labels;
        cout << points << endl << labels << endl;

 
        clusterCount = MIN(clusterCount, sampleCount);
        vector<Point2f> centers; // 2d float coords
        cout << centers << endl;
 
        /* generate random sample from multigaussian/normal distribution */
        for( k = 0; k < clusterCount; k++ )
        {
            // points are generated around the centers
            Point center;
            center.x = rng.uniform(0, img.cols);
            center.y = rng.uniform(0, img.rows);
            Mat pointChunk = points.rowRange(k*sampleCount/clusterCount,
                                             k == clusterCount - 1 ? sampleCount :
                                             (k+1)*sampleCount/clusterCount);
            rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y), Scalar(img.cols*0.05, img.rows*0.05));
            // Scalar(center.x, center.y) <- mean
            // Scalar(img.cols * 0.05, img.rows * 0.05) <- std dev. (0.05 of total width/height)
        }
 
        randShuffle(points, 1, &rng);
 
        double compactness = kmeans(points, clusterCount, labels,
            TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
               3, KMEANS_PP_CENTERS, centers);
        // compactness: sum of squared distances
        // TermCriteria::EPS + TermCriteria::COUNT: either condiiton met:
        // 10 iterations, or
        // EPS: 1.0 pixel

        // 3 attempts


        img = Scalar::all(0);
 
        for( i = 0; i < sampleCount; i++ )
        {
            // draw points
            int clusterIdx = labels.at<int>(i); // .at<int>(i) <- read int out of that index of the matrix
            Point ipt = points.at<Point2f>(i); // integer values work here (whole # for pixels)
            circle( img, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA );
            // cv::Mat is a generic, raw block of memory designed to hold almost anything (floats, integers, RGB pixels, double-precision numbers)
            // therefore use .at<_>(i) to tell it how to read it
        }
        for (i = 0; i < (int)centers.size(); ++i)
        {
            // draw centers
            Point2f c = centers[i];
            circle( img, c, 40, colorTab[i], 1, LINE_AA );
        }
        cout << "Compactness: " << compactness << endl;
 
        imshow("clusters", img);
 
        char key = (char)waitKey();
        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
            break;
    }
 
    return 0;
}