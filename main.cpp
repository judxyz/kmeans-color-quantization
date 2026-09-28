#include <algorithm>
#include <iostream>
#include <opencv2/core/matx.hpp>
#include <string>
#include <set>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;


/**
Vec3bLess: comparision function for comparing 2 colors (B,G,R)
*/
struct Vec3bLess 
{
    bool operator()(const cv::Vec3b& a, const cv::Vec3b& b) const 
    {
        return std::tie(a[0], a[1], a[2]) < std::tie(b[0], b[1], b[2]);
    }
};

int main(int /*argc*/, char** /*argv*/) {
    //set log level for opencv
    cv::utils::logging::setLogLevel(
        cv::utils::logging::LOG_LEVEL_SILENT
    );

    string image_path;
    image_path = "../../image.jpeg";
    
    Mat img = imread(image_path, IMREAD_COLOR);
    // OpenCV loads in image in BGR, but k means algorithm will group colors no matter the ordering of pixel values

    // check image exists
    if (img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }


    // reshape to 1 column 32bit F type
    Mat img_data = img.reshape(1, static_cast<int>(img.total()));

    img_data.convertTo(img_data, CV_32F);


    //count # of unique values / all pixels

    set<Vec3b, Vec3bLess> uniqueset;
    for (int y = 0; y < img.rows; ++y) 
    {
        for (int x = 0; x < img.cols; ++x) 
        {
            uniqueset.insert(img.at<Vec3b>(y, x)); // OpenCV uses at(row, column)
        }
    }
    
    cout << uniqueset.size() << " unique BGR values out of " << img.total() << " total pixels" << endl;

    cout << "Data:\n" << img_data.rowRange(0, 5) << endl;


    /////// perform K means algo

    Mat labels;
    // kmeans returns one 3-component floating-point center per cluster.
    Mat centers; // 100 rows x 3 columns, CV_32F


    double compactness = kmeans(img_data, 5, labels,
            TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
               3, KMEANS_PP_CENTERS, centers);
    // compactness: sum of squared distances
    // TermCriteria::EPS + TermCriteria::COUNT: either condiiton met:
    // 10 iterations, or
    // EPS: 1.0 pixel
    // 3 attempts

    cout << centers << endl;

    // apply BGR values of cluster centers to all pixels
    // Replace each pixel with the center of its assigned cluster.
    Mat colors(img_data.rows, 1, CV_32FC3);
    for (int i = 0; i < labels.rows; ++i)
    {
        int cluster = labels.at<int>(i, 0);
        colors.at<Vec3f>(i, 0) = Vec3f(
            centers.at<float>(cluster, 0),
            centers.at<float>(cluster, 1),
            centers.at<float>(cluster, 2)
        );
    }

    Mat new_img;
    colors.convertTo(new_img, CV_8UC3);
    new_img = new_img.reshape(3, img.rows);


    
    // show image
    imshow("Display window", new_img);


    int k = waitKey(0);
    if (k== 's')
    {
        imwrite("../../new_image.jpeg", img);
    }

    return 0;
}