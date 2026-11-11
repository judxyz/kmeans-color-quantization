#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/utility.hpp>
#include <string>
#include <set>
#include <vector>
#include <limits.h>

#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;



class Pixel {
    float B, G, R;

    public:
    Pixel(float b, float g, float r) : B(b), G(g), R(r) {}

    int findClosestCentroid(Mat centers) {
        float mindist = numeric_limits<float>::max();
        int closest_center = 0;

        for (int i = 0; i < centers.rows; ++i) 
        {
            const Vec3f centroid = centers.at<Vec3f>(i, 0);
            float dist = sqrt(pow(centroid[0] - B, 2)  + pow(centroid[1] - G, 2) + pow(centroid[2] - R, 2));

            if (dist < mindist)
            {
                mindist = dist;
                closest_center = i;
            }
        }
        return closest_center;
    }

    array<float, 3> getColor() {
        return {B, G, R};
    }

    void setColor(float newB, float newG, float newR) {
        B = newB;
        G = newG;
        R = newR;        
    }

};

class KMeans{
    /**
    @brief performs the kmeans clustering algo
    @param pixels vector of Pixels
    @param centers size k matrix for the calculated centers 
    @param assignments 2d matrix (k rows) for the assignments
    @param k # of colors to compress to (k centroids)
    @param iterations # times algorithm runs
    @param EPS epsilon - convergence threshold
    */

    public:
    void calculate_kmeans(vector<Pixel>& pixels, Mat centers,  vector<vector<Pixel*>>& assignments, int k, int iterations, float EPS) {
        float current_eps = numeric_limits<float>::max();

        // 1. randomly select k cluster centroids from all pixels

        RNG rng(getTickCount()); // OpenCV's random number generator

        for (int i = 0; i < k; i++)
        {
            int index = rng.uniform(0, pixels.size());
            centers.at<Vec3f>(i, 0) = Vec3f(pixels[index].getColor()[0], pixels[index].getColor()[1], pixels[index].getColor()[2]);
        }

        // loop 
        for (int i=0; i < iterations; i++) {
            // 1. check epsilon
            if (current_eps < EPS)
            {
                return;
            }
            // reset assignments
            for (auto& row: assignments) {
                row.clear();
            }
            // 2. iterate through points
            for (int i = 0; i < pixels.size(); i++) {
                int closest_center = pixels[i].findClosestCentroid(centers);
                assignments[closest_center].push_back(&pixels[i]);
            }

            // 3. recalculate the centroid of the cluster by averaging all points inside that cluster
            for (int k = 0; k < assignments.size(); k++) {
                float b, g, r;
                b = g = r = 0.0f;
                for (int p = 0; p < assignments[k].size(); p++) {
                    b += (*assignments[k][p]).getColor()[0];
                    g += (*assignments[k][p]).getColor()[1];
                    r += (*assignments[k][p]).getColor()[2];
                }
                b /= assignments[k].size();
                g /= assignments[k].size();
                r /= assignments[k].size();
                centers.at<Vec3f>(k, 0) = Vec3f(b, g, r);
            }

        }

        return;

    }

};


int main(int /*argc*/, char** /*argv*/) {
    //set log level for opencv
    cv::utils::logging::setLogLevel(
        cv::utils::logging::LOG_LEVEL_SILENT
    );

    string image_path;
    image_path = "../../image.jpeg"; // TODO: change to be argv
    
    Mat img8u = imread(image_path, IMREAD_COLOR); // opencv default reads in as 8UC3
    Mat img;
    img8u.convertTo(img, CV_32FC3);

    // OpenCV imread loads in image in BGR, but k means algorithm will group colors no matter the ordering of pixel values

    // check image exists
    if (img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    // Store each image pixel as a Pixel object. 
    vector<Pixel> pixels;
    pixels.reserve(img.total());
    for (int y = 0; y < img.rows; ++y)
    {
        for (int x = 0; x < img.cols; ++x)
        {
            const Vec3f color = img.at<Vec3f>(y, x);
            pixels.emplace_back(color[0], color[1], color[2]);
        }
    }


    cout << img.total() << " total pixels" << endl;



    /////// perform K means algo

    KMeans KMeans;

    int k = 5;
    int iterations = 5;
    float eps = 0.1;
    vector<vector<Pixel*>> assignments(k); // 2d array for each center point
    Mat centers;
    centers = Mat::zeros(k, 1, CV_32FC3); // fill with k zeros


    // TODO: get compactness
    KMeans.calculate_kmeans(pixels, centers, assignments, k, iterations, eps);

    cout << centers << endl;
    cout << assignments[0].size() << endl;
    // apply BGR values of cluster centers to all pixels
    // Replace each pixel with the center of its assigned cluster.

    for (int i = 0; i < k; i++) {
        Vec3f newc = centers.at<Vec3f>(i, 0);

        for (int p = 0; p < assignments[i].size(); p++) {
            // set color of each pixel
            assignments[i][p]->setColor(newc[0], newc[1], newc[2]);
        }
    }

    // convert pixels back into image
    Mat new_img; 
    new_img = Mat::zeros(img.size(), img.type());

    for (int y = 0; y < img.rows; ++y)
    {
        for (int x = 0; x < img.cols; ++x)
        {
            array<float, 3> newcolor = pixels[y*img.cols +x].getColor();
            new_img.at<Vec3f>(y, x) = Vec3f(newcolor[0], newcolor[1], newcolor[2]);
        }
    }

    cout << img.at<Vec3f>(0, 0) << endl;
    cout << new_img.at<Vec3f>(0, 0) << endl;

    
    // show image
    imshow("Display window", new_img / 255.0); // values need to be 0-1.0
    

    int key = waitKey(0);
    if (key == 's')
    {
        imwrite("../../new_image.jpeg", img);
    }
    return 0;
}

