#include "b_openmp_kmeans.hpp"
#include <array>
#include <filesystem>
#include <iostream>

#include <vector>
#include <omp.h>

#include <opencv2/core/matx.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <fstream>
#include <chrono>

using namespace cv;
using namespace std;


int Pixel::findClosestCentroid(Mat centers) {
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


void KMeans::calculate_kmeans(vector<Pixel>& pixels, Mat& centers, vector<int>& labels, int k, int iterations, float EPS, int seed) {
    float current_eps = numeric_limits<float>::max();

    // 1. initialization
    // randomly select k cluster centroids from all pixels

    RNG rng(seed); // OpenCV's random number generator

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
        fill(labels.begin(), labels.end(), 0);

        // 2. e-step/reassigning points: 
        // iterate through points
        #pragma omp parallel for
        for (int i = 0; i < pixels.size(); i++) {
            int closest_center = pixels[i].findClosestCentroid(centers);
            labels[i] = closest_center;
            //push_back is not thread safe
        }

        // 3. m-step/moving centroids
        // recalculate the centroid of the cluster by averaging all points inside that cluster
        vector<int> num_points_per_cluster(k, 0);

        // reset all values for centers
        for (int i = 0; i < k; i++) {
            centers.at<Vec3f>(i, 0) = Vec3f(0.0f, 0.0f, 0.0f);
        }

        fill(num_points_per_cluster.begin(), num_points_per_cluster.end(), 0);

        // OMP - use local partial sums
        #pragma omp parallel 
        {
            // // print # of threads here
            // if (omp_get_thread_num() == 0) {
            //     printf("Running with %d threads \n", omp_get_num_threads());
            // }

            
            // variables declared inside #pragma omp parallel are private to each thread
            vector<Vec3f> local_sums(k, Vec3f(0, 0, 0));
            vector<int> local_counts(k, 0);
            // but not inside below for loop

            #pragma omp for nowait // nowait means each thread moves past loop as soon as it finishes its chunk of pixels, and doesn't have to wait for slower threads
            for (int p = 0; p < pixels.size(); p++) 
            {
                // declare b, g, r, target_center inside the for block so OpenMP doesn't treat them as shared
                float b, g, r;
                int target_center;
                b = (pixels[p]).getColor()[0];
                g = (pixels[p]).getColor()[1];
                r = (pixels[p]).getColor()[2];
                
                
                target_center = labels[p];
                local_sums[target_center][0] += b;
                local_sums[target_center][1] += g;
                local_sums[target_center][2] += r;

                local_counts[target_center] += 1;
            }
            #pragma omp critical
            // 
            {
                for (int i = 0; i < k; i++) 
                {
                    centers.at<Vec3f>(i, 0)[0] += local_sums[i][0];
                    centers.at<Vec3f>(i, 0)[1] += local_sums[i][1];
                    centers.at<Vec3f>(i, 0)[2] += local_sums[i][2];

                    num_points_per_cluster[i] += local_counts[i];
                }
            }
        }
        
        for (int c = 0; c < centers.rows; c++) {
            if (num_points_per_cluster[c] > 0)
            {
                centers.at<Vec3f>(c, 0) /= num_points_per_cluster[c];
            }
        }
    
    }

    // apply BGR values of cluster centers to all pixels
    // Replace each pixel with the center of its assigned cluster.

    #pragma omp parallel for
    for (int p = 0; p < pixels.size(); p++) {
        // set color of each pixel
        Vec3f newc = centers.at<Vec3f>(labels[p], 0);
        pixels[p].setColor(newc[0], newc[1], newc[2]);
    }

    return;

}


#ifndef BENCHMARK

int main(int argc, char** argv) {
    #ifdef _OPENMP
        #pragma message("Compiled with OpenMP")
    #else
        #pragma message("Not compiled with OpenMP")
    #endif

    //set log level for opencv
    cv::utils::logging::setLogLevel(
        cv::utils::logging::LOG_LEVEL_SILENT
    );

    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <filepath> <k> <iterations> \n";
        return 1;
    }

    string image_path;
    image_path = argv[1];

    ifstream file(image_path);
    if (!file.is_open()) {
        cerr << "Error could not open file at " << image_path << endl;
        return 1;
    }

    
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



    // ---- perform K means algorithm

    KMeans KMeans;

    int k;
    k = stoi(argv[2]);
    int iterations;
    iterations = stoi(argv[3]);
    float eps = 0.1;
    vector<int> labels(img.total()); // 2d array for each center point
    cout << labels.size() << " " << pixels.size() << endl;


    Mat centers;
    centers = Mat::zeros(k, 1, CV_32FC3); // fill with k zeros


    auto start_serial = chrono::high_resolution_clock::now();

    KMeans.calculate_kmeans(pixels, centers, labels, k, iterations, eps);

    auto end_serial = chrono::high_resolution_clock::now();

    auto duration_serial = chrono::duration_cast<chrono::milliseconds>(end_serial-start_serial);
    cout << "Time taken for kmeans algorithm (in ms): " << duration_serial.count() << endl;


    cout << centers << endl;
    cout << labels.size() << endl;
    
    

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
        filesystem::path input_path(argv[1]);
        filesystem::path output_path = input_path.parent_path() / ("after_" + input_path.filename().string());
        cout << output_path << endl;
        if (imwrite(output_path.string(), new_img)) {
            cout << "Successfully saved image" << endl;
        } else {
            cout << "!! Failed to save image" << endl;

        }
        
    }
    return 0;
}

#endif