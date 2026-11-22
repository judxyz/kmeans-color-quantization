#include <iostream>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <chrono>
#include <fstream>
#include "src/b_openmp_kmeans.hpp"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    #ifdef _OPENMP
        #pragma message("Compiled with OpenMP")
    #else
        #pragma message("Not compiled with OpenMP")
    #endif

        
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <filepath> <k> <iterations> \n";
        return 1;
    }


    // setup 
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

    // --- warmup run
    KMeans.calculate_kmeans(pixels, centers, labels, k, iterations, eps);

    // --- Timed runs
    const int NUM_RUNS = 20;
    vector<double> times;

    for (int i = 0; i < NUM_RUNS; i++) {
        auto start_serial = chrono::high_resolution_clock::now();
        KMeans.calculate_kmeans(pixels, centers, labels, k, iterations, eps, 42); // fix seed=42
        
        auto end_serial = chrono::high_resolution_clock::now();

        auto duration_serial = chrono::duration_cast<chrono::milliseconds>(end_serial-start_serial);
        
        times.push_back(duration_serial.count());


    }

    
    // --- Reports
    sort(times.begin(), times.end());
    
    
    cout << "Best time (in ms): " << times.front() << endl;
    cout << "Median time (in ms): " << times[NUM_RUNS / 2] << endl;
    cout << "Mean time (in ms): " << accumulate(times.begin(), times.end(), 0.0) / times.size() << endl;


}