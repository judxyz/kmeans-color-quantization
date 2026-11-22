#pragma once

#include <vector>
#include <opencv2/core/mat.hpp> 
#include <opencv2/core/utility.hpp>



class Pixel {
    float B, G, R;

public:

    Pixel(float b, float g, float r) : B(b), G(g), R(r) {};

    std::array<float, 3> getColor() {
        return {B, G, R};
    }

    void setColor(float newB, float newG, float newR) {
        B = newB;
        G = newG;
        R = newR;        
    }

    int findClosestCentroid(cv::Mat centers);

};

class KMeans{
    /**
    @brief performs the kmeans clustering algo
    @param pixels vector of Pixels
    @param centers size k matrix for the calculated centers 
    @param labels 1d matrix for the assignments
    @param k # of colors to compress to (k centroids)
    @param iterations # times algorithm runs
    @param EPS epsilon - convergence threshold
    */
public:
    void calculate_kmeans(std::vector<Pixel>& pixels, cv::Mat& centers,  std::vector<int>& labels, int k, int iterations, float EPS, int seed=cv::getTickCount());
};

