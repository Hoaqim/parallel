#include <opencv2/opencv.hpp>
#include <random>
#include <chrono>
#include <omp.h>

void Encrypt(cv::Mat& img, cv::Mat& output1, cv::Mat& output2) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            int rand = dis(gen);

            output1.at<unsigned char>(y, x) = rand;
            output2.at<unsigned char>(y, x) = img.at<unsigned char>(y, x) ^ rand;
        }
    }
}

void Decrypt(cv::Mat& input1, cv::Mat& input2, cv::Mat& output) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < input1.rows; ++y) {
        for (int x = 0; x < input1.cols; ++x) {
            output.at<unsigned char>(y, x) = input1.at<unsigned char>(y, x) ^ input2.at<unsigned char>(y, x);
        }
    }
}

int main() {
    cv::Mat img = cv::imread("input.png", cv::IMREAD_GRAYSCALE);
    cv::Mat output1(img.rows, img.cols, CV_8UC1);
    cv::Mat output2(img.rows, img.cols, CV_8UC1);
    cv::Mat decrypted(img.rows, img.cols, CV_8UC1);

    auto start = std::chrono::high_resolution_clock::now();
    Encrypt(img, output1, output2);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Encryption time elapsed: " << duration.count() << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    Decrypt(output1, output2, decrypted);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Decryption time elapsed: " << duration.count() << " ms\n";

    cv::imwrite("output1.png", output1);
    cv::imwrite("output2.png", output2);
    cv::imwrite("decrypted.png", decrypted);

    return 0;
}