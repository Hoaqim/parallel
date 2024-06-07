#include <opencv2/opencv.hpp>
#include <random>
#include <chrono>
#include <omp.h>

void Encrypt(cv::Mat& img, cv::Mat& output1, cv::Mat& output2) {
    int rows = img.rows;
    int cols = img.cols;

    std::vector<unsigned char> randomNumbers(rows * cols);

    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_int_distribution<> dis(0, 255);

        #pragma omp for
        for (int i = 0; i < rows * cols; ++i) {
            randomNumbers[i] = dis(gen);
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int idx = y * cols + x;
            unsigned char rand = randomNumbers[idx];
            unsigned char pixel = img.data[idx];

            output1.data[idx] = rand;
            output2.data[idx] = pixel ^ rand;
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Main loop time elapsed: " << duration.count() << " ms\n";
}

void Decrypt(cv::Mat& input1, cv::Mat& input2, cv::Mat& output) {
    int rows = input1.rows;
    int cols = input1.cols;

    #pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int idx = y * cols + x;
            output.data[idx] = input1.data[idx] ^ input2.data[idx];
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