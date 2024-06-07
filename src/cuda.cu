#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void setupCurand(curandState* state, unsigned long seed, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

__global__ void EncryptKernel(unsigned char* input, unsigned char* output1, unsigned char* output2, int width, int height, curandState* state) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        unsigned char pixel = input[idx];
        unsigned char randVal = curand(&state[idx]) % 256;

        output1[idx] = randVal;
        output2[idx] = pixel ^ randVal;
    }
}

__global__ void DecryptKernel(unsigned char* input1, unsigned char* input2, unsigned char* output, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        output[idx] = input1[idx] ^ input2[idx];
    }
}

void Encrypt(cv::Mat& img, cv::Mat& output1, cv::Mat& output2) {
    int width = img.cols;
    int height = img.rows;

    unsigned char* d_input;
    unsigned char* d_output1;
    unsigned char* d_output2;
    curandState* d_state;

    cudaMalloc(&d_input, width * height * sizeof(unsigned char));
    cudaMalloc(&d_output1, width * height * sizeof(unsigned char));
    cudaMalloc(&d_output2, width * height * sizeof(unsigned char));
    cudaMalloc(&d_state, width * height * sizeof(curandState));

    cudaMemcpy(d_input, img.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    setupCurand<<<numBlocks, threadsPerBlock>>>(d_state, time(NULL), width, height);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    EncryptKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output1, d_output2, width, height, d_state);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Encryption time elapsed: %f ms\n", milliseconds);

    cudaMemcpy(output1.data, d_output1, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(output2.data, d_output2, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_state);
}

void Decrypt(cv::Mat& input1, cv::Mat& input2, cv::Mat& output) {
    int width = input1.cols;
    int height = input1.rows;

    unsigned char* d_input1;
    unsigned char* d_input2;
    unsigned char* d_output;

    cudaMalloc(&d_input1, width * height * sizeof(unsigned char));
    cudaMalloc(&d_input2, width * height * sizeof(unsigned char));
    cudaMalloc(&d_output, width * height * sizeof(unsigned char));

    cudaMemcpy(d_input1, input1.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input2.data, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    DecryptKernel<<<numBlocks, threadsPerBlock>>>(d_input1, d_input2, d_output, width, height);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Decryption time elapsed: %f ms\n", milliseconds);

    cudaMemcpy(output.data, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
}

int main() {
    cv::Mat img = cv::imread("input.png", cv::IMREAD_GRAYSCALE);
    cv::Mat output1(img.rows, img.cols, CV_8UC1);
    cv::Mat output2(img.rows, img.cols, CV_8UC1);
    cv::Mat decrypted(img.rows, img.cols, CV_8UC1);

    Encrypt(img, output1, output2);
    Decrypt(output1, output2, decrypted);

    cv::imwrite("output1.png", output1);
    cv::imwrite("output2.png", output2);
    cv::imwrite("decrypted.png", decrypted);

    return 0;
}