// Header inclusions, if any...
#include "lib/cnn.cuh"
#include "cnn_gpu.cuh"

// Using declarations, if any...

__global__ void cnn_gpu(float* input, float* weight, float* bias, float* output) {
  // identify thread and create thread indices for output pixel (i, h, w)
  int w = (blockIdx.x * blockDim.x) + threadIdx.x;
  int h = (blockIdx.y * blockDim.y) + threadIdx.y;
  int i = (blockIdx.z * blockDim.z) + threadIdx.z;

  // boundary check 
  if (i < kNum && h < kOutImSize && w < kOutImSize) {
    float res[4]; // 4 pixels stored for max pooling

    // calculate 4 convolution results for the 2x2 pooling window
    for (int ph = 0; ph < 2; ++ph) {
      for (int pw = 0; pw < 2; ++pw) {
        float sum = bias[i]; // bias

        int cur_h = h * 2 + ph;
        int cur_w = w * 2 + pw;

        // Convolution
        for (int j = 0; j < kNum; ++j) {
          for (int p = 0; p < kKernel; ++p) {
            for (int q = 0; q < kKernel; ++q) {
              sum += weight(i,j,p,q) * input(j,cur_h+p,cur_w+q);  
            }
          }
        }

        res[ph * 2 + pw] = max(0.f, sum); // ReLU
      }
    }

    output(i, h, w) = max( max(res[0], res[1]), max(res[2], res[3]) ); // Max pooling
  }
}
