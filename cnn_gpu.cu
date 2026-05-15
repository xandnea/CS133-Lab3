#include "lib/cnn.cuh"
#include "cnn_gpu.cuh"

// Using thread block dims: 16 16 1
// Using grid dims: 7 7 256
// 1130 GFlops

#define BLOCK_H 16
#define BLOCK_W 16

#define TILE_H (BLOCK_H * 2 + kKernel - 1)
#define TILE_W (BLOCK_W * 2 + kKernel - 1)

__global__ void cnn_gpu(float* input, float* weight, float* bias, float* output) {
  // identify thread and create thread indices for output pixel (i, h, w)
  int w = (blockIdx.x * blockDim.x) + threadIdx.x;
  int h = (blockIdx.y * blockDim.y) + threadIdx.y;
  int i = (blockIdx.z * blockDim.z) + threadIdx.z;

  // shared memory tile for input data 
  __shared__ float tile[TILE_H][TILE_W];

  // boundary check 
  if (i < kNum && h < kOutImSize && w < kOutImSize) {

    float res[4]; // 4 pixels stored for max pooling

    // initialize all result values to the bias value for the current output channel
    for (int t = 0; t < 4; ++t) {
      res[t] = bias[i];
    }

    for (int j = 0; j < kNum; ++j) {

      // load tile into shared memory
      for (int tile_y = threadIdx.y; tile_y < TILE_H; tile_y += blockDim.y) {
        for (int tile_x = threadIdx.x; tile_x < TILE_W; tile_x += blockDim.x) {
          int global_y = blockIdx.y * BLOCK_H * 2 + tile_y;
          int global_x = blockIdx.x * BLOCK_W * 2 + tile_x;

          tile[tile_y][tile_x] = input(j, global_y, global_x);
        }
      }

      __syncthreads(); // wait until all threads have loaded the tile
    
      // calculate 4 convolution results for the 2x2 pooling window
      for (int ph = 0; ph < 2; ++ph) {
        for (int pw = 0; pw < 2; ++pw) {

          int local_h = threadIdx.y * 2 + ph;
          int local_w = threadIdx.x * 2 + pw;

          float sum = 0.f;

          // Convolution
          for (int p = 0; p < kKernel; ++p) {
            for (int q = 0; q < kKernel; ++q) {
              sum += weight(i,j,p,q) * tile[local_h + p][local_w + q];
            }
          }

          res[ph * 2 + pw] += sum;
        }
      }

      __syncthreads(); // wait until all threads have finished convolution calculations for the tile
    }

    // ReLU
    for (int t = 0; t < 4; ++t) {
      res[t] = max(0.f, res[t]);
    }  

    output(i, h, w) = max( max(res[0], res[1]), max(res[2], res[3]) ); // Max pooling
  }
}