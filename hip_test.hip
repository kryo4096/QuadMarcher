#include <hip/hip_runtime.h>

#define HIP_ASSERT(x) (assert((x) == hipSuccess))

__global__ void hip_set(float *a) {
  int i = threadIdx.x;
  a[i] = 100.0f + i;
}

__global__ void hip_test(float *a) {
  
  int i = threadIdx.x;
  printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
  printf("a[%d] = %f\n", i, a[i]);
}

int main() {
  float *a;

  HIP_ASSERT(hipMalloc<float>(&a, 10 * sizeof(float)));

  hipLaunchKernelGGL(hip_set, dim3(1), dim3(10), 0, 0, a);
  hipLaunchKernelGGL(hip_test, dim3(1), dim3(10), 0, 0, a);

  HIP_ASSERT(hipFree(a));

  return 0;
}