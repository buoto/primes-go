#include <stdio.h>
#include <sys/time.h>

#define NUM_BLOCKS 100
#define NUM_THREADS 100

__device__ void is_prime(int number, int *output) {
    for (int i = 2; i*i < number; i++) {
        if (number%i == 0 ) {
            *output = 0;
            return;
        }
    }
    *output = 1;
    return;
}

__global__ void primes_kernel(int from, int to, int *output) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    while (from + index < to) {
        is_prime(from + index, &output[index]);
        index = index +  blockDim.x * gridDim.x;
    }
}

int main(void) {
    struct timeval t1, t2;
    gettimeofday(&t1, 0);

    int from = 10, to = 400000;
    size_t size = sizeof(int) * (to - from);

    int *output = (int*) malloc(size);
    int *d_output;



    cudaMalloc((void**) &d_output, size);

    primes_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(from, to, d_output);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_output);

    gettimeofday(&t2, 0);

    for (int i = 0; i < to-from; i ++) {
        if (output[i]) {
            printf("%d\n", from + i);
        }
    }
    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("Elapsed time:  %3.1f ms \n", time);


    free(output);
    return 0;
}
