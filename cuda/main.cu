#include <stdio.h>
#include <sys/time.h>

#define NUM_BLOCKS 1000
#define NUM_THREADS 1000
#define ALL_THREADS NUM_BLOCKS * NUM_THREADS

__device__ void is_prime(int number, int *output) {
    for (int i = 2; i*i < number; i++) {
        if (number%i == 0) {
            *output = 0;
            return;
        }
    }
    *output = 1;
    return;
}

__global__ void primes_kernel(int from, int to, int *range_ends, int ranges_count, int *output) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= ranges_count) {
        return;
    }

    int thread_from = from + (index == 0 ? 0 : range_ends[index-1]);
    int thread_to = from + range_ends[index];

    for (int i = thread_from; i < thread_to; i++) {
        is_prime(i, &output[i-from]);
    }
}

int main(void) {
    struct timeval t1, t2;
    gettimeofday(&t1, 0);

    int from = 10, to = 20000;
    int interval = to - from;
    size_t size = sizeof(int) * interval;

    int *output = (int*) malloc(size);

    int ranges_count = ALL_THREADS > interval ? interval : ALL_THREADS;
    int small_interval = interval / ranges_count;
    int remainder = interval % ranges_count;


    int offset = 0;
    int *range_ends = (int*) malloc(sizeof(int) * ranges_count);
    for (int i = 0; i < ranges_count; i++) {
        offset += small_interval + (remainder > i ? 1 : 0);
        range_ends[i] = offset;
    }

    int *d_output, *d_range_ends;
    cudaMalloc((void**) &d_output, size);
    cudaMalloc((void**) &d_range_ends, sizeof(int)*ranges_count);
    cudaMemcpy(d_range_ends, range_ends, sizeof(int)*ranges_count, cudaMemcpyHostToDevice);

    primes_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(from, to, d_range_ends, ranges_count, d_output);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_output);
    cudaFree(d_range_ends);

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
