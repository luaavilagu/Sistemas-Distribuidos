%%cu

/* Programa: Approximate Matrix Multiplication (AMM) CUDA - Uniform sampling  */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#include "omp.h"

#define THREADS 16

#define ADIM 4*2048//2
#define DIM 4*1024//8
#define BDIM 4*2048//4

//float pk = 1.0/DIM;
float *pk;

__global__ void multiplyMatrix_d(float *d_matS, float *d_matR, float *d_res, int s){

    int row = threadIdx.x;
    int col = threadIdx.y;
    for (int t = 0; t < s; t++){
        float va = *( d_matS + (( row * s ) + t));
        float vb = *( d_matR + (( t * BDIM ) + col));
        float vc = *( d_res + (( row * BDIM ) + col));
        *(d_res + ((row * BDIM) + col)) = ((va * vb) + vc);
    }
}

void pkt(float *matA, float *matB){
    int sumAB = 0;
    #pragma omp parallel for reduction (+:sumAB) num_threads(THREADS)
        for (int t = 0; t < DIM; t++){
            int sumA = 0;

            for (int row = 0; row < ADIM; row++){
                //sumA += matA[row][t];
                sumA += *( matA + ( row * DIM ) + t );
            }
            int sumB = 0;
            for (int col = 0; col < BDIM; col++){
                //sumB += matB[t][col];
                sumB += *( matB + ( t * BDIM ) + col );
            }
            pk[t] = sumA * sumB;
            sumAB += sumA * sumB;
        }

    #pragma omp parallel for
    for (int t = 0; t < DIM; t++){
        pk[t] /= sumAB;
    }
}

void printMatrix(int dimy, int dimx, float *mat)
{
    printf("\n");
    for (int i = 0; i < dimy * dimx; i++){
        if ( i % dimx == 0) printf("\n");
        printf("%f  ", *(mat + i));
    }
    printf("\n");
}

void sampleAB(float *matS, float *matA, float *matR, float *matB, int s, float *pk){
    for (int t = 0; t < s; t++ ){
        int it = rand() % DIM;

        #pragma omp parallel for num_threads(THREADS)
            for (int row = 0; row < ADIM; row++){
                *( matS + ( row * s ) + t ) = (*( matA + ( row * DIM) + it )/(sqrt(s*pk[it])));
            }

        #pragma omp parallel for num_threads(THREADS)
            for (int col = 0; col < BDIM; col++){
                *( matR + ( t * BDIM ) + col ) = (*( matB + ( it * BDIM ) + col )/(sqrt(s*pk[it])));
            }
    }

}

void multiplyMatrixUniformSampling(float *matA, float *matB, float *res, int s){

    float *matS, *matR;
    matS = (float*)malloc( ADIM * s * sizeof(float));    
    matR = (float*)malloc( s * BDIM * sizeof(float));

    if ( matS == NULL || matR == NULL ){
        printf("NULL POINTER!\n");
    }

    sampleAB(matS, matA, matR, matB, s, pk);

    //Device malloc
    float *d_matS, *d_matR, *d_res;
    cudaMalloc(&d_matS, ADIM * s * sizeof(float));
    cudaMalloc(&d_matR, s * BDIM * sizeof(float));
    cudaMalloc(&d_res, ADIM * BDIM * sizeof(float));

    //Device copyToDevice
    cudaMemcpy(d_matS, matS, ADIM * s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matR, matR, s * BDIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, ADIM * BDIM * sizeof(float), cudaMemcpyHostToDevice);

    //Device Kernel Execution
    dim3 numThreadsPerBlock (ADIM, BDIM);
    multiplyMatrix_d <<< 1, numThreadsPerBlock >>> (d_matS, d_matR, d_res, s);   

    //Device copyToHost
    cudaMemcpy(res, d_res, ADIM * BDIM * sizeof(float), cudaMemcpyDeviceToHost);

    //Device Free
    cudaFree(d_matS);
    cudaFree(d_matR);
    cudaFree(d_res);

    //Host free
    free(matS);
    free(matR);
}

void initializeMatrix(int dimy, int dimx, float *mat)
{
    for (int y = 0; y < dimy; y++){
        for (int x = 0; x < dimx; x++){
            *(mat + (( y * dimx) + x)) = ((float)rand()/(float)(RAND_MAX)) * 20;
        }
    }
}

void initializeMatrix0(int dimy, int dimx, float *mat)
{
    for (int y = 0; y < dimy; y++){
        for (int x = 0; x < dimx; x++){
            *(mat + (( y * dimx) + x)) = 0;
        }
    }
}

void initializeMatrixes(float* matA, float* matB, float* res, float* real){
    initializeMatrix(ADIM, DIM, matA);
    initializeMatrix(DIM, BDIM, matB);
    initializeMatrix0(ADIM, BDIM, res);
    initializeMatrix0(ADIM, BDIM, real);
}

void beginUniformSampling(){

    struct timeval tval_before, tval_after, tval_result;

    //Instanciamos variables de host
    float *matA, *matB, *res, *real;

    //Reservamos memoria para variables host
    matA = (float*)malloc(ADIM * DIM * sizeof(float));
    matB = (float*)malloc(DIM * BDIM * sizeof(float));
    res = (float*)malloc(ADIM * BDIM * sizeof(float));
    real = (float*)malloc(ADIM * BDIM * sizeof(float));

    pk = (float*)malloc( DIM * sizeof(float));

    //Verificamos que no sean nulos los apuntadores
    if (matA == NULL || matB == NULL || res == NULL ) printf("Error: if (matA == NULL || matB == NULL || res == NULL )");

    srand(200);

    initializeMatrixes(matA, matB, res, real);
    
    printf("\n Begin Uniform Sampling CUDA \n");
    gettimeofday(&tval_before, NULL);
    pkt(matA, matB);
    int s = rand() % DIM;
    multiplyMatrixUniformSampling(matA, matB, res, s);
    
    gettimeofday(&tval_after, NULL);

    timersub(&tval_after, &tval_before, &tval_result);
    printf("\ns: %d\n", s);
    printf("%ld,%06ld\n", (long int) tval_result.tv_sec, (long int) tval_result.tv_usec);
    //printf("\nResult US DEVICE");
    //printMatrix(ADIM, BDIM, res);

    //Liberación memoria de host
    free(matA);
    free(matB);
    free(res);
    free(real);
  
}

int main()
{
    
    beginUniformSampling();

    return 0;
}