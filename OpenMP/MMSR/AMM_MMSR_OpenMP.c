/* Programa: Approximate Matrix Multiplication (AMM) sequential - Uniform sampling  */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#include "omp.h"

#define ADIM 2048//2
#define DIM 1024//8
#define BDIM 2048//4

#define THREADS 16

float *pk;

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

void printMatrix (int dimy, int dimx, float *mat)
{
    printf("\n");
    for (int i = 0; i < dimy * dimx; i++){
        if ( i % dimx == 0) printf("\n");
        printf("%f  ", *(mat + i));
    }
    printf("\n");
}

void printPkt ()
{
    //float sum = 0.0;
    printf("\n");
    for (int i = 0; i <  DIM; i++){
        if ( i % DIM == 0) printf("\n");
        printf("%f  ", *(pk + i));
        //sum += *(pk + i);
    }
    //printf("sum:%f \n", sum);
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

void sampleAB(float *matS, float *matA, float *matR, float *matB, int s){
    for (int t = 0; t < s; t++ ){
        int it = rand() % DIM;
        //printf("it: %d\t", it);
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

void multiplyMatrix(float *matS, float *matR, float *res, int s){

    #pragma omp parallel for num_threads(THREADS)
    for (int row = 0; row < ADIM; row++){
        //#pragma omp parallel for num_threads(THREADS)
        for (int col = 0; col < BDIM; col++){
            //#pragma omp parallel for num_threads(THREADS)
            for (int t = 0; t < s; t++){
                float va = *( matS + (( row * s ) + t));
                float vb = *( matR + (( t * BDIM ) + col));
                float vc = *( res + (( row * BDIM ) + col));
                *(res + ((row * BDIM) + col)) = ((va * vb) + vc);
            }
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

    sampleAB(matS, matA, matR, matB, s);
    //printf("matS");
    //printMatrix(ADIM, s, matS);
    //printf("matR");
    //printMatrix(s, BDIM, matR);
    multiplyMatrix(matS, matR, res, s);

    free(matS);
    free(matR);
}


void initializeMatrixes(float* matA, float* matB, float* res){
    initializeMatrix(ADIM, DIM, matA);
    initializeMatrix(DIM, BDIM, matB);
    initializeMatrix0(ADIM, BDIM, res);
}

void beginUniformSampling(){
    struct timeval tval_before, tval_after, tval_result;

    float *matA, *matB, *res;

    matA = (float*)malloc(ADIM * DIM * sizeof(float));
    matB = (float*)malloc(DIM * BDIM * sizeof(float));
    res = (float*)malloc(ADIM * BDIM * sizeof(float));

    pk = (float*)malloc( DIM * sizeof(float));

    if (matA == NULL || matB == NULL || res == NULL || pk == NULL) printf("Error: if (matA == NULL || matB == NULL || res == NULL )");
    
    srand(200);

    initializeMatrixes(matA, matB, res);
    
    //printf("\n Matrix A \n");
    //printMatrix(ADIM, DIM, matA);
    //printf("\n Matrix B \n");
    //printMatrix(DIM, BDIM, matB);
    
    printf("\n Begin Sample Randomized\n");
    gettimeofday(&tval_before, NULL);
    //printf("pkt\n");
    pkt(matA, matB);
    //printPkt();
    int s = rand() % DIM;
    multiplyMatrixUniformSampling(matA, matB, res, s);
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    printf("\ns: %d\n", s);
    //printf("\nTIME\n");
    printf("%ld,%06ld\n", (long int) tval_result.tv_sec, (long int) tval_result.tv_usec);
    //printf("\nResult US");
    //printMatrix(ADIM, BDIM, res);

    free(matA);
    free(matB);
    free(res);
    free(pk);
    
}

int main()
{
    
    beginUniformSampling();

    return 0;
}