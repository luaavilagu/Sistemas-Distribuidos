/* Programa: Approximate Matrix Multiplication (AMM) sequential - Uniform sampling  */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#include "omp.h"

#define ADIMX 1024*2
#define DIM 1024*2
#define BDIMY 1024*2

#define THREADS 16

float matA[ADIMX][DIM], matB[DIM][BDIMY], result[ADIMX][BDIMY];

float pk[DIM];

void initializeMatrix(int dimy, int dimx, float mat[dimy][dimx])
{
    for (int x = 0; x < dimy; x++){
        for (int y = 0; y < dimx; y++){
            mat[x][y] = rand() % 20;
        }
    }
}

void initializeMatrix0(int dimy, int dimx, float mat[dimy][dimx])
{
    for (int x = 0; x < dimy; x++){
        for (int y = 0; y < dimx; y++){
            mat[x][y] = 0;
        }
    }
}

void printMatrix (int dimy, int dimx, float mat[dimy][dimx])
{
    for (int y = 0; y < dimy; y++){
        printf("\n");
        for (int x = 0; x < dimx; x++){
            printf("%f  ", mat[y][x]);
        }
    }
    printf("\n");
}

void pkt(){

    int sumAB = 0;
    #pragma omp parallel for reduction (+:sumAB) num_threads(THREADS)
        for (int t = 0; t < DIM; t++){
            int sumA = 0;
            for (int row = 0; row < ADIMX; row++){
                sumA += matA[row][t];
            }
            int sumB = 0;
            for (int col = 0; col < BDIMY; col++){
                sumB += matB[t][col];
            }
            pk[t] = sumA * sumB;
            sumAB += sumA * sumB;
        }
    

    #pragma omp parallel for
        for (int t = 0; t < DIM; t++){
            pk[t] /= sumAB;
        }

}

void multiplyMatrixSimpleRandomized(int s){

    #pragma omp parallel for num_threads(THREADS)
        for (int t = 0; t < s; t++){
            int it = rand() % DIM;
            for (int row = 0; row < ADIMX; row++){
                for (int col = 0; col < BDIMY; col++)
                {
                    result[row][col] += ((matA[row][it])/(sqrt(s*pk[t]))) * ((matB[it][col])/(sqrt(s*pk[t])));
                } 
            }
        }
    
}

void multiplyMatrix(){
    for (int t = 0; t < DIM; t++){
        for (int row = 0; row < ADIMX; row++){
            for (int col = 0; col < BDIMY; col++)
            {
                result[row][col] += matA[row][t] * matB[t][col];
            } 
        }
    }
}

void initializeMatrixes(){
    initializeMatrix(ADIMX, DIM, matA);
    initializeMatrix(DIM, BDIMY, matB);
    initializeMatrix0(ADIMX, BDIMY, result);
}

void printMatrixes(){
    printf("\n Matrix A \n");
    printMatrix(ADIMX, DIM, matA);
    printf("\n Matrix B \n");
    printMatrix(DIM, BDIMY, matB);
    printf("\n Matrix Result \n");
    printMatrix(ADIMX, BDIMY, result);
}

int main()
{
    struct timeval tval_before, tval_after, tval_result, tval_beforeF, tval_afterF, tval_resultF;

    //time_t tt;
    //srand((unsigned) time(&tt));
    srand(200);

    initializeMatrixes();
    //printMatrixes();
    
    //Simple Randomized  -   funcional
    printf("\n-------------------- Begin Simple Randomized\n");
    initializeMatrix0(ADIMX, BDIMY, result);
    gettimeofday(&tval_before, NULL);
    int s = rand() % DIM;
    pkt();
    multiplyMatrixSimpleRandomized(s);
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    printf("\nResult SR - s = %d\n", s);
    printf("%ld,%06ld\n", (long int) tval_result.tv_sec, (long int) tval_result.tv_usec);
    //printMatrix(ADIMX, BDIMY, result);
    
    /*
    //Real value    -   funcional
    printf("\n-------------------- Begin Real value\n");
    initializeMatrix0(ADIMX, BDIMY, result);
    gettimeofday(&tval_beforeF, NULL);
    multiplyMatrix();
    gettimeofday(&tval_afterF, NULL);
    timersub(&tval_afterF, &tval_beforeF, &tval_resultF);
    printf("\nResult AB\n");
    printf("%ld,%06ld\n", (long int) tval_resultF.tv_sec, (long int) tval_resultF.tv_usec);
    //printMatrix(ADIMX, BDIMY, result);
    */

    return 0;
}