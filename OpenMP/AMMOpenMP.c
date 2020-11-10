/* Programa: Approximate Matrix Multiplication (AMM) OpenMP */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#include "omp.h"

#define ADIMX 1024
#define DIM 1024
#define BDIMY 1024

#define THREADS 16
#define TIMES 10

float matA[ADIMX][DIM], matB[DIM][BDIMY], result[ADIMX][BDIMY];

float pk = 1.0/DIM;
float pkt = 0.0;

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

//funciona
void multiplyMatrixUniformSampling(){

    int s = rand() % DIM;

    for (int t = 0; t < s; t++){

        int it = rand() % DIM;
        for (int row = 0; row < ADIMX; row++){
            for (int col = 0; col < BDIMY; col++)
            {
                result[row][col] += ((matA[row][it])/(sqrt(s*pk))) * ((matB[it][col])/(sqrt(s*pk)));
            } 
        }
    }
}

int pktCalculateA(int t){
    int sumA = 0;
    for (int row = 0; row < ADIMX; row++){
        sumA += matA[row][t];
    }
    return sumA;
}

int pktCalculateB(int t){
    int sumB = 0;
    for (int col = 0; col < BDIMY; col++){
        sumB += matA[t][col];
    }
    return sumB;
}

int pktCalculateAB(){
    int sumAB = 0;
    for (int t = 0; t < DIM; t++){
        sumAB += pktCalculateA(t) * pktCalculateB(t);
    }
    return sumAB;
}

float pktCalculate(int t)
{
    
    float a = pktCalculateA(t) * 1.0;
    float b = pktCalculateB(t) * 1.0;
    float ab = pktCalculateAB(t) * 1.0;

    return ( a * b ) / ab;
}

void multiplyMatrixSimpleRandomized(){

    int s = rand() % DIM;
    //printf("\ns = %d\n", s);

    for (int t = 0; t < s; t++){

        int it = rand() % DIM;
        //printf("\nit = %d - t = %d\n", it, t);
        pkt = pktCalculate(it);
        //printf("\nit = %d - t = %d - pkt = %f\n", it, t, pkt);

        for (int row = 0; row < ADIMX; row++){
            for (int col = 0; col < BDIMY; col++)
            {
                result[row][col] += ((matA[row][it])/(sqrt(s*pkt))) * ((matB[it][col])/(sqrt(s*pkt)));
                //printf(" s = %d -  pk = %f -s*pk = %lf \n", s, pk, s*pk);
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

void multiplyMatrixSubset(){

    int s1 = rand() % DIM;
    int s2 = rand() % DIM;

    printf("\ns1 = %d     -   s2 = %d\n", s1, s2);

    if (s1 > s2){
        int temp = s1;
        s1 = s2;
        s2 = temp;
    } 

    for (int t = s1; t < s2; t++){
        for (int row = 0; row < ADIMX; row++){
            for (int col = 0; col < BDIMY; col++)
            {
                result[row][col] += matA[row][t] * matB[t][col];
                //printf("t = %d  -  row = %d  -  col = %d  - %d * %d = %d\n", t, row, col, matA[row][t], matB[t][col], result[row][col]);
            } 
        }
    }
}

int main()
{
    struct timeval tval_before, tval_after, tval_result;

    time_t tt;
    srand((unsigned) time(&tt));
    //srand(200);

    initializeMatrixes();
    //printMatrixes();

    /*
    //Uniform Sampling  -   funcional
    printf("\n-------------------- Begin Uniform Sampling\n");
    printf("\npk: %f\n", pk);
    multiplyMatrixUniformSampling();
    printf("\nResult US\n");
    //printMatrix(ADIMX, BDIMY, result);
    */

    
    //Simple Randomized
    printf("\n-------------------- Begin Simple Randomized\n");
    initializeMatrix0(ADIMX, BDIMY, result);
    //printf("pk: %f\n", pk);
    gettimeofday(&tval_before, NULL);
    multiplyMatrixSimpleRandomized();
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    printf("\nResult SR\n");
    printf("%ld,%06ld\n", (long int) tval_result.tv_sec, (long int) tval_result.tv_usec);
    
    //printMatrix(ADIMX, BDIMY, result);
    

    /*
    
    //Subset
    printf("\n-------------------- Begin Subset\n");
    initializeMatrix0(ADIMX, BDIMY, result);
    multiplyMatrixSubset();
    printf("\nResult s1-s2\n");
    printMatrix(ADIMX, BDIMY, result);
    */

    //Real value    -   funcional
    printf("\n-------------------- Begin Real value\n");
    initializeMatrix0(ADIMX, BDIMY, result);
    gettimeofday(&tval_before, NULL);
    multiplyMatrix();
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    printf("\nResult AB\n");
    printf("%ld,%06ld\n", (long int) tval_result.tv_sec, (long int) tval_result.tv_usec);
    //printMatrix(ADIMX, BDIMY, result);

    return 0;
}