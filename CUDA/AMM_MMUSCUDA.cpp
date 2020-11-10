%%cu

#include <stdio.h>
#include <stdlib.h>

void initMatrixes(int *a, int *b, int *c, int *test, int N){
    int count = 8;
    for (int i = 0; i < N * N; i++){
        *(a + i) = i;
        *(b + i) = count;
        *(c + i) = 0;
        *(test + i) = 0;
        count--;
    }
}

void multAB_h(int *a, int *b, int *c, int N){
    for (int t = 0; t < N; t++){
        for (int j = 0; j < N; j++){
            for (int k = 0; k < N; k++){
                int va = *( a + ((j * N) + t));
                int vb = *( b + ((t * N) + k));
                int vc = *( c + ((j * N) + k));
                *(c + ((j * N) + k)) = ((va * vb) + vc);
            }
        }
    }
}

void printMatrix(int *mat, int N){
    printf("\n");
    for (int i = 0; i < N * N; i++){
        if ( i % N == 0) printf("\n");
        printf("%d\t", *(mat + i));
    }
    printf("\n");
}

void printLinearMatrix(int *mat, int N){
    printf("\n");
    for (int i = 0; i < N * N; i++){
        printf("%d\t", *(mat + i));
    }
    printf("\n");
}

__global__ void multAB_d(int *a, int *b, int *c, int N){
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    if ( (tidx < N) && (tidy < N) ){
        for (int k = 0; k < N; k++){
            int va = *( a + (( tidx * N ) + k ));
            int vb = *( b + (( k * N ) + tidy ));
            int vc = *( c + (( tidx * N ) + tidy ));
            *(c + (( tidx * N) + tidy )) = ((va * vb) + vc);
        }
    }
}

int main(){

    //Tamagnos a ser usados
    int N = 10;
    size_t size = N * sizeof(int);

    //Instanciamos variables
    int *h_a, *h_b, *h_c, *h_test;
    int *d_a, *d_b, *d_c;

    //Reservamos memoria para variables host
    h_a = (int*)malloc(N * N * sizeof(int));
    h_b = (int*)malloc(N * N * sizeof(int));
    h_c = (int*)malloc(N * N * sizeof(int));
    h_test = (int*)malloc(N * N * sizeof(int));
    
    //*
    //Reservamos memoria para variables device
    cudaMalloc(&d_a, N * N * size);
    cudaMalloc(&d_b, N * N * size);
    cudaMalloc(&d_c, N * N * size);
    //*/

    //inicializacion de vectores a, b y c
    initMatrixes(h_a, h_b, h_c, h_test, N);

    //*
    //Copiamos vectores desde host a device
    cudaMemcpy(d_a, h_a, N * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, N * size, cudaMemcpyHostToDevice);
    //*/

    //Creación del kernel
    dim3 threadsPerBlock (N, N);
    multAB_d<<< 1, threadsPerBlock >>> (d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, N * size, cudaMemcpyDeviceToHost);

    //Comprobación resultado
    multAB_h(h_a, h_b, h_test, N);
    
    //Imprimos resultado
    //printf("-------------------------------- h_a --------------------------------");
    //printMatrix(h_a, N);
    //printf("-------------------------------- h_b --------------------------------");
    //printMatrix(h_b, N);

    printf("-------------------------------- h_c --------------------------------");
    printMatrix(h_c, N);
    printf("------------------------------- h_test -------------------------------");
    printMatrix(h_test, N);

    //printf("---------------------------- Linear h_test ----------------------------");
    //printLinearMatrix(h_test, N);

    //Liberación memoria de host
    free(h_a);
    free(h_b);
    free(h_c);
    
    //*
    //Liberación memoria de device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    //*/
}
