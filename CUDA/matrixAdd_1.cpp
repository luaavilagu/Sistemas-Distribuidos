%%cu

#include <stdio.h>
#include <stdlib.h>

void initMatrixes(int *a, int *b, int *c, int *test, int N){
    int count = 15;
    for (int i = 0; i < N * N; i++){
        *(a + i) = i;
        *(b + i) = count;
        *(c + i) = 0;
        //*(test + i) = 0;
        count--;
    }
}

void sumAB_h(int *a, int *b, int *c, int N){
    for (int i = 0; i < N * N; i++){
        *(c + i) = *(a + i) + *(b + i);
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

__global__ void sumAB_d(int *a, int *b, int *c, int N){
    int tidx = threadIdx.x;

    if ((tidx < N * N)){
        *(c + tidx) = *(a + tidx) + *(b + tidx);
    }
}


int main(){

    //Tamagnos a ser usados
    int N = 4;
    size_t size = N * sizeof(int);

    //Instanciamos variables
    int *h_a, *h_b, *h_c, *h_test;
    int *d_a, *d_b, *d_c;

    //Reservamos memoria para variables host
    h_a = (int*)malloc(N * N * sizeof(int));
    h_b = (int*)malloc(N * N * sizeof(int));
    h_c = (int*)malloc(N * N * sizeof(int));
    h_test = (int*)malloc(N * N * sizeof(int));
    
    
    //Reservamos memoria para variables device
    cudaMalloc(&d_a, N * N * size);
    cudaMalloc(&d_b, N * N * size);
    cudaMalloc(&d_c, N * N * size);
    

    //inicializacion de vectores a, b y c
    initMatrixes(h_a, h_b, h_c, h_test, N);

    
    //Copiamos vectores desde host a device
    cudaMemcpy(d_a, h_a, N * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, N * size, cudaMemcpyHostToDevice);
    

    //Creación del kernel
    sumAB_d<<< 1, N * N >>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, N * size, cudaMemcpyDeviceToHost);

    //Comprobación resultado
    sumAB_h(h_a, h_b, h_test, N);
    
    //Imprimos resultado
    printf("-------------------------------- h_a --------------------------------");
    printMatrix(h_a, N);
    printf("-------------------------------- h_b --------------------------------");
    printMatrix(h_b, N);

    printf("-------------------------------- h_c --------------------------------");
    printMatrix(h_c, N);
    printf("------------------------------- h_test -------------------------------");
    printMatrix(h_test, N);

    //Liberación memoria de host
    free(h_a);
    free(h_b);
    free(h_c);
    
    //Liberación memoria de device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
}
