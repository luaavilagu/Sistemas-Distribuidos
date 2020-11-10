//Suma de dos vectores en uno tercero usando sólo un bloque con la cantidad de hilos

%%cu

#include <stdio.h>
#include <stdlib.h>

void initVectors(int *a, int *b, int *c, int *test, int N){
    for (int i = 0; i < N; i++){
        a[i] = i;
        b[i] = i;
        c[i] = 0;
        test[i] = 0;
    }
}

__global__ void sumAB_d(int *a, int *b, int *c, int N){
    int x = threadIdx.x;
    if (x < N){
        c[x] = a[x] + b[x];
    }
}


void sumAB_h(int *a, int *b, int *c, int N){
    for (int i = 0; i < N; i++){
        c[i] = a[i] + b[i];
    }
}


int main(){

    //Tamagnos a ser usados
    int N = 100;
    size_t size = N * sizeof(int);

    //Inicializamos variables
    int *h_a, *h_b, *h_c, *h_test;
    int *d_a, *d_b, *d_c;

    //Reservamos memoria para variables host
    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);
    h_test = (int*)malloc(size);
    
    //Reservamos memoria para variables device
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    //inicializacion de vectores a, b y c
    initVectors(h_a, h_b, h_c, h_test, N);

    //Copiamos vectores desde host a device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, size, cudaMemcpyHostToDevice);

    //Creación del kernel
    sumAB_d<<<1,N>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    //Comprobación resultado
    sumAB_h(h_a, h_b, h_test, N);

    //Imprimos resultado
    for (int i = 0 ; i < N; i++){
        printf("\n i:%d     -    h_c:%d    -   h_test:%d \n",i, h_c[i], h_test[i]);
    }
    

    //Liberación memoria de host
    free(h_a);
    free(h_b);
    free(h_c);

    //Liberación memoria de device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}
