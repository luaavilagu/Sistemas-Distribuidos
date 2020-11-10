%%cu

#include <stdio.h>
#include <stdlib.h>

void initMatrixes(int **a, int **b, int **c, int **test, int N){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            a[i][j] = i * j;
            b[i][j] = i * j;
            c[i][j] = 0;
            test[i][j] = 0;
        }
    }
}

void printMatrix(int **mat, int N){
    printf("\n");
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            printf("\ni:%d - j:%d - mat[i][j]:%d\n", i, j, mat[i][j]);
        }
    }
    printf("\n");
}

void printMatrixes(int **a, int **b, int **c, int **test, int N){
    printf("\n----------------------- init a -----------------------\n");
    printMatrix(a, N);
    printf("----------------------- end a -----------------------\n");
    printf("----------------------- init b -----------------------\n");
    printMatrix(b, N);
    printf("----------------------- end b -----------------------\n");
    printf("----------------------- init c -----------------------\n");
    printMatrix(c, N);
    printf("----------------------- end c -----------------------\n");
    printf("----------------------- init test -----------------------\n");
    printMatrix(test, N);
    printf("----------------------- end test -----------------------\n");
}

void sumAB_h(int **a, int **b, int **c, int N){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            c[i][j] = a[i][j] + b[i][j];
        }
    }
}

__global__ void sumAB_d(int* a, size_t a_p, int* b, size_t b_p, int* c, size_t c_p, int N){
    
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    if ((tidy < N) && (tidx < N)){
        int* rowc = (int*)((char*)c + tidy * c_p);
        int* rowa = (int*)((char*)a + tidy * a_p);
        int* rowb = (int*)((char*)b + tidy * b_p);

        //rowc[tidx] = 55;
        rowc[tidx] = rowa[tidx] + rowb[tidx];
    }
}

int main(){

    //Tamagnos a ser usados
    int N = 4;
    size_t size = N * sizeof(int);
    int width = N;
    int height = N;

    //Instanciamos variables
    int **h_a, **h_b, **h_c, **h_test;
    int *d_a, *d_b, *d_c;
    int t[N][N];

    //Reservamos memoria para variables host
    h_a = (int**)malloc(N * sizeof(int*));
    h_b = (int**)malloc(N * sizeof(int*));
    h_c = (int**)malloc(N * sizeof(int*));
    h_test = (int**)malloc(N * sizeof(int*));
    for (int i = 0; i < N; i++){
        h_a[i] = (int*)malloc(size);
        h_b[i] = (int*)malloc(size);
        h_c[i] = (int*)malloc(size);
        h_test[i] = (int*)malloc(size);
    }
    
    //Reservamos memoria para variables device
    size_t pitch_a;
    cudaError_t err_pitch_a = cudaMallocPitch( reinterpret_cast<void **> (&d_a), &pitch_a, width * sizeof(int), height);
    if (err_pitch_a != cudaSuccess) printf("%s\n", cudaGetErrorString(err_pitch_a));
    size_t pitch_b;
    cudaError_t err_pitch_b = cudaMallocPitch( reinterpret_cast<void **> (&d_b), &pitch_b, width * sizeof(int), height);
    if (err_pitch_b != cudaSuccess) printf("%s\n", cudaGetErrorString(err_pitch_b));
    size_t pitch_c;
    cudaError_t err_pitch_c = cudaMallocPitch( reinterpret_cast<void **> (&d_c), &pitch_c, width * sizeof(int), height);
    if (err_pitch_c != cudaSuccess) printf("%s\n", cudaGetErrorString(err_pitch_c));
    
    //inicializacion de vectores a, b y c
    initMatrixes(h_a, h_b, h_c, h_test, N);
    
    //Copiamos vectores desde host a device
    cudaError_t err_memcpy_d_a = cudaMemcpy2D(d_a, pitch_a, h_a, N * sizeof(int), N * sizeof(int), N, cudaMemcpyHostToDevice);
    if (err_memcpy_d_a != cudaSuccess) printf("%s\n", cudaGetErrorString(err_memcpy_d_a));
    cudaError_t err_memcpy_d_b = cudaMemcpy2D(d_b, pitch_b, h_b, N * sizeof(int), N * sizeof(int), N, cudaMemcpyHostToDevice);
    if (err_memcpy_d_b != cudaSuccess) printf("%s\n", cudaGetErrorString(err_memcpy_d_b));
    cudaError_t err_memcpy_d_c = cudaMemcpy2D(d_c, pitch_c, h_c, N * sizeof(int), N * sizeof(int), N, cudaMemcpyHostToDevice);
    if (err_memcpy_d_c != cudaSuccess) printf("%s\n", cudaGetErrorString(err_memcpy_d_c));
    
    //Creación del kernel
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    sumAB_d<<< numBlocks, threadsPerBlock >>>(d_a, pitch_a, d_b, pitch_b, d_c, pitch_c, N);

    //Copiamos las matrices desde device a host
    cudaError_t err_memcpy_t = cudaMemcpy2D(t, N * sizeof(int), d_c, pitch_c, N * sizeof(int), N, cudaMemcpyDeviceToHost);
    if (err_memcpy_t != cudaSuccess) printf("%s\n", cudaGetErrorString(err_memcpy_t));

    cudaError_t err_memcpy_h_b = cudaMemcpy2D(h_b, N * sizeof(int), d_b, pitch_b, N * sizeof(int), N, cudaMemcpyDeviceToHost);
    if (err_memcpy_h_b != cudaSuccess) printf("%s\n", cudaGetErrorString(err_memcpy_h_b));

    cudaError_t err_memcpy_h_a = cudaMemcpy2D(h_a, N * sizeof(int), d_a, pitch_a, N * sizeof(int), N, cudaMemcpyDeviceToHost);
    if (err_memcpy_h_a != cudaSuccess) printf("%s\n", cudaGetErrorString(err_memcpy_h_a));

    //cudaError_t err_memcpy_h_c = cudaMemcpy2D(h_c, (N * N) * sizeof(int), d_c, pitch_c, N * sizeof(int), N, cudaMemcpyDeviceToHost);
    //if (err_memcpy_h_c != cudaSuccess) printf("%s\n", cudaGetErrorString(err_memcpy_h_c));

    //Calculo de la suma de forma secuencial
    sumAB_h(h_a, h_b, h_test, N);

    printf("---------------- inicio h_c ----------------");
    printMatrix(h_c, N);
    printf("---------------- fin h_c ----------------");
    //printMatrixes(h_a, h_b, h_c, h_test, N);
    
    //Imprimos resultado de CPU y GPU
    for (int i = 0 ; i < N; i++){
        for (int j = 0; j < N; j++){
            printf("\n i:%d     -     j:%d     -    h_c:%d    -   h_test:%d \n", i, j, t[i][j], h_test[i][j]);
        }
    }
    
    //Liberación memoria de host
    for (int i = 0; i < N; i++){
        free(h_a[i]);
        free(h_b[i]);
        free(h_c[i]);
    }

    //Liberación memoria de device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    printf("\n--------------------------    SALIO    --------------------------\n");
}