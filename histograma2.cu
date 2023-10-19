#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <cuda.h>
#include <curand_kernel.h>

#define N 50000000             //Numero de valores de entrada
#define M 8                     //Tamaño del histograma

#define REPETICIONES 20     //Repeticion de pruevas para calculo de media, max y min
#define SCALA 100              //Datos calculados en cada hilo

__device__ int vector_V[N];     //Vector de datos de entrada
__device__ int vector_H[M];     //Vector del histograma

/**
* Funcion para la comprovacion de errores cuda 
*/
static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
*   Kernel para inicializacion de datos de entrada
*/
__global__ void inicializa_v(int random, curandState *states, int threadsPerBlock, int blocksPerGrid){
    int iteraciones= SCALA;
    if(blocksPerGrid-1 == blockIdx.x && threadIdx.x == threadsPerBlock -1){
        iteraciones = iteraciones + (N % SCALA);
    }
    unsigned id_x = blockIdx.x*blockDim.x + threadIdx.x;
    curandState *state = states + id_x;

    curand_init(random, id_x, 0, state);
    for(int i = 0; i < iteraciones; i++){
        if(id_x*SCALA+i < N){
            vector_V[id_x*SCALA+i] = (int)((curand_uniform(state)*1000)) % M;
        }
    }

}

/**
*   Kernel para inicializacion del vector de histograma
*/
__global__ void inicializa_h(){
    unsigned id_x = blockIdx.x*blockDim.x + threadIdx.x;
    vector_H[id_x] = 0;
}
/**
*  Segunda version del histograma , con histogramas locales . Dividido en dos fases.
*/
/*
  cg::grid_group grid = cg::this_grid(); //creamos la estructura que refleja los hilos del grid

    int i = grid.thread_rank(); //Somos el hilo i dentro de todo el grid

    ti[i] = tini;

    if (i==0) { //El hilo 0 de todo el grid inicializa la suma
    	suma = 0.0;
    }

    //__syncthreads();
    grid.sync(); //Barrera para todos los hilos del grid
*/
__global__ void histograma(int threadsPerBlock, int blocksPerGrid){
    int vector[M];
    for(int i =0; i < M;i++){
        vector[i] =0;
    }
    int iteraciones= SCALA;
    if(blocksPerGrid-1 == blockIdx.x && threadIdx.x == threadsPerBlock -1){
       iteraciones = iteraciones + (N % SCALA);
    }
    unsigned id_x = blockIdx.x*blockDim.x + threadIdx.x;
    for(int i = 0; i < iteraciones; i++){
        if(id_x*SCALA+i < N){
            int mod = vector_V[id_x*SCALA+i]%M;
            vector[mod]++;
        }
    }
    for(int i =0; i < M;i++){
        int a =vector[i];
        atomicAdd(&vector_H[i],a);
    }
}



int main(){

    //valores aleatorios
    srand(time(NULL));
    static curandState *states = NULL;

   
    int h_v_h[M];
    int threadsPerBlock = 1024;
    int blocksPerGrid =((N/SCALA) + threadsPerBlock - 1) / threadsPerBlock;

    float t_duration[REPETICIONES];
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for(int j = 0; j< REPETICIONES; j++){
        CUDA_CHECK_RETURN(cudaEventRecord(start, 0));

        CUDA_CHECK_RETURN(cudaMalloc((void **)&states, sizeof(curandState) * threadsPerBlock  * blocksPerGrid));
        inicializa_v<<<blocksPerGrid, threadsPerBlock>>>(rand(),states, threadsPerBlock,blocksPerGrid);
        CUDA_CHECK_RETURN(cudaGetLastError());
        inicializa_h<<<1,M>>>();
        CUDA_CHECK_RETURN(cudaGetLastError());

        histograma<<<blocksPerGrid,threadsPerBlock>>>(threadsPerBlock,blocksPerGrid);
        CUDA_CHECK_RETURN(cudaGetLastError());

        
        CUDA_CHECK_RETURN(cudaMemcpyFromSymbol(h_v_h, vector_H, M*sizeof(int)));
        int acumula =0;
        for(int  i = 0; i<M; i++){
            std::cout<<h_v_h[i]<<" ";
            acumula += h_v_h[i];
        }
        std::cout<<"\n-------------------------"<<acumula<<"-----------------------------------\n";
        
        CUDA_CHECK_RETURN(cudaFree(states));
        CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
        CUDA_CHECK_RETURN(cudaEventSynchronize(stop));

        CUDA_CHECK_RETURN(cudaEventElapsedTime(&t_duration[j],start,stop));  
    }
    float t_max =0, t_min= FLT_MAX, media=0;
    for(int i = 0; i< REPETICIONES; i++){
        media +=t_duration[i];
        if(t_duration[i] > t_max){
            t_max =t_duration[i]; 
        }
        if(t_duration[i]< t_min){
            t_min= t_duration[i];
        }
    }
    std::cout<< "Se han realizado "<<REPETICIONES<<" repeticiones\n";
    std::cout<<"Obteniendo de media: "<<media/REPETICIONES<<"ms \n";
    std::cout<<"Y de máximo: "<<t_max<<"ms  y mínimo: "<<t_min<<"ms\n";

    return 0;
}



/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {

	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (EXIT_FAILURE);
}