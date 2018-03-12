#include <vector>
#include <iostream>
#include <algorithm>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

#define WARP_SZ 32
#define MAX_WARPS 64
#define SSSP_INF 1073741824


/*
 * Segmented Scan Device Code for finding the minimum of the values array in a warp.
*/
__device__ int segmentedScan_min( int lane , int *rows, int *values ) {

    if( lane >= 1 && ( rows[threadIdx.x] == rows[threadIdx.x - 1] ) ) {
        values[threadIdx.x] = values[threadIdx.x] < values[threadIdx.x - 1] ? values[threadIdx.x] : values[threadIdx.x-1];
    }

    if( lane >= 2 && ( rows[threadIdx.x] == rows[threadIdx.x - 2] ) ) {
        values[threadIdx.x] = values[threadIdx.x] < values[threadIdx.x - 2] ? values[threadIdx.x] : values[threadIdx.x-2];
    }

    if( lane >= 4 && ( rows[threadIdx.x] == rows[threadIdx.x - 4] ) ) {
        values[threadIdx.x] = values[threadIdx.x] < values[threadIdx.x - 4] ? values[threadIdx.x] : values[threadIdx.x-4];
    }

    if( lane >= 8 && ( rows[threadIdx.x] == rows[threadIdx.x - 8] ) ) {
        values[threadIdx.x] = values[threadIdx.x] < values[threadIdx.x - 8] ? values[threadIdx.x] : values[threadIdx.x-8];
    }

    if( lane >= 16 && ( rows[threadIdx.x] == rows[threadIdx.x - 16] ) ) {
        values[threadIdx.x] = values[threadIdx.x] < values[threadIdx.x - 16] ? values[threadIdx.x] : values[threadIdx.x-16];
    }

    return values[threadIdx.x];
}


/*
 * Shared Memory implementation, outcore method.
 * Since the maximum number of threads in the block is 1024.
 * We use the size of the shared memory of size 1024.
*/
__global__ void parallelBMF_sharedMemory(edge_list *edgeList, int length, int *distance_prev, int *distance_current, int *anyChange) {

    __shared__ int distance[1024];
    __shared__ int dest[1024];

    int totalThreads = (blockDim.x*gridDim.x);
    int totalWarps = totalThreads / 32;
    int globalThreadId = threadIdx.x + (blockIdx.x * blockDim.x);
    int warpId = globalThreadId / 32;

    int blockWarpId = threadIdx.x >> 5;
    int warpFirst = blockWarpId << 5;
    int warpLast = warpFirst + 31;

    int load = length % totalWarps == 0 ? length/totalWarps : length/totalWarps + 1;
    int beginning = load * warpId;
    int end = (beginning+load) > length ? length : beginning + load;
    beginning  = beginning + lane_id();

    for( int i = beginning ; i < end ; i += 32 ) {
        int source = edgeList[i].srcIndex;
        int destination = edgeList[i].destIndex;
        int weight = edgeList[i].weight;

        distance[threadIdx.x] = distance_prev[source] + weight;
        dest[threadIdx.x] = destination;
        __syncthreads();

        int minimum = segmentedScan_min(lane_id(),dest,distance);

        __syncthreads();

        if( i == end - 1 ) {
            if( distance_current[destination] > minimum)
                *anyChange = 1;
            atomicMin(&distance_current[destination],minimum);
        }
        else if ( threadIdx.x != warpLast ) {
            if( dest[threadIdx.x] != dest[threadIdx.x+1] ) {
                if( distance_current[destination] > minimum)
                    *anyChange = 1;
                atomicMin(&distance_current[destination],minimum);
            }
        }
        else {
            if( distance_current[destination] > minimum)
                *anyChange = 1;
            atomicMin(&distance_current[destination],minimum);
        }
    }
}


__global__ void parallelBMF_incore_noSharedMemory(edge_list *edgeList, int length , int *distance , int *anyChange ) {

    int totalThreads = (blockDim.x*gridDim.x);
    int totalWarps = totalThreads / 32;
    int globalThreadId = threadIdx.x + (blockIdx.x * blockDim.x);
    int warpId = globalThreadId / 32;

    int load = length % totalWarps == 0 ? length/totalWarps : length/totalWarps + 1;
    int beginning = load * warpId;
    int end = (beginning+load) > length ? length : beginning + load;
    beginning  = beginning + lane_id();

    for( int i = beginning ; i < end ; i += 32 ) {
        int source = edgeList[i].srcIndex;
        int destination = edgeList[i].destIndex;
        int weight = edgeList[i].weight;
        int temp_distance = distance[source] + weight;
        if( temp_distance < distance[destination]) {
            atomicMin(&distance[destination],distance[source]+weight);
            *anyChange = 1;
        }
    }
}


__global__ void parallelBMF_outcore_noSharedMemory(edge_list *edgeList, int length, int *distance_prev, int *distance_current, int *anyChange){

    int totalThreads = (blockDim.x*gridDim.x);
    int totalWarps = totalThreads / 32;
    int globalThreadId = threadIdx.x + (blockIdx.x * blockDim.x);
    int warpId = globalThreadId / 32;

    int load = length % totalWarps == 0 ? length/totalWarps : length/totalWarps + 1;
    int beginning = load * warpId;
    int end = (beginning+load) > length ? length : beginning + load;
    beginning  = beginning + lane_id();

    for( int i = beginning ; i < end ; i += 32 ) {
        int source = edgeList[i].srcIndex;
        int destination = edgeList[i].destIndex;
        int weight = edgeList[i].weight;
        if( distance_prev[source] + weight < distance_current[destination]) {
            atomicMin(&distance_current[destination],distance_prev[source]+weight);
            *anyChange = 1;
        }
    }
}

void puller(std::vector<initial_vertex> * parsedGraph, std::vector<edge_list> *edgeList,int blockSize, int blockNum,int edgeListOrder, int syncMethod, int smemMethod){

    /* For Outcore Method and No Shared Memory Method */
    if(syncMethod == 1 && smemMethod == 1) {

        setTime();
        /* Size of vertices and edges of the graph */
        uint vertices = parsedGraph->size();
        uint edges = edgeList->size();
        /* Edge Lists */
        edge_list *device_edgeList;
        edge_list *temp_edgeList;
        temp_edgeList = (edge_list*)malloc(sizeof(edge_list)*edges);
        /* Distance Arrays */
        int *distance_current = (int*)malloc(sizeof(int)*vertices);
        int *distance_prev = (int*)malloc(sizeof(int)*vertices);
        int *anyChange = (int *)malloc(sizeof(int));;
        int *device_distance_prev;
        int *device_distance_current;
        int *device_anyChange;


        /* Setting all vertices distances to infinite except the source vertex */
        std::fill_n(distance_current,vertices,SSSP_INF);
        distance_current[0] = 0;
        std::fill_n(distance_prev,vertices,SSSP_INF);
        distance_prev[0] = 0;
        std::copy(edgeList->begin(),edgeList->end(),temp_edgeList);
        /* Allocate and Copy the data from the host to the device */
        cudaMalloc((void **)&device_edgeList,sizeof(edge_list)*edges);
        cudaMalloc((void **)&device_distance_prev, sizeof(int)*vertices );
        cudaMalloc((void **)&device_distance_current, sizeof(int)*vertices);
        cudaMalloc((void **)&device_anyChange,sizeof(int));
        cudaMemcpy(device_distance_current,distance_current,sizeof(int)*vertices,cudaMemcpyHostToDevice);
        cudaMemcpy(device_distance_prev,distance_prev,sizeof(int)*vertices,cudaMemcpyHostToDevice);
        cudaMemcpy(device_edgeList,temp_edgeList,sizeof(edge_list)*edges,cudaMemcpyHostToDevice);

        /* Measure the time taken by the kernel */
        double kernelTimeTaken = 0;
        /* Bellman-Ford Parallel Algorithm */
        for(uint i = 0 ; i < vertices ; i++) {

            *anyChange = 0;
            cudaMemcpy(device_anyChange,anyChange,sizeof(int),cudaMemcpyHostToDevice);

            /* Kernel Call and measure the time taken by the kernel */
            double kernel_time_start = getTime();
            parallelBMF_outcore_noSharedMemory<<< blockNum , blockSize >>>(device_edgeList,edges,device_distance_prev,device_distance_current,device_anyChange);
            cudaDeviceSynchronize();
            double kernel_time_end = getTime();
            kernelTimeTaken += (kernel_time_end - kernel_time_start);

            cudaMemcpy(anyChange,device_anyChange,sizeof(int),cudaMemcpyDeviceToHost);
            if( *anyChange == 0 ) {
                std::cout << "Parallel Implementation Iteration breaks at : " << i << " .\n";
                break;
            }
            /* Swap distance_prev and distance_current arrays */
            int *temp = device_distance_current;
            device_distance_current = device_distance_prev;
            device_distance_prev = temp;
        }
        cudaMemcpy(distance_current,device_distance_current,sizeof(int)*vertices,cudaMemcpyDeviceToHost);
        /* Copy the distance back into the parsed graph instance */
        for( int i = 0 ; i < vertices ; i++ ) {
            parsedGraph->at(i).vertexValue.distance = distance_current[i];
        }

        /* Free all the allocation of the data on the device */
        cudaFree(device_edgeList);
        cudaFree(device_distance_current);
        cudaFree(device_distance_prev);
        cudaFree(device_anyChange);
        std::cout << "Took Total Time Taken is :" << getTime() << "ms.\n";
        std::cout << "Took Kernel Time Taken Outcore No Shared Memory is : " << kernelTimeTaken << "ms.\n";

    }
    /* For incore method and no shared memory */
    else if(syncMethod == 0 && smemMethod == 1){

        setTime();
        /* Size of vertices and edges of the graph */
        uint vertices = parsedGraph->size();
        uint edges = edgeList->size();
        /* Edge List definition */
        edge_list *device_edgeList;
        edge_list *temp_edgeList;
        temp_edgeList = (edge_list*)malloc(sizeof(edge_list)*edges);
        /* Distance Arrays */
        int *distance = (int*)malloc(sizeof(int)*vertices);;
        int *anyChange = (int *)malloc(sizeof(int));;
        int *device_distance;
        int *device_anyChange;

        /* Setting all vertices distances to infinite except the source vertex */
        std::fill_n(distance,vertices,SSSP_INF);
        distance[0] = 0;
        std::copy(edgeList->begin(),edgeList->end(),temp_edgeList);
        /* Allocation and copying of data from host to the device */
        cudaMalloc((void **)&device_edgeList,sizeof(edge_list)*edges);
        cudaMalloc((void **)&device_distance, sizeof(int)*vertices);
        cudaMalloc((void **)&device_anyChange,sizeof(int));
        cudaMemcpy(device_distance,distance,sizeof(int)*vertices,cudaMemcpyHostToDevice);
        cudaMemcpy(device_edgeList,temp_edgeList,sizeof(edge_list)*edges,cudaMemcpyHostToDevice);

        double kernelTimeTaken = 0;
        for(uint i = 0 ; i < vertices ; i++) {

            *anyChange = 0;
            cudaMemcpy(device_anyChange,anyChange,sizeof(int),cudaMemcpyHostToDevice);

            /* Kernel call and measure the time taken by the kernel */
            double kernel_time_start = getTime();
            parallelBMF_incore_noSharedMemory<<< blockNum , blockSize >>>(device_edgeList,edges,device_distance,device_anyChange);
            cudaDeviceSynchronize();
            double kernel_time_end = getTime();
            kernelTimeTaken += (kernel_time_end - kernel_time_start);

            cudaMemcpy(anyChange,device_anyChange,sizeof(int),cudaMemcpyDeviceToHost);
            if( *anyChange == 0 ) {
                std::cout << "Parallel Implementation Iteration breaks at : " << i << " .\n";
                break;
            }
        }
        cudaMemcpy(distance,device_distance,sizeof(int)*vertices,cudaMemcpyDeviceToHost);
        /* Copy the distance into the parsed graph structure */
        for( int i = 0 ; i < vertices ; i++ ) {
            parsedGraph->at(i).vertexValue.distance = distance[i];
        }

        /* Free the memory allocated */
        cudaFree(device_edgeList);
        cudaFree(device_distance);
        cudaFree(device_anyChange);
        std::cout << "Took Total Time : " << getTime() << "ms.\n";
        std::cout << "Took Kernel Time Incore No Shared Memory : " << kernelTimeTaken << "ms.\n";
    }
    /* Destination sorted edge list, outcore method and shared memory */
    else if( syncMethod == 1 && smemMethod == 0 && edgeListOrder == 2 ){

        setTime();
        /* Size of vertices and edges of the graph */
        uint vertices = parsedGraph->size();
        uint edges = edgeList->size();
        /* Edge Lists */
        edge_list *device_edgeList;
        edge_list *temp_edgeList;
        temp_edgeList = (edge_list*)malloc(sizeof(edge_list)*edges);
        /* Distance Arrays */
        int *distance_current = (int*)malloc(sizeof(int)*vertices);;
        int *distance_prev = (int*)malloc(sizeof(int)*vertices);;
        int *anyChange = (int *)malloc(sizeof(int));;
        int *device_distance_prev;
        int *device_distance_current;
        int *device_anyChange;


        /* Setting all vertices distances to infinite except the source vertex */
        std::fill_n(distance_current,vertices,SSSP_INF);
        distance_current[0] = 0;
        std::fill_n(distance_prev,vertices,SSSP_INF);
        distance_prev[0] = 0;
        std::copy(edgeList->begin(),edgeList->end(),temp_edgeList);
        /* Allocating and copying the data from the host to the device */
        cudaMalloc((void **)&device_edgeList,sizeof(edge_list)*edges);
        cudaMalloc((void **)&device_distance_prev, sizeof(int)*vertices );
        cudaMalloc((void **)&device_distance_current, sizeof(int)*vertices);
        cudaMalloc((void **)&device_anyChange,sizeof(int));
        cudaMemcpy(device_distance_current,distance_current,sizeof(int)*vertices,cudaMemcpyHostToDevice);
        cudaMemcpy(device_distance_prev,distance_prev,sizeof(int)*vertices,cudaMemcpyHostToDevice);
        cudaMemcpy(device_edgeList,temp_edgeList,sizeof(edge_list)*edges,cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        double kernelTimeTaken = 0;
        for(uint i = 0 ; i < vertices ; i++) {

            *anyChange = 0;
            cudaMemcpy(device_anyChange,anyChange,sizeof(int),cudaMemcpyHostToDevice);

            double kernel_time_start = getTime();
            parallelBMF_sharedMemory<<< blockNum , blockSize >>>(device_edgeList,edges,device_distance_prev,device_distance_current,device_anyChange);
            cudaDeviceSynchronize();
            double kernel_time_end = getTime();
            kernelTimeTaken += (kernel_time_end - kernel_time_start);


            cudaMemcpy(anyChange,device_anyChange,sizeof(int),cudaMemcpyDeviceToHost);
            if( *anyChange == 0 ) {
                std::cout << "Parallel Implementation Iteration breaks at : " << i << " .\n";
                break;
            }
            /* Swap the distance arrays */
            int *temp = device_distance_current;
            device_distance_current = device_distance_prev;
            device_distance_prev = temp;
        }

        cudaMemcpy(distance_current,device_distance_current,sizeof(int)*vertices,cudaMemcpyDeviceToHost);
        /* Copy the distance array into the parsed graph structure */
        for( int i = 0 ; i < vertices ; i++ ) {
            parsedGraph->at(i).vertexValue.distance = distance_current[i];
        }
        /* Free the allocated memory */
        cudaFree(device_edgeList);
        cudaFree(device_distance_current);
        cudaFree(device_distance_prev);
        cudaFree(device_anyChange);

        std::cout << "Took Total Time Taken :" << getTime() << "ms.\n";
        std::cout << "Took Kernel Time Taken Outcore Shared Memory : " << kernelTimeTaken << "ms.\n";
    }
    /* None of the input configurations matter */
    else {
        std::cout << "Not A correct Configuration for parallelization.\n";
    }
}
