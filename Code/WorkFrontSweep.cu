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

__device__ int segmentedScan( int lane , int *values ) {
    if( lane >= 1  )
        values[threadIdx.x] += values[threadIdx.x - 1];
    if( lane >= 2 )
        values[threadIdx.x] += values[threadIdx.x - 2];
    if( lane >= 4 )
        values[threadIdx.x] += values[threadIdx.x - 4];
    if( lane >= 8  )
        values[threadIdx.x] += values[threadIdx.x - 8];
    if( lane >= 16 )
        values[threadIdx.x] += values[threadIdx.x - 16];
    return ( lane > 0 ) ? values[threadIdx.x - 1 ] : 0;
}

__global__ void filtering_stage_1(edge_list *edgeList, int length , int *x, int *updated_vertices, int *size) {

    int totalThreads = (blockDim.x*gridDim.x);
    int totalWarps = totalThreads / 32;
    int globalThreadId = threadIdx.x + (blockIdx.x * blockDim.x);
    int warpId = globalThreadId / 32;

    int load = length % totalWarps == 0 ? length/totalWarps : length/totalWarps + 1;
    int beginning = load * warpId;
    int end = (beginning+load) > length ? length : beginning + load;
    beginning  = beginning + lane_id();
    int mask = 0;
    for( int i = beginning ; i < end ; i += 32 ) {
        int source = edgeList[i].srcIndex;
        mask = __ballot(updated_vertices[source] == 1);
        __syncthreads();
        if( lane_id() == 0 ) {
            int count = __popc(mask);
            atomicAdd(&x[warpId],count);
            atomicAdd(size,count);
        }
        __syncthreads();
        mask = 0;
        __syncthreads();
    }
}

__global__ void filtering_stage_2(int *x) {
    int globalThreadId = threadIdx.x + (blockIdx.x * blockDim.x);
    int warpId = globalThreadId / 32;

    __syncthreads();
    int value = segmentedScan(lane_id(),x);
    if( lane_id() == 31 ) x[warpId] = x[globalThreadId];
    __syncthreads();

    if( warpId == 0 ) segmentedScan(lane_id(),x);
    __syncthreads();

    if(warpId > 0) value = x[warpId-1] + value;
    __syncthreads();

    x[globalThreadId] = value;
}

__global__ void filtering_stage_3(edge_list *edgeList ,int length, edge_list *toProcessEdgeList, int *y,  int *updated_vertices ) {

    int totalThreads = (blockDim.x*gridDim.x);
    int totalWarps = totalThreads / 32;
    int globalThreadId = threadIdx.x + (blockIdx.x * blockDim.x);
    int warpId = globalThreadId / 32;

    int load = length % totalWarps == 0 ? length/totalWarps : length/totalWarps + 1;
    int beginning = load * warpId;
    int end = (beginning+load) > length ? length : beginning + load;
    beginning = beginning + lane_id();

    int current_offset = y[warpId];
    int mask = 0;
    for( int i = beginning ; i < end ; i += 32 ) {
        int source = edgeList[i].srcIndex;
        mask = __ballot(updated_vertices[source] == 1);
        __syncthreads();
        int localId = __popc(mask <<(32 - lane_id()));
        if(updated_vertices[source] == 1) toProcessEdgeList[localId+current_offset] = edgeList[i];
        __syncthreads();
        current_offset += __popc(mask);
        __syncthreads();
        mask = 0;
        __syncthreads();
    }

}

__global__ void computation_outcore_noSharedMemory(edge_list *edgeList, int *size, int *distance_prev, int *distance_current, int *updated_vertices, int *anyChange){

    int totalThreads = (blockDim.x*gridDim.x);
    int totalWarps = totalThreads / 32;
    int globalThreadId = threadIdx.x + (blockIdx.x * blockDim.x);
    int warpId = globalThreadId / 32;

    int load = *size % totalWarps == 0 ? *size/totalWarps : *size/totalWarps + 1;
    int beginning = load * warpId;
    int end = (beginning+load) > *size ? *size : beginning + load;
    beginning  = beginning + lane_id();

    for( int i = beginning ; i < end ; i += 32 ) {
        int source = edgeList[i].srcIndex;
        int destination = edgeList[i].destIndex;
        int weight = edgeList[i].weight;
        if( distance_prev[source] + weight < distance_prev[destination]) {
            atomicMin(&distance_current[destination],distance_prev[source]+weight);
            updated_vertices[destination] = 1;
            *anyChange = 1;
        }
    }
}

__global__ void computation_incore_noSharedMemory(edge_list *edgeList, int *size , int *distance , int *updated_vertices, int *anyChange ) {

    int totalThreads = (blockDim.x*gridDim.x);
    int totalWarps = totalThreads / 32;
    int globalThreadId = threadIdx.x + (blockIdx.x * blockDim.x);
    int warpId = globalThreadId / 32;

    int load = *size % totalWarps == 0 ? *size/totalWarps : *size/totalWarps + 1;
    int beginning = load * warpId;
    int end = (beginning+load) > *size ? *size : beginning + load;
    beginning  = beginning + lane_id();

    for( int i = beginning ; i < end ; i += 32 ) {
        int source = edgeList[i].srcIndex;
        int destination = edgeList[i].destIndex;
        int weight = edgeList[i].weight;
        int temp_distance = distance[source] + weight;
        if( temp_distance < distance[destination]) {
            atomicMin(&distance[destination],distance[source]+weight);
            updated_vertices[destination] = 1;
            *anyChange = 1;
        }
    }
}

void print(int *x, int size) {
    printf("[ ");
    for(int i = 0 ; i < size; i++ ) {
        printf("%d, ",x[i]);
    }
    printf("]\n");
}

void neighborHandler(std::vector<initial_vertex> * parsedGraph, std::vector<edge_list> *edgeList, int blockSize, int blockNum, int edgeListOrder, int syncMethod, int smemMethod){
    setTime();

    if(syncMethod == 1) {
        setTime();

        uint vertices = parsedGraph->size();
        uint edges = edgeList->size();
        edge_list *temp_edgeList = (edge_list*)malloc(sizeof(edge_list)*edges);
        int *distance_current = (int*)malloc(sizeof(int)*vertices);
        int *distance_prev = (int*)malloc(sizeof(int)*vertices);
        int *anyChange = (int *)malloc(sizeof(int));

        edge_list *device_edgeList;
        edge_list *device_toProcessEdgeList;
        int *device_distance_prev;
        int *device_distance_current;
        int *device_anyChange;



        std::copy(edgeList->begin(),edgeList->end(),temp_edgeList);
        std::fill_n(distance_current,vertices,SSSP_INF);
        distance_current[0] = 0;
        std::fill_n(distance_prev,vertices,SSSP_INF);
        distance_prev[0] = 0;



        cudaMalloc((void **)&device_edgeList,sizeof(edge_list)*edges);
        cudaMalloc((void **)&device_toProcessEdgeList,sizeof(edge_list)*edges);
        cudaMalloc((void **)&device_distance_prev, sizeof(int)*vertices );
        cudaMalloc((void **)&device_distance_current, sizeof(int)*vertices);
        cudaMalloc((void **)&device_anyChange,sizeof(int));
        cudaMemcpy(device_distance_current,distance_current,sizeof(int)*vertices,cudaMemcpyHostToDevice);
        cudaMemcpy(device_distance_prev,distance_prev,sizeof(int)*vertices,cudaMemcpyHostToDevice);
        cudaMemcpy(device_edgeList,temp_edgeList,sizeof(edge_list)*edges,cudaMemcpyHostToDevice);

        int *updated_vertices = (int*)malloc(sizeof(int)*vertices);                                                // Represents the updated vertices in the previous iteration
        int *x = (int*)malloc(sizeof(int)*MAX_WARPS);                                                                 // Represents the number of to process edges in each warp

        std::fill_n(updated_vertices,vertices,0);
        updated_vertices[0] = 1;


        int *device_updated_vertices;
        int *device_x;
        int *device_size;

        cudaMalloc((void **)&device_updated_vertices,sizeof(int)*vertices);
        cudaMalloc((void **)&device_x, sizeof(int)*MAX_WARPS);
        cudaMalloc((void **)&device_size, sizeof(int));

        cudaMemcpy(device_updated_vertices,updated_vertices,sizeof(int)*vertices,cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        double totalFilteringTime = 0;
        double totalKernelTime = 0;
        for(uint i = 0 ; i < vertices ; i++) {
            cudaMemset(device_x,0,sizeof(int)*MAX_WARPS);
            cudaMemset(device_size,0,sizeof(int));

            double filtering_time_start = getTime();
            filtering_stage_1<<<blockNum,blockSize>>>(device_edgeList,edges,device_x,device_updated_vertices,device_size);
            int size;
            cudaMemcpy(&size,device_size,sizeof(int),cudaMemcpyDeviceToHost);
            cudaMemcpy(x,device_x,sizeof(int)*MAX_WARPS,cudaMemcpyDeviceToHost);
            filtering_stage_2<<<1,64>>>(device_x);
            cudaMemcpy(x,device_x,sizeof(int)*MAX_WARPS,cudaMemcpyDeviceToHost);
            filtering_stage_3<<<blockNum,blockSize>>>(device_edgeList,edges,device_toProcessEdgeList,device_x,device_updated_vertices);
            cudaDeviceSynchronize();
            double filtering_time_end = getTime();
            totalFilteringTime += (filtering_time_end - filtering_time_start);

            cudaMemset(device_updated_vertices,0,sizeof(int)*vertices);
            *anyChange = 0;
            cudaMemcpy(device_anyChange,anyChange,sizeof(int),cudaMemcpyHostToDevice);

            double kernel_time_start = getTime();
            computation_outcore_noSharedMemory<<<blockNum,blockSize>>>(device_toProcessEdgeList,device_size,device_distance_prev,device_distance_current,device_updated_vertices,device_anyChange);
            cudaDeviceSynchronize();
            double kernel_time_end = getTime();
            totalKernelTime += (kernel_time_end - kernel_time_start);


            cudaMemcpy(anyChange,device_anyChange,sizeof(int),cudaMemcpyDeviceToHost);
            if( *anyChange == 0 ) {
                printf("Outcore Iteration breaks at i : %d\n",i);
                break;
            }
            cudaMemcpy(device_distance_prev,device_distance_current,sizeof(int)*vertices,cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
        }
        cudaMemcpy(distance_current,device_distance_current,sizeof(int)*vertices,cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        for( int i = 0 ; i < vertices ; i++ ) {
            parsedGraph->at(i).vertexValue.distance = distance_current[i];
        }
        cudaFree(device_edgeList);
        cudaFree(device_distance_current);
        cudaFree(device_distance_prev);
        cudaFree(device_anyChange);
        cudaFree(device_toProcessEdgeList);
        std::cout << "Took Outcore Total Time :" << getTime() << "ms.\n";
        std::cout << "Took Outcore Filtering Time : " << totalFilteringTime << "ms.\n";
        std::cout << "Took Outcore Kernel Time : " << totalKernelTime << "ms.\n";
    }
    else if(syncMethod == 0) {

        setTime();
        /* Size of vertices and edges of the graph */
        uint vertices = parsedGraph->size();
        uint edges = edgeList->size();
        /* Edge List definition */
        edge_list *device_edgeList;
        edge_list *temp_edgeList;
        temp_edgeList = (edge_list*)malloc(sizeof(edge_list)*edges);
        /* Distance Arrays */
        int *distance = (int*)malloc(sizeof(int)*vertices);
        int *anyChange = (int *)malloc(sizeof(int));
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


        edge_list *device_toProcessEdgeList;
        cudaMalloc((void **)&device_toProcessEdgeList,sizeof(edge_list)*edges);

        int *updated_vertices = (int*)malloc(sizeof(int)*vertices);                                                // Represents the updated vertices in the previous iteration
        int *x = (int*)malloc(sizeof(int)*MAX_WARPS);                                                                 // Represents the number of to process edges in each warp

        std::fill_n(updated_vertices,vertices,0);
        updated_vertices[0] = 1;


        int *device_updated_vertices;
        int *device_x;
        int *device_size;

        cudaMalloc((void **)&device_updated_vertices,sizeof(int)*vertices);
        cudaMalloc((void **)&device_x, sizeof(int)*MAX_WARPS);
        cudaMalloc((void **)&device_size, sizeof(int));

        cudaMemcpy(device_updated_vertices,updated_vertices,sizeof(int)*vertices,cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        double totalFilteringTime = 0;
        double totalKernelTime = 0;
        for(uint i = 0 ; i < vertices ; i++) {
            cudaMemset(device_x,0,sizeof(int)*MAX_WARPS);
            cudaMemset(device_size,0,sizeof(int));

            double filtering_time_start = getTime();
            filtering_stage_1<<<blockNum,blockSize>>>(device_edgeList,edges,device_x,device_updated_vertices,device_size);
            int size;
            cudaMemcpy(&size,device_size,sizeof(int),cudaMemcpyDeviceToHost);
            filtering_stage_2<<<1,64>>>(device_x);
            cudaMemcpy(x,device_x,sizeof(int)*MAX_WARPS,cudaMemcpyDeviceToHost);
            filtering_stage_3<<<blockNum,blockSize>>>(device_edgeList,edges,device_toProcessEdgeList,device_x,device_updated_vertices);
            cudaDeviceSynchronize();
            double filtering_time_end = getTime();
            totalFilteringTime += (filtering_time_end - filtering_time_start);

            cudaMemset(device_updated_vertices,0,sizeof(int)*vertices);
            *anyChange = 0;
            cudaMemcpy(device_anyChange,anyChange,sizeof(int),cudaMemcpyHostToDevice);

            double kernel_time_start = getTime();
            computation_incore_noSharedMemory<<<blockNum,blockSize>>>(device_toProcessEdgeList,device_size,device_distance,device_updated_vertices,device_anyChange);
            cudaDeviceSynchronize();
            double kernel_time_end = getTime();
            totalKernelTime += (kernel_time_end - kernel_time_start);


            cudaMemcpy(anyChange,device_anyChange,sizeof(int),cudaMemcpyDeviceToHost);

            if( *anyChange == 0 ) {
                printf("Incore Iteration breaks at : %d.\n",i);
                break;
            }
            cudaDeviceSynchronize();
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
        cudaFree(device_toProcessEdgeList);
        std::cout << "Took Incore Total Time :" << getTime() << "ms.\n";
        std::cout << "Took Incore Filtering Time : " << totalFilteringTime << "ms.\n";
        std::cout << "Took Incore Kernel Time : " << totalKernelTime << "ms.\n";

    }
    else {
        std::cout << "Not A Correct input parameter Implementation.\n";
    }
    std::cout << "Took " << getTime() << "ms.\n";
}
