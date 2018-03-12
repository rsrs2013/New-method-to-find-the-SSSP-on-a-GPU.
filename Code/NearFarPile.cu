#include <vector>
#include <iostream>
#include <queue>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

#define WARP_SZ 32
#define MAX_WARPS 64
#define SSSP_INF 1073741824

__device__ inline int lane_id(void) { return threadIdx.x % WARP_SZ; }

__global__ void opt_incore_noSharedMemory(edge_list *edgeList, int size , int *distance , int *updated_vertices, int *anyChange ) {

    int totalThreads = (blockDim.x*gridDim.x);
    int totalWarps = totalThreads / 32;
    int globalThreadId = threadIdx.x + (blockIdx.x * blockDim.x);
    int warpId = globalThreadId / 32;

    int load = size % totalWarps == 0 ? size/totalWarps : size/totalWarps + 1;
    int beginning = load * warpId;
    int end = (beginning+load) > size ? size : beginning + load;
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

int divide_near_far_pile(int *distance , int *updated_vertices, std::vector<int>* nearPileVertices , std::vector<int>* farPileVertices , int vertices , int delta , int* increment ) {
    int nearPileElements = 0;
    for( int i = 0 ; i < vertices ; i++ ) {
        if( updated_vertices[i] == 1 ) {
            if( distance[i] < (*increment)*delta ) {
                nearPileVertices->push_back(i);
                nearPileElements++;
            }
            else if( distance[i] >= (*increment)*delta ) {
                farPileVertices->push_back(i);
            }
        }
    }
    return nearPileElements;
}


int divide_far_pile(int *distance , int *updated_vertices, std::vector<int>* nearPileVertices , std::vector<int>* farPileVertices , int vertices , int delta , int* increment ) {
        int nearPileElements = 0;

        int farPile[vertices];
        std::fill_n(farPile,vertices,0);
        for( int i = 0 ; i < farPileVertices->size() ; i++ ) {
            farPile[farPileVertices->at(i)] = 1;
        }
        int count = 0;

        while( nearPileElements == 0 ) {
            nearPileVertices->clear();
            farPileVertices->clear();
            *increment = *increment + 1;
            for( int i = 0 ; i < vertices ; i++ ) {
                if( farPile[i] == 1 ) {
                    if( distance[i] < (*increment)*delta ) {
                        nearPileVertices->push_back(i);
                        nearPileElements++;
                    }
                    else if( distance[i] >= (*increment)*delta ) {
                        farPileVertices->push_back(i);
                    }
                }
            }
            count++;
            if( count == SSSP_INF ) {
                printf("\nCant get the near pile vertices");
                break;
            }
        }
        return nearPileElements;

}

int collect_toProcessEdgeList(std::vector<int>* nearPileVertices, edge_list *edgeList, edge_list *toProcessEdgeList, uint vertices, uint edges ) {
    int readyEdges = 0;
    int nearPile[vertices];
    std::fill_n(nearPile,vertices,0);
    for( int i = 0 ; i < nearPileVertices->size() ; i++ ) {
        nearPile[nearPileVertices->at(i)] = 1;
    }

    for( int i = 0 ; i < edges ; i++ ) {
        int source = edgeList[i].srcIndex;
        if( nearPile[source] == 1 ) {
            toProcessEdgeList[readyEdges] = edgeList[i];
            readyEdges++;
        }
    }
    return readyEdges;

}


void own(std::vector<initial_vertex> * parsedGraph, std::vector<edge_list> *edgeList, int blockSize, int blockNum, int edgeListOrder, int syncMethod, int smemMethod){

    int delta;
    std::cout << "Input the Delta Parameter for The Near-Far Pile Implementation : ";
    std::cin >> delta;

    setTime();

    uint vertices = parsedGraph->size();
    uint edges = edgeList->size();
    edge_list *temp_edgeList = (edge_list*)malloc(sizeof(edge_list)*edges);
    std::copy(edgeList->begin(),edgeList->end(),temp_edgeList);
    edge_list *toProcessEdgeList = (edge_list*)malloc(sizeof(edge_list)*edges);;

    int *distance = (int*)malloc(sizeof(int)*vertices);
    std::fill_n(distance,vertices,SSSP_INF);
    distance[0] = 0;

    int *updated_vertices = (int*)malloc(sizeof(int)*vertices);                                                // Represents the updated vertices in the previous iteration
    std::fill_n(updated_vertices,vertices,0);
    updated_vertices[0] = 1;

    int *anyChange = (int *)malloc(sizeof(int));

    std::vector<int>* nearPileVertices = new std::vector<int>();
    std::vector<int>* farPileVertices = new std::vector<int>();
    int nearPileElements = 0;
    int farPileElements = 0;

    edge_list *device_toProcessEdgeList;
    int *device_distance;
    int *device_updated_vertices;
    int *device_anyChange;
    cudaMalloc((void **)&device_toProcessEdgeList,sizeof(edge_list)*edges);
    cudaMalloc((void **)&device_distance, sizeof(int)*vertices);
    cudaMalloc((void **)&device_updated_vertices,sizeof(int)*vertices);
    cudaMalloc((void **)&device_anyChange,sizeof(int));
    cudaMemcpy(device_distance,distance,sizeof(int)*vertices,cudaMemcpyHostToDevice);

    int* increment = new int[1];
    *increment = 0;

    printf("All Memory has been allocated");

    double totalKernelTime = 0;
    for( int i = 0 ; i < vertices*edges ; i++ ) {
        /* Get the near Pile Elements till since have updated vertices */
        nearPileElements = divide_near_far_pile(distance,updated_vertices,nearPileVertices,farPileVertices,vertices,delta,increment);
        if( nearPileElements == 0 ) {
            nearPileElements = divide_far_pile(distance,updated_vertices,nearPileVertices,farPileVertices,vertices,delta,increment);
        }

        int currentEdgeListSize = collect_toProcessEdgeList(nearPileVertices, temp_edgeList, toProcessEdgeList, vertices, edges);

        cudaMemset(device_updated_vertices,0,sizeof(int)*vertices);
        *anyChange = 0;
        cudaMemcpy(device_anyChange,anyChange,sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(device_toProcessEdgeList,toProcessEdgeList,sizeof(edge_list)*edges,cudaMemcpyHostToDevice);

        double kernel_time_start = getTime();
        opt_incore_noSharedMemory<<<blockNum,blockSize>>>(device_toProcessEdgeList,currentEdgeListSize,device_distance,device_updated_vertices,device_anyChange);
        cudaDeviceSynchronize();
        double kernel_time_end = getTime();
        totalKernelTime += (kernel_time_end - kernel_time_start);


        cudaMemcpy(distance,device_distance,sizeof(int)*vertices,cudaMemcpyDeviceToHost);
        cudaMemcpy(updated_vertices,device_updated_vertices,sizeof(int)*vertices,cudaMemcpyDeviceToHost);
        cudaMemcpy(anyChange,device_anyChange,sizeof(int),cudaMemcpyDeviceToHost);

        if( *anyChange == 0 && farPileVertices->size() == 0 ) {
            printf("Incore Iteration breaks at : %d.\n",i);
            break;
        }

        nearPileVertices->clear();
        nearPileElements = 0;
        cudaDeviceSynchronize();
    }

    cudaMemcpy(distance,device_distance,sizeof(int)*vertices,cudaMemcpyDeviceToHost);
    /* Copy the distance into the parsed graph structure */
    for( int i = 0 ; i < vertices ; i++ ) {
        parsedGraph->at(i).vertexValue.distance = distance[i];
    }
    /* Free the memory allocated */
    cudaFree(device_updated_vertices);
    cudaFree(device_distance);
    cudaFree(device_anyChange);
    cudaFree(device_toProcessEdgeList);

    std::cout << "Took Outcore Kernel Time : " << totalKernelTime << "ms.\n";
    std::cout << "Took " << getTime() << "ms.\n";
}
