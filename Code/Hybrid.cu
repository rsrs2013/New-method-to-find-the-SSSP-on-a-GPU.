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


typedef struct up {
    int vertex;
    int distance;
    int edges;
} upver;

int decider(int *updated_vertices, edge_list *edgeList , int *edgeCount, uint vertices, uint edges, int parameter ) {
        int number = 0;
        for( int i = 0 ; i < vertices ; i++ ) {
            for( int j = 0 ; j < edges ; j++ ) {
                int source = edgeList[j].srcIndex;
                if( updated_vertices[i] == 1 ) {
                    if( i == source ) {
                        edgeCount[i]++;
                        number++;
                    }
                }
            }
        }
        if( number > parameter ) {
            return 1;
        }
        return 0;
}

int compare_vertex(upver a, upver b) { 
    return a.distance < b.distance;
}

void hybrid(std::vector<initial_vertex> * parsedGraph, std::vector<edge_list> *edgeList, int blockSize, int blockNum, int edgeListOrder, int syncMethod, int smemMethod){

    int delta;
    std::cout << "Input the Parameter for The Hybrid Implementation : ";
    std::cin >> delta;

    setTime();

    /*
     *  Initial Graph Declarations
     */
    uint vertices = parsedGraph->size();
    uint edges = edgeList->size();
    edge_list *temp_edgeList = (edge_list*)malloc(sizeof(edge_list)*edges);
    std::copy(edgeList->begin(),edgeList->end(),temp_edgeList);
    edge_list *toProcessEdgeList = (edge_list*)malloc(sizeof(edge_list)*edges);;


    /*
     * Distance of Every Vertices
    */
    int *distance = (int*)malloc(sizeof(int)*vertices);
    std::fill_n(distance,vertices,SSSP_INF);
    distance[0] = 0;

    /*
     * Updated Vertices
    */
    int *updated_vertices = (int*)malloc(sizeof(int)*vertices);                                                // Represents the updated vertices in the previous iteration
    std::fill_n(updated_vertices,vertices,0);
    updated_vertices[0] = 1;

    /* Priority Queue */
    std::vector<upver>* work_queue = new std::vector<upver>();
    int *edgeCount = (int*)malloc(sizeof(int)*vertices);
    std::fill_n(edgeCount,vertices,0);


    int *anyChange = (int *)malloc(sizeof(int));

    /*
     * For Near Far Pile Implementation we require to show both near pile and the far pile
    */
    std::vector<int>* nearPileVertices = new std::vector<int>();
    std::vector<int>* farPileVertices = new std::vector<int>();
    int nearPileElements = 0;
    int farPileElements = 0;

    int* increment = new int[1];
    *increment = 0;


    /* To process Edge List */
    edge_list *device_toProcessEdgeList;

    /* Device Implementation */
    int *device_distance;
    int *device_updated_vertices;
    int *device_anyChange;
    cudaMalloc((void **)&device_toProcessEdgeList,sizeof(edge_list)*edges);
    cudaMalloc((void **)&device_distance, sizeof(int)*vertices);
    cudaMalloc((void **)&device_updated_vertices,sizeof(int)*vertices);
    cudaMalloc((void **)&device_anyChange,sizeof(int));
    cudaMemcpy(device_distance,distance,sizeof(int)*vertices,cudaMemcpyHostToDevice);



    printf("All Memory has been allocated");
    double totalKernelTime = 0;
    for( int i = 0 ; i < vertices; i++ ) {

        /* Parameter Definition  */
        int parameter = delta*blockNum*blockSize;
        int decision = decider(updated_vertices,temp_edgeList,edgeCount,vertices,edges,parameter);

        if( decision ) {

            for( int j = 0 ; j < vertices ; j++ ) {
                if( updated_vertices[j] == 1 ) {
                    upver w;
                    w.vertex = j;
                    w.distance = distance[j];
                    w.edges = edgeCount[j];
                    work_queue->push_back(w);
                }
            }

            std::sort(work_queue->begin(),work_queue->end(),compare_vertex);


            int breaking_point = 0;
            bool first = false;
            int count = 0;
            for( int j = 0 ; j < work_queue->size() ; j++ ) {
                work_queue->at(j).edges = count + work_queue->at(j).edges;
                count =  count + work_queue->at(j).edges;
                if( count > parameter && !first ) {
                    breaking_point = j;
                    first = true;
                }
            }

            for( int j = 0 ; j <= breaking_point ; j++ ) {
                nearPileVertices->push_back(work_queue->at(j).vertex);
            }

            for( int j = breaking_point+1 ; j < work_queue->size() ; j++ ) {
                farPileVertices->push_back(work_queue->at(j).vertex);
            }

            count = 0;
            for( int j = 0 ; j < edges ; j++ ) {
                for( int k = 0 ; k < nearPileVertices->size() ; k++ ) {
                    int source = temp_edgeList[j].srcIndex;
                    if( source == nearPileVertices->at(k) ) {
                        toProcessEdgeList[count] = temp_edgeList[j];
                        count++;
                    }
                }
            }
            int currentEdgeListSize = count;
            cudaMemset(device_updated_vertices,0,sizeof(int)*vertices);
            *anyChange = 0;
            cudaMemcpy(device_anyChange,anyChange,sizeof(int),cudaMemcpyHostToDevice);
            cudaMemcpy(device_toProcessEdgeList,toProcessEdgeList,sizeof(edge_list)*edges,cudaMemcpyHostToDevice);

            double kernel_time_start = getTime();
            opt_incore_noSharedMemory<<<blockNum,blockSize>>>(device_toProcessEdgeList,currentEdgeListSize,device_distance,device_updated_vertices,device_anyChange);
            cudaDeviceSynchronize();
            double kernel_time_end = getTime();
            totalKernelTime += (kernel_time_end - kernel_time_start);

        }
        else {

            *anyChange = 0;
            cudaMemcpy(device_anyChange,anyChange,sizeof(int),cudaMemcpyHostToDevice);

            int count = 0;
            for( int j = 0 ; j < edges ; j++ ) {
                for(int k = 0 ; k < vertices ; k++ ) {
                    int source = temp_edgeList[j].srcIndex;
                    if( updated_vertices[k] == 1 )
                    {
                        if( source == k ) {
                            toProcessEdgeList[count] = temp_edgeList[j];
                            count++;
                        }
                    }
                }
            }
            int currentEdgeListSize = count;

            cudaMemset(device_updated_vertices,0,sizeof(int)*vertices);
            *anyChange = 0;
            cudaMemcpy(device_anyChange,anyChange,sizeof(int),cudaMemcpyHostToDevice);
            cudaMemcpy(device_toProcessEdgeList,toProcessEdgeList,sizeof(edge_list)*edges,cudaMemcpyHostToDevice);

            double kernel_time_start = getTime();
            opt_incore_noSharedMemory<<<blockNum,blockSize>>>(device_toProcessEdgeList,currentEdgeListSize,device_distance,device_updated_vertices,device_anyChange);
            cudaDeviceSynchronize();
            double kernel_time_end = getTime();
            totalKernelTime += (kernel_time_end - kernel_time_start);

        }

        cudaMemcpy(distance,device_distance,sizeof(int)*vertices,cudaMemcpyDeviceToHost);
        cudaMemcpy(updated_vertices,device_updated_vertices,sizeof(int)*vertices,cudaMemcpyDeviceToHost);
        cudaMemcpy(anyChange,device_anyChange,sizeof(int),cudaMemcpyDeviceToHost);

        // Appending Far Pile to the updated Vertices
        for( int j = 0 ; j < farPileVertices->size() ; j++ ) {
            updated_vertices[farPileVertices->at(j)] = 1;
        }

        if( *anyChange == 0 && farPileVertices->size() == 0 ) {
            printf("No updated Vertices Left and iteration breaks at %d",i);
        }

        nearPileVertices->clear();
        farPileVertices->clear();

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
