#include <cstring>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"



#include "NearFarPile.cu"
#include "Hybrid.cu"
#include "WorkFrontSweep.cu"
#include "ParallelBMF.cu"

#define SSSP_INF 1073741824

enum class ProcessingType {Push, Neighbor, Own, Hyb, Unknown};
enum SyncMode {InCore=0, OutOfCore=1};
enum SyncMode syncMethod;
enum SmemMode {UseSmem=0, UseNoSmem=1};
enum SmemMode smemMethod;
enum EdgeListMode {input=0,source=1,destination=2};
enum EdgeListMode edgeListOrder;

// Open files safely.
template <typename T_file>
void openFileToAccess( T_file& input_file, std::string file_name ) {
	input_file.open( file_name.c_str() );
	if( !input_file )
		throw std::runtime_error( "Failed to open specified file: " + file_name + "\n" );
}


// Execution entry point.
int main( int argc, char** argv )
{

	std::string usage =
		"\tRequired command line arguments:\n\
			Input file: E.g., --input in.txt\n\
                        Block size: E.g., --bsize 512\n\
                        Block count: E.g., --bcount 192\n\
                        Output path: E.g., --output output.txt\n\
			Processing method: E.g., --method bmf (bellman-ford), or wfs (WorkFrontSweep), or nfp (Near Far Pile ) or hyb (Hybrid)\n\
			Shared memory usage: E.g., --usesmem yes, or no \n\
			Sync method: E.g., --sync incore, or outcore\n\
			Edge List Order : E.g., --edgelist input,source,destination\n";

	try {

		std::ifstream inputFile;
		std::ofstream outputFile;
		int selectedDevice = 0;
		int bsize = 0, bcount = 0;
		long long arbparam = 0;
		bool nonDirectedGraph = false;		// By default, the graph is directed.
		ProcessingType processingMethod = ProcessingType::Unknown;
		syncMethod = OutOfCore;


		/********************************
		 * GETTING INPUT PARAMETERS.
		 ********************************/

		for( int iii = 1; iii < argc; ++iii )
			if ( !strcmp(argv[iii], "--method") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "bmf") )
				        processingMethod = ProcessingType::Push;
				else if ( !strcmp(argv[iii+1], "wfs") )
    				        processingMethod = ProcessingType::Neighbor;
				else if ( !strcmp(argv[iii+1], "nfp") )
				    processingMethod = ProcessingType::Own;
				else if ( !strcmp(argv[iii+1], "hyb") )
					processingMethod = ProcessingType::Hyb;
				else{
           std::cerr << "\n Un-recognized method parameter value \n\n";
           exit(1);
         }
			}
			else if ( !strcmp(argv[iii], "--sync") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "incore") )
				        syncMethod = InCore;
				else if ( !strcmp(argv[iii+1], "outcore") )
    				        syncMethod = OutOfCore;
				else{
           std::cerr << "\n Un-recognized sync parameter value \n\n";
           exit(1);
         }

			}
			else if ( !strcmp(argv[iii], "--usesmem") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "yes") )
				        smemMethod = UseSmem;
				else if ( !strcmp(argv[iii+1], "no") )
    				        smemMethod = UseNoSmem;
        else{
           std::cerr << "\n Un-recognized usesmem parameter value \n\n";
           exit(1);
         }
			}
			else if ( !strcmp(argv[iii], "--edgelist") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "input") )
				        edgeListOrder = input;
				else if ( !strcmp(argv[iii+1], "source") )
    				        edgeListOrder = source;
				else if ( !strcmp(argv[iii+1], "destination") )
							edgeListOrder = destination;
        else{
           std::cerr << "\n Un-recognized edgelist parameter value \n\n";
           exit(1);
         }
			}
			else if( !strcmp( argv[iii], "--input" ) && iii != argc-1 /*is not the last one*/)
				openFileToAccess< std::ifstream >( inputFile, std::string( argv[iii+1] ) );
			else if( !strcmp( argv[iii], "--output" ) && iii != argc-1 /*is not the last one*/)
				openFileToAccess< std::ofstream >( outputFile, std::string( argv[iii+1] ) );
			else if( !strcmp( argv[iii], "--bsize" ) && iii != argc-1 /*is not the last one*/)
				bsize = std::atoi( argv[iii+1] );
			else if( !strcmp( argv[iii], "--bcount" ) && iii != argc-1 /*is not the last one*/)
				bcount = std::atoi( argv[iii+1] );

		if(bsize <= 0 || bcount <= 0){
			std::cerr << "Usage: " << usage;
      exit(1);
			throw std::runtime_error("\nAn initialization error happened.\nExiting.");
		}
		if( !inputFile.is_open() || processingMethod == ProcessingType::Unknown ) {
			std::cerr << "Usage: " << usage;
			throw std::runtime_error( "\nAn initialization error happened.\nExiting." );
		}
		if( !outputFile.is_open() )
			openFileToAccess< std::ofstream >( outputFile, "out.txt" );
		CUDAErrorCheck( cudaSetDevice( selectedDevice ) );
		std::cout << "Device with ID " << selectedDevice << " is selected to process the graph.\n";


		/********************************
		 * Read the input graph file.
		 ********************************/

		std::cout << "Collecting the input graph ...\n";
		std::vector<initial_vertex> parsedGraph( 0 );
		std::vector<edge_list> edgeList( 0 );

		uint nEdges = parse_graph::parse(inputFile,parsedGraph,edgeList,arbparam,nonDirectedGraph,edgeListOrder );
		std::cout << "Input graph collected with " << parsedGraph.size() << " vertices and " << nEdges << " edges.\n";


		std::cout << "Sequential Bellman-Ford Implmentation Running ....\n";

		/* Sequential Implementation of Bellman-Ford Algorithm */
		uint vertices = parsedGraph.size();
		uint edges = nEdges;
		int distance[vertices];

		setTime();
		for( int i = 0 ; i < vertices ; i++ ) {
			distance[i] = SSSP_INF;
		}
		distance[0] = 0;

		for(uint i = 0 ; i < vertices ; i++ ) {
				bool change = false;
				for(uint j = 0 ; j < edges ; j++ ){
						int source = edgeList[j].srcIndex;
						int destination = edgeList[j].destIndex;
						int weight = edgeList[j].weight;
						if(distance[source] != SSSP_INF && distance[source] + weight < distance[destination]) {
								distance[destination] = distance[source] + weight;
								change = true;
						}
				}
				if( !change ) {
					printf("Sequential Bellman-Ford ends at iteration %d.\n",i);
					break;
				}
		}
		int max = 0;
		for( int i = 0 ; i < vertices ; i++ ) {
			if( distance[i] > max && distance[i] != SSSP_INF ) {
				max = distance[i];
			}
		}

		std::cout << "The maximum distance is the diameter of the graph : " << max << '\n';
		std::cout << "Sequential Bellman-Ford Took " << getTime() << "ms.\n";

		/********************************
		 * Process the graph.
		 ********************************/
		switch(processingMethod){
		case ProcessingType::Push:
		    puller(&parsedGraph, &edgeList, bsize, bcount,edgeListOrder,syncMethod,smemMethod);
		    break;
		case ProcessingType::Neighbor:
		    neighborHandler(&parsedGraph, &edgeList, bsize, bcount,edgeListOrder,syncMethod,smemMethod);
		    break;
		case ProcessingType::Hyb:
			hybrid(&parsedGraph, &edgeList, bsize, bcount,edgeListOrder,syncMethod,smemMethod);
			break;
		default:
		    own(&parsedGraph, &edgeList , bsize, bcount,edgeListOrder,syncMethod,smemMethod);
		}

		int differences = 0;
		for(int i = 0 ; i < vertices ; i++ ) {
			if( parsedGraph[i].vertexValue.distance == SSSP_INF )
				outputFile << i << ":\tInf\n";
			else
				outputFile << i << ":\t" << parsedGraph[i].vertexValue.distance << "\n";

			if( parsedGraph[i].vertexValue.distance != distance[i]) {
				//std::cout << "Sequential Distance for v " << i << " is : " << distance[i] << " and parallel gave : " << parsedGraph[i].vertexValue.distance << " \n.";
				differences++;
			}
		}

		if( differences == 0 ) {
			std::cout << "The serial and parallel versions of SSSP output matches.\n";
		}
		else {
			std::cout << "The serial and parallel versions of SSSP differs in " << differences << " distance values.\n";
		}


		/********************************
		 * It's done here.
		 ********************************/

		CUDAErrorCheck( cudaDeviceReset() );
		std::cout << "Done.\n";
		return( EXIT_SUCCESS );

	}
	catch( const std::exception& strException ) {
		std::cerr << strException.what() << "\n";
		return( EXIT_FAILURE );
	}
	catch(...) {
		std::cerr << "An exception has occurred." << std::endl;
		return( EXIT_FAILURE );
	}

}
