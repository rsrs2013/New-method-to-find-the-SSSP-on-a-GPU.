Files :

1. ParallelBMF.cu : Implements parallel Bellman-Ford
2. WorkFrontSweep.cu : Implements Work Front Sweep Method
3. NearFarPile.cu : Implements Near Far Pile Method
4. Hybrid.cu : Implements Hybrid Method.

To Run the Code :

./sssp --input <input file>  --bsize 512 --bcount 4 --output <output file> --method <method name> --usesmem no --sync incore --edgelist source

<method name> :
1. bmf (Parallel Bellman-Ford)
2. wfs (WorkFrontSweep)
3. nfp (NearFarPile)
4. hyb (Hybrid)
