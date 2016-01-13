
import java.io.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

import mpi.MPI;


class Tuple3 implements Comparator<Tuple3>, Comparable<Tuple3>{
    private Integer key;
    private Double pagerank;
    Tuple3(){}
    Tuple3(Integer key, Double pagerank){
        this.key = key;
        this.pagerank = pagerank;
    }

    public Integer getTuple3Key(){
        return this.key;
    }
    public Double getTuple3PageRank(){
        return this.pagerank;
    }

    public int compareTo(Tuple3 t){
        return (this.key).compareTo(t.key);
    }

    public int compare(Tuple3 t1, Tuple3 t2){
        if(t1.getTuple3PageRank() > t2.getTuple3PageRank()) return -1;
        if(t1.getTuple3PageRank() < t2.getTuple3PageRank()) return 1;
        return 0;
    }

    public String toString(){
        return "[" + this.key + ": " + this.pagerank + "]";
    }
}


public class MPIPageRank {

    // adjacency matrix read from file
    private HashMap<Integer, List<Integer>> adjMatrix = new HashMap<Integer, List<Integer>>();
    // input file name
    private String inputFile = "";
    // output file name
    private String outputFile = "";
    // number of iterations
    private int iterations = 10;
    // damping factor
    private double df = 0.85;
    // number of URLs
    private int totalURLs = 0;
    // calculating rank values
    private HashMap<Integer, Double> rankValues = new HashMap<Integer, Double>();

    private Double danglingEffect;

    private Double delta = .001;
    /**
     * Parse the command line arguments and update the instance variables. Command line arguments are of the form
     * <input_file_name> <output_file_name> <num_iters> <damp_factor>
     *
     * @param args arguments
     */
    public void parseArgs(String[] args) {
        this.inputFile = args[3];
        this.outputFile = args[4];
        if (!args[5].isEmpty()) {
	    this.delta = Double.parseDouble(args[5]);
	    //this.iterations = Integer.parseInt(args[5]);
        }
        if (!args[6].isEmpty()) {
            this.df = Double.parseDouble(args[6]);
        }
    }

    /**
     * Read the input from the file and populate the adjacency matrix
     *
     * The input is of type
     *
          0
               1 2
                    2 1
                         3 0 1
                              4 1 3 5
                                   5 1 4
                                        6 1 4
                                             7 1 4
                                                  8 1 4
                                                       9 4
                                                            10 4
                                                            * The first value in each line is a URL. Each value af$
                                                            * For example the page represented by the 0 URL doesn'$
                                                            * represented by 1 refer the URL 2.
                                                            *
                                                            * @throws java.io.IOException if an error occurs
                                                            */
    public void loadInput() throws IOException {

        String line;
        BufferedReader reader = new BufferedReader(new FileReader(inputFile));
        while ((line = reader.readLine()) != null) {


            String[] fline = line.split(" ");

            List<Integer> fline2 = new ArrayList<>();

            for(int i = 0; i < fline.length; i++){
		//System.out.println(fline[i]);
                fline2.add(Integer.parseInt(fline[i]));
            }

            int firstElement = fline2.get(0);

            List<Integer> tempArray = fline2.subList(1, fline2.size());

            adjMatrix.put(firstElement,tempArray);
        }
        //System.out.println(adjMatrix);
        this.totalURLs = adjMatrix.keySet().size();
    }

    /**
     * Do fixed number of iterations and calculate the page rank values. You may keep the
     * intermediate page rank values in a hash table.
     * @param rank
     * @param localHash
     */

    public void calculatePageRank(int rank, HashMap<Integer, List<Integer>> localHash) {
        HashMap<Integer, Double> rankValuesIntermed = new HashMap<Integer, Double>();

        /*Initialize rankValueHashMap*/
        for(int x = 0; x < this.totalURLs; x++){
            rankValues.put(x, 1.0/this.totalURLs);
        }

        //for (Integer i=0 ; i<iterations ; i++){

        //you need to comment out these two lines to run interatively
        boolean placeBol = true;
        while(placeBol){
	    //
            danglingEffect = 0.0;
            //rankValuesIntermed.clear();
            /*Initialize intermedHashMap with (1-d)/N to begin partial sums*/
            for (Integer key : rankValues.keySet()){
                rankValuesIntermed.put(key, 0.0);
            }
            /*Loop through each key
	    **If key has no outbound links, add PR of dangling/size to dangleEffect to add to all after loop
            **If key has outbound links, lookup partial sum, add key's PR/#KeyOutbounds, update rankValues
            */
            for (Integer site : localHash.keySet()) {
                if (localHash.get(site).size() == 0) {
                    Double dangleAdd = df*(rankValues.get(site)/this.totalURLs);
                    danglingEffect += dangleAdd;
                } else {
                    for (Integer outbound : localHash.get(site)) {
                        Double newrank = rankValuesIntermed.get(outbound) + df*(rankValues.get(site)/(localHash.get(site).size()));
                        rankValuesIntermed.put(outbound, newrank);
                    }
                }
            }
            /*Add danglingEffect to all nodes.
	    **Lookup current partial sum, add dangling effect, update hash
            */

	    for (Integer key: rankValuesIntermed.keySet()){
                Double newrank = rankValuesIntermed.get(key) + danglingEffect;
                rankValuesIntermed.put(key, newrank);
            }

            double sendArray[] = new double[this.totalURLs];
            for(int y = 0; y < this.totalURLs; y++){
                if(rankValuesIntermed.containsKey(y)){
                    sendArray[y] = rankValuesIntermed.get(y);
                }else{
                    sendArray[y] += danglingEffect;
                }
            }

            MPI.COMM_WORLD.Allreduce(sendArray,0, sendArray, 0, sendArray.length, MPI.DOUBLE, MPI.SUM);

            for (Integer key : rankValues.keySet()){
                sendArray[key] += (1.0-df) / this.totalURLs;
            }

	    //need to comment out these lines if you want to run iteratively
	    placeBol = false;
            for(Integer key: rankValuesIntermed.keySet()){
                if (Math.abs(sendArray[key] - rankValues.get(key)) > this.delta){
                    placeBol = true;
                }

            }

            for(Integer key : rankValuesIntermed.keySet()){
                rankValues.put(key, sendArray[key]);
            }
        }

    }

    /**
     * Print the pagerank values. Before printing you should sort them according to decreasing order.
     * Print all the values to the output file. Print only the first 10 values to console.
     *
     * @throws IOException if an error occurs
     */
    public void printValues() throws IOException {

        List<Tuple3> list = new ArrayList<>();

        for(int i= 0; i < this.totalURLs; i++){
            list.add(new Tuple3(i, rankValues.get(i)));
        }

        Collections.sort(list, new Tuple3());

        double sumRV = 0.0;
        for(int i = 0; i < this.totalURLs; i++){
	    sumRV += rankValues.get(i);
        }
        System.out.println("Total of all pageranks: " + sumRV);

        System.out.println("These are the top 10 pageranks: ");

        for(int i = 0; i <11 ; i++){
            System.out.println(list.get(i).toString());
        }

        PrintWriter writer = new PrintWriter(outputFile, "UTF-8");
        writer.println("Sum of pageranks: " + sumRV);
        for(int i = 0; i < this.totalURLs; i++) {
            if(i==11){
                writer.println("----------------------------------------------------");
            }
	    writer.println("[key: pagerank] = " + list.get(i));

        }
        writer.close();

    }

    public static void main(String[] args) throws IOException {

        MPIPageRank mpiPR = new MPIPageRank();

        mpiPR.parseArgs(args);

        MPI.Init(args);

	int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();

        /**
           Initialize pageranks to be broadcasted
	**/

        int urlSize[] = new int[1];

        if (rank == 0) {
            mpiPR.loadInput();
            urlSize[0] = mpiPR.totalURLs;
        }


        MPI.COMM_WORLD.Bcast(urlSize, 0, 1, MPI.INT, 0);

        mpiPR.totalURLs = urlSize[0];

        HashMap<Integer, List<Integer>> localHash = new HashMap<Integer, List<Integer>>();

        /**
           Loop through all keys, send to proc in modulo to evenly distribute
	**/


        for(int i = 0; i < mpiPR.totalURLs; i++){

            if (rank == 0){
		int localKey[] = new int[1];
                int singleKey[] = new int[1];
                singleKey[0] = i;
                int localValues[] = new int[mpiPR.adjMatrix.get(i).size()];

                if(i%size == 0){
                    localKey = singleKey;
                    for(int i1 = 0; i1 < mpiPR.adjMatrix.get(localKey[0]).size(); i1++){
                        localValues[i1] = mpiPR.adjMatrix.get(localKey[0]).get(i1);
                    }

                    List<Integer> tempList = new ArrayList<Integer>();
                    for(Integer val: localValues){
                        tempList.add(val);
                    }

                    localHash.put(localKey[0], tempList);

                }else{

                    int[] valuesArray = new int [mpiPR.adjMatrix.get(singleKey[0]).size() + 1];

                    valuesArray[0] = singleKey[0];
                    for(int x = 1; x < mpiPR.adjMatrix.get(singleKey[0]).size() + 1; x++){
                        valuesArray[x] = mpiPR.adjMatrix.get(singleKey[0]).get(x-1);
                    }

                    int[] sizeOfVal = new int[1];

                    sizeOfVal[0] = mpiPR.adjMatrix.get(singleKey[0]).size() + 1;

                    MPI.COMM_WORLD.Send(sizeOfVal, 0, 1, MPI.INT, i%size, 2);

                    MPI.COMM_WORLD.Send(valuesArray, 0, valuesArray.length, MPI.INT, i%size, 3);

                }
            }
            else{
                if(i%size ==rank){
                    int localKey[] = new int[1];

                    int[] tempBuf = new int[1];
                    MPI.COMM_WORLD.Recv(tempBuf, 0, 1, MPI.INT, 0, 2);
                    int[] recValues = new int[tempBuf[0]];

                    MPI.COMM_WORLD.Recv(recValues, 0, recValues.length, MPI.INT, 0 , 3);
                    localKey[0] = recValues[0];
                    int[] localValues = Arrays.copyOfRange(recValues, 1, recValues.length);
                    //localValues = recValues;

                    List<Integer> tempList = new ArrayList<Integer>();

                    for(Integer val: localValues){
                        tempList.add(val);
                    }
                    localHash.put(localKey[0], tempList);
                }
            }
        }

        mpiPR.calculatePageRank(rank, localHash);
        if(rank == 0){
            mpiPR.printValues();
        }
        MPI.Finalize();
    }
}
