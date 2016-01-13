
import java.io.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;
import java.util.ArrayList;
import java.util.List;



class Tuple implements Comparator<Tuple>, Comparable<Tuple>{
    private Integer key;
    private Double pagerank;
    Tuple(){}
    Tuple(Integer key, Double pagerank){
        this.key = key;
        this.pagerank = pagerank;
    }

    public Integer getTupleKey(){
        return this.key;

    }
    public Double getTuplePageRank(){
        return this.pagerank;
    }

    public int compareTo(Tuple t){
        return (this.key).compareTo(t.key);
    }

    public int compare(Tuple t1, Tuple t2){
        if(t1.getTuplePageRank() > t2.getTuplePageRank()) return -1;
        if(t1.getTuplePageRank() < t2.getTuplePageRank()) return 1;
        return 0;
    }

    public String toString(){
        return "[" + this.key + ": " + this.pagerank + "]";
    }
}


public class SequentialPageRank {
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
    private int size = 0;
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
        this.inputFile = args[0];
        this.outputFile = args[1];
        this.delta = Double.parseDouble(args[2]);
        this.df = Double.parseDouble(args[3]);
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
     * The first value in each line is a URL. Each value after the first value is the URLs referred by the first U
     * For example the page represented by the 0 URL doesn't refer any other URL. Page
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
	// System.out.println(adjMatrix);
    }

    /**
     * Do fixed number of iterations and calculate the page rank values. You may keep the
     * intermediate page rank values in a hash table.
     */

    public void calculatePageRank() {
	HashMap<Integer, Double> rankValuesIntermed = new HashMap<Integer, Double>();

	this.size = adjMatrix.keySet().size();

        /*Initialize rankValueHashMap*/
	for (Integer key : adjMatrix.keySet()){
	    rankValues.put(key, 1.0/this.size);
	}

        //for (Integer i=0 ; i<iterations ; i++){
        boolean placeBol = true;
        while(placeBol){
	    
            danglingEffect = 0.0;

            rankValuesIntermed.clear();

	    /*Initialize intermedHashMap with (1-d)/N to begin partial sums*/
	    for (Integer key : adjMatrix.keySet()){
		rankValuesIntermed.put(key, (1.0-df) / this.size);
	    }
	    /*Loop through each key
	    **If key has no outbound links, add PR of dangling/size to dangleEffect to add to all after loop
	    **If key has outbound links, lookup partial sum, add key's PR/#KeyOutbounds, update rankValues
	    */
	    for (Integer site : adjMatrix.keySet()) {
		if (adjMatrix.get(site).size() == 0) {
		    Double dangleAdd = df*(rankValues.get(site)/this.size);
		    danglingEffect += dangleAdd;
		} else {
		    for (Integer outbound : adjMatrix.get(site)) {
                        //System.out.println(outbound);
			Double newrank = rankValuesIntermed.get(outbound) + df*(rankValues.get(site)/(adjMatrix.get(site).size()));
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


	    placeBol = false;
            for(Integer key: rankValuesIntermed.keySet()){
                if (Math.abs(rankValuesIntermed.get(key) - rankValues.get(key)) > this.delta){
                    placeBol = true;
                }

            }
            // cloning humans.
            for(Integer key : rankValuesIntermed.keySet()){
                rankValues.put(key, rankValuesIntermed.get(key));
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

        List<Tuple> list = new ArrayList<>();

        for(int i= 0; i < this.size; i++){
	    list.add(new Tuple(i, rankValues.get(i)));
	}

        Collections.sort(list, new Tuple());

        System.out.println("These are the top 10 pageranks: ");

        for(int i = 0; i <11 ; i++){
            System.out.println(list.get(i).toString());
        }


	PrintWriter writer = new PrintWriter(outputFile, "UTF-8");
	for(int i = 0; i < this.size; i++) {
	    writer.println("[key: pagerank] = " + list.get(i));
	}
	writer.close();



    }

    public static void main(String[] args) throws IOException {
        SequentialPageRank sequentialPR = new SequentialPageRank();

        sequentialPR.parseArgs(args);
        sequentialPR.loadInput();
        sequentialPR.calculatePageRank();
        sequentialPR.printValues();

    }
}

