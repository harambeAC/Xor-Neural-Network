import java.util.Arrays;

public class Network {
    Layer input_layer;
    Matrix hidden_layer_weights; // these are the weights going into the hidden layer. Will be a 2X5 matrix
    Matrix hidden_layer_bias; // this is the bias weights going into the hidden layer. Will be a 1X5 matrix
    Layer hidden_layer;
    Matrix hidden_layer_output;
    Matrix output_layer_weights; // these are the weights going into the output layer. Will be a 5X2 matrix
    Matrix output_layer_bias; // this is the bias weights going into the output layer. Will be a 1X2 matrix
    Layer output_layer;

    Matrix hidden_layer_weights_derivatives;
    Matrix hidden_layer_bias_derivatives;
    Matrix output_layer_weights_derivatives;
    Matrix output_layer_bias_derivatives;
    Matrix network_input;

    //my hyperparameters
    final int inputLayerNodes = 2;
    final int hiddenLayerNodes = 5;
    final int outputLayerNodes = 2;
    double learning_rate = 1;


    Network(Matrix network_input, Matrix target_output){ // trains network
    		this.network_input = network_input;
        double[][] hidden_layer_weights = new double[inputLayerNodes][hiddenLayerNodes]; //2*2
        double[][] hidden_layer_bias = new double[1][hiddenLayerNodes];
        double[][] output_layer_weights = new double[hiddenLayerNodes][outputLayerNodes]; //2*2
        double[][] output_layer_bias = new double[1][outputLayerNodes];

        initialize(hidden_layer_weights);
        initialize(hidden_layer_bias);
        initialize(output_layer_weights);
        initialize(output_layer_bias);

        this.hidden_layer_weights = new Matrix(hidden_layer_weights);
        this.hidden_layer_weights_derivatives = new Matrix(hidden_layer_weights);

        this.hidden_layer_bias = new Matrix(hidden_layer_bias);
        this.hidden_layer_bias_derivatives = new Matrix(hidden_layer_bias);

        this.output_layer_weights = new Matrix(output_layer_weights);
        this.output_layer_weights_derivatives = new Matrix(output_layer_weights);

        this.output_layer_bias = new Matrix(output_layer_bias);
        this.output_layer_bias_derivatives = new Matrix(output_layer_bias);

        //train
        backpropagation(target_output);

    }
    
    int test(Matrix input) {
    		Matrix output = feedforward(input);
    		System.out.println(Arrays.deepToString(output.matrix));
		if(output.matrix[0][0] > output.matrix[0][1]) {
			return 0;
		}
		return 1;
    }
    
    void printInfo(Matrix network_output, double[] target_output, double[] network_input){
    		System.out.println("------------------------------------------------------");
        System.out.println("network input: "+ Arrays.toString(network_input));
        System.out.println("network output: "+ Arrays.deepToString(network_output.matrix));
        System.out.println("target output: "+Arrays.toString(target_output));
        System.out.println("hidden layer weights"+Arrays.deepToString(this.hidden_layer_weights.matrix));
        System.out.println("hidden layer bias" + Arrays.deepToString(this.hidden_layer_bias.matrix));
        System.out.println("hidden layer output"+Arrays.deepToString(this.hidden_layer_output.matrix));
        System.out.println("output layer weights"+Arrays.deepToString(this.output_layer_weights.matrix));
        System.out.println("output layer bias"+Arrays.deepToString(this.output_layer_bias.matrix));
        System.out.println("hidden layer weights derivative"+Arrays.deepToString(this.hidden_layer_weights_derivatives.matrix));
        System.out.println("hidden layer bias derivative " + Arrays.deepToString(this.hidden_layer_bias_derivatives.matrix));
        System.out.println("output layer weights derivative"+Arrays.deepToString(this.output_layer_weights_derivatives.matrix));
        System.out.println("output layer bias derivative"+Arrays.deepToString(this.output_layer_bias_derivatives.matrix));
    }

    void initialize(double[][] arr){
        for(int i = 0; i<arr.length; i++){
            for(int j = 0; j<arr[i].length; j++){
                arr[i][j] = Math.random();
            }
        }
    }

    Matrix feedforward(Matrix network_input){
        this.input_layer = new Layer(inputLayerNodes); // output is already determined
        this.hidden_layer = new Layer(hiddenLayerNodes);

        Matrix sum = hidden_layer.summation(hidden_layer_weights, network_input, hidden_layer_bias);
        hidden_layer_output = Layer.sigmoid(sum);

        this.output_layer = new Layer(outputLayerNodes);
        sum = hidden_layer.summation(output_layer_weights, hidden_layer_output, output_layer_bias);

        return Layer.sigmoid(sum);
    }

    void backpropagation(Matrix target_output){
        for(int i = 0; i<network_input.matrix.length; i++) {
	        Matrix network_output = feedforward(new Matrix(network_input.matrix[i]));
	        double cost = mean_square_error(network_output, new Matrix(target_output.matrix[i]));

	        printInfo(network_output, target_output.matrix[i], network_input.matrix[i]);
	        calculate_gradient(new Matrix(network_input.matrix[i]), network_output,new Matrix(target_output.matrix[i]));
	        //check_gradients(cost, new Matrix(network_input.matrix[i]), new Matrix(target_output.matrix[i]));
	        printInfo(network_output, target_output.matrix[i], network_input.matrix[i]);

	        
	        update_weights();
	        //System.out.println(Arrays.deepToString(network_input.matrix));

	        System.out.println(cost);
        }
    }
    
    void check_gradients(double cost, Matrix network_input,Matrix target_output) { // prints relative Error
        double epsilon = 0.0001;
		// check for hidden layer weights
        for(int i = 0; i<hidden_layer_weights.matrix.length; i++){
            for(int j = 0; j<hidden_layer_weights.matrix[i].length; j++){                        	
            		hidden_layer_weights.matrix[i][j] += epsilon;         		
            		double error2 = mean_square_error(feedforward(network_input),target_output);
            		
            		hidden_layer_weights.matrix[i][j] -= 2*epsilon;
            		double error1 = mean_square_error(feedforward(network_input),target_output);
            		
            		double numerical_gradient = (error1 - error2)/(2*epsilon);
            		
            		hidden_layer_weights.matrix[i][j] += epsilon;
            }
        }
        
		// check for hidden layer bias
        for(int i = 0; i<hidden_layer_bias.matrix.length; i++){
            for(int j = 0; j<hidden_layer_bias.matrix[i].length; j++){                        	
            		hidden_layer_bias.matrix[i][j] += epsilon;         		
            		double error2 = mean_square_error(feedforward(network_input),target_output);
            		
            		hidden_layer_bias.matrix[i][j] -= 2*epsilon;
            		double error1 = mean_square_error(feedforward(network_input),target_output);
            		
            		double numerical_gradient = (error1 - error2)/(2*epsilon);
            		//System.out.println(cost - numerical_gradient);
            		
            		hidden_layer_bias.matrix[i][j] += epsilon;
            }
        }
        
        // check for output layer weights
        for(int i = 0; i<output_layer_weights.matrix.length; i++){
            for(int j = 0; j<output_layer_weights.matrix[i].length; j++){                        	
            		output_layer_weights.matrix[i][j] += epsilon;         		
            		double error2 = mean_square_error(feedforward(network_input),target_output);
            		
            		output_layer_weights.matrix[i][j] -= 2*epsilon;
            		double error1 = mean_square_error(feedforward(network_input),target_output);
            		
            		double numerical_gradient = (error1 - error2)/(2*epsilon);
            		//System.out.println(cost - numerical_gradient);
            		
            		output_layer_weights.matrix[i][j] += epsilon;
            }
        }
        
        // check for output layer bias
        for(int i = 0; i<output_layer_bias.matrix.length; i++){
            for(int j = 0; j<output_layer_bias.matrix[i].length; j++){                        	
            		output_layer_bias.matrix[i][j] += epsilon;         		
            		double error2 = mean_square_error(feedforward(network_input),target_output);
            		
            		output_layer_bias.matrix[i][j] -= 2*epsilon;
            		double error1 = mean_square_error(feedforward(network_input),target_output);
            		
            		double numerical_gradient = (error1 - error2)/(2*epsilon);
            		//System.out.println(cost - numerical_gradient);
            		
            		output_layer_bias.matrix[i][j] += epsilon;
            }
        }
    }
    
    void update_weights() {  	
    		//update hidden layer weights
        for(int i = 0; i<hidden_layer_weights.matrix.length; i++){
            for(int j = 0; j<hidden_layer_weights.matrix[i].length; j++){
            		hidden_layer_weights.matrix[i][j] -= learning_rate * hidden_layer_weights_derivatives.matrix[i][j];
            }
        }
        
		//update hidden layer bias weights
        for(int i = 0; i<hidden_layer_bias.matrix.length; i++){
            for(int j = 0; j<hidden_layer_bias.matrix[i].length; j++){
            		hidden_layer_bias.matrix[i][j] -= learning_rate * hidden_layer_bias_derivatives.matrix[i][j];
            }
        }
        
		//update output layer  weights
        for(int i = 0; i<output_layer_weights.matrix.length; i++){
            for(int j = 0; j<output_layer_weights.matrix[i].length; j++){
            		output_layer_weights.matrix[i][j] -= learning_rate * output_layer_weights_derivatives.matrix[i][j];
            }
        }
        
		//update output layer bias weights
        for(int i = 0; i<output_layer_bias.matrix.length; i++){
            for(int j = 0; j<output_layer_bias.matrix[i].length; j++){
            		output_layer_bias.matrix[i][j] -= learning_rate * output_layer_bias_derivatives.matrix[i][j];
            }
        }
    }

    void calculate_gradient(Matrix network_input, Matrix network_output, Matrix target_output){
        //calculate gradients of output layer weights
        for(int j = 0; j<output_layer_weights.matrix.length;j++){
            for(int i = 0; i<output_layer_weights.matrix[i].length;i++){ // since its flip flopped, a 5X2 matrix. first index is which weight,second is which node
                double node_output = network_output.matrix[0][i];
                double target = target_output.matrix[0][i];
                double previous_node_output = hidden_layer_output.matrix[0][j];
                output_layer_weights_derivatives.matrix[j][i] = (node_output-target)*node_output*(1-node_output)*previous_node_output;
            }

        }
        
        //calculate gradients of output layer bias weights
       for(int i = 0; i<output_layer_bias.matrix.length;i++){
            for(int j = 0; j<output_layer_bias.matrix[i].length;j++){
                double node_output = network_output.matrix[0][j];
                double target = target_output.matrix[0][j];
                double deltaZ =(node_output-target)*node_output*(1-node_output);
                output_layer_bias_derivatives.matrix[i][j] = deltaZ;
            }
        }

        //calculate gradients of hidden layer weights
        for(int i = 0; i<hidden_layer_weights.matrix.length;i++){ // 2 options
            for(int j = 0; j<hidden_layer_weights.matrix[i].length;j++){ // 5 options
                double node_output = network_output.matrix[0][i];
                double target = target_output.matrix[0][i];
                double previous_node_output = network_input.matrix[0][i];
                double next_node_output = hidden_layer_output.matrix[0][j];
                
                double deltaZ =(node_output-target)*node_output*(1-node_output);
                
                double sum = 0;
                for(int k = 0; k<output_layer_weights.matrix.length; k++) {
                		sum += deltaZ * output_layer_weights.matrix[k][i];
                }
                
                hidden_layer_weights_derivatives.matrix[i][j] = (sum)*previous_node_output*(1-next_node_output)*next_node_output;
            }
        }

        //calculate gradients of hidden layer bias weights
        for(int i = 0; i<inputLayerNodes;i++){ // 2
            for(int j = 0; j<hidden_layer_bias.matrix[0].length;j++){ // 5
                double node_output = network_output.matrix[0][i];
                double target = target_output.matrix[0][i];
                double next_node_output = hidden_layer_output.matrix[0][j];
                
                double deltaZ =(node_output-target)*node_output*(1-node_output);
                
                double sum = 0;
                for(int k = 0; k<output_layer_weights.matrix.length; k++) {
	            		sum += deltaZ * output_layer_weights.matrix[k][i];
                }

                hidden_layer_bias_derivatives.matrix[0][j] = (sum)*(1-next_node_output)*next_node_output;
            }
        }
    }

    double mean_square_error(Matrix network_output, Matrix target_output){
        //System.out.println("network output: "+ Arrays.deepToString(network_output.matrix));
        //System.out.println("target output: "+ Arrays.deepToString(target_output.matrix));
    	
        double cost = 0;
        for(int i = 0; i<target_output.matrix[0].length; i++){
            cost += Math.pow(target_output.matrix[0][i]-network_output.matrix[0][i], 2);
        }
        return (cost)/target_output.matrix[0].length;
    }
    double squared_error(Matrix network_output, Matrix target_output){
        double cost = 0;
        for(int i = 0; i<target_output.matrix[0].length; i++){
            cost += Math.pow(target_output.matrix[0][i]-network_output.matrix[0][i], 2);
        }
        return 0.5*(cost);
    }
    double root_mean_square_error(Matrix network_output, Matrix target_output){
        double cost = 0;
        for(int i = 0; i<target_output.matrix[0].length; i++){
            cost += Math.pow(target_output.matrix[0][i]-network_output.matrix[0][i], 2);
        }
        return Math.sqrt(cost/target_output.matrix[0].length);
    }
    double sum_square_error(Matrix network_output, Matrix target_output) {
        double cost = 0;
        for (int i = 0; i < target_output.matrix[0].length; i++) {
            cost += Math.pow(target_output.matrix[0][i] - network_output.matrix[0][i], 2);
        }
        return cost;
    }
    double cross_entropy(Matrix network_output, Matrix target_output) {
        double cost = 0;
        for (int i = 0; i < target_output.matrix[0].length; i++) {
            cost += -1*target_output.matrix[0][i]* Math.log(network_output.matrix[0][i]);
        }
        return cost;
    }
}
