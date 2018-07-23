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

    final int inputLayerNodes = 2;
    final int hiddenLayerNodes = 2;
    final int outputLayerNodes = 2;


    Network(Matrix network_input, Matrix target_output){
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

        Matrix network_output = feedforward(network_input); //index 0 = 0, index 1 = 1
        System.out.println("network output: "+ Arrays.deepToString(network_output.matrix));
        System.out.println("target output: "+Arrays.deepToString(target_output.matrix));
        System.out.println("hidden layer weights"+Arrays.deepToString(this.hidden_layer_weights.matrix));
        System.out.println("hidden layer bias" + Arrays.deepToString(this.hidden_layer_bias.matrix));
        System.out.println("hidden layer output"+Arrays.deepToString(this.hidden_layer_output.matrix));
        System.out.println("output layer weights"+Arrays.deepToString(this.output_layer_weights.matrix));
        System.out.println("output layer bias"+Arrays.deepToString(this.output_layer_bias.matrix));

        //backpropagation(network_input,network_output, target_output);

        System.out.println("hidden layer weights derivative"+Arrays.deepToString(this.hidden_layer_weights_derivatives.matrix));
        System.out.println("hidden layer bias derivative" + Arrays.deepToString(this.hidden_layer_bias_derivatives.matrix));
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

        Matrix sum = hidden_layer.summation(network_input, hidden_layer_weights, hidden_layer_bias);
        hidden_layer_output = Layer.sigmoid(sum);

        this.output_layer = new Layer(outputLayerNodes);
        sum = hidden_layer.summation(hidden_layer_output, output_layer_weights, output_layer_bias);
        Matrix network_output = Layer.sigmoid(sum);

        return network_output;
    }

    void backpropagation(Matrix network_input, Matrix network_output, Matrix target_output){
        double cost = mean_square_error(network_output, target_output);
        //System.out.println(cost);

        //calculate_gradient(network_input, network_output,target_output);
        double epsilon = 0.0001;
        /*for(int i = 0; i<hidden_layer_weights_derivatives; i++){
            double numerical_gradient = (mean_square_error(+epsilon) - mean_square_error(-epsilon))/(2*epsilon);
        }*/
    }

    void calculate_gradient(Matrix network_input, Matrix network_output, Matrix target_output){
        //calculate gradients of output layer weights
        for(int i = 0; i<output_layer_weights.matrix.length;i++){
            for(int j = 0; j<output_layer_weights.matrix[i].length;j++){
                double node_output = network_output.matrix[i][j];
                double target = target_output.matrix[i][j];
                double previous_node_output = hidden_layer_output.matrix[i][j];
                output_layer_weights_derivatives.matrix[i][j] = (node_output-target)*node_output*(1-node_output)*previous_node_output;
            }

        }
        //calculate gradients of output layer bias weights
        for(int i = 0; i<output_layer_bias.matrix.length;i++){
            for(int j = 0; j<output_layer_bias.matrix[i].length;j++){
                double node_output = network_output.matrix[i][j];
                double target = target_output.matrix[i][j];
                double previous_node_output = hidden_layer_output.matrix[i][j];
                double deltaZ =(node_output-target)*node_output*(1-node_output);
                output_layer_bias_derivatives.matrix[i][j] = deltaZ;
            }
        }

        //calculate gradients of hidden layer weights
        for(int i = 0; i<hidden_layer_weights.matrix.length;i++){
            for(int j = 0; j<hidden_layer_weights.matrix[i].length;j++){
                double node_output = network_output.matrix[i][j];
                double target = target_output.matrix[i][j];
                double previous_node_output = network_input.matrix[i][j];
                hidden_layer_weights_derivatives.matrix[i][j] = (node_output-target)*node_output*(1-node_output)*previous_node_output;
            }
        }

        //calculate gradients of hidden layer bias weights
        for(int i = 0; i<hidden_layer_bias.matrix.length;i++){
            for(int j = 0; j<hidden_layer_bias.matrix[i].length;j++){
                double node_output = network_output.matrix[i][j];
                double target = target_output.matrix[i][j];
                double previous_node_output = hidden_layer_output.matrix[i][j];

                double deltaZ =(node_output-target)*node_output*(1-node_output);

                hidden_layer_bias_derivatives.matrix[i][j] = deltaZ*previous_node_output;
            }
        }
    }

    double mean_square_error(Matrix network_output, Matrix target_output){
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
