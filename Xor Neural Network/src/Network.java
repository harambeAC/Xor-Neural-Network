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

	// my hyperparameters
	final int inputLayerNodes = 2;
	final int hiddenLayerNodes = 5;
	final int outputLayerNodes = 2;
	double learning_rate = 1;

	Network(Matrix network_input, Matrix target_output) { // trains network
		this.network_input = network_input;
		double[][] hidden_layer_weights = new double[inputLayerNodes][hiddenLayerNodes]; // 2*2
		double[][] hidden_layer_bias = new double[1][hiddenLayerNodes];
		double[][] output_layer_weights = new double[hiddenLayerNodes][outputLayerNodes]; // 2*2
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

		// train
		BackProp train = new BackProp(this);
		train.backpropagation(target_output);
	}

	void initialize(double[][] arr) {
		for (int i = 0; i < arr.length; i++) {
			for (int j = 0; j < arr[i].length; j++) {
				arr[i][j] = Math.random();
			}
		}
	}

	Matrix feedforward(Matrix network_input) {
		this.input_layer = new Layer(this.inputLayerNodes); // output is already determined
		this.hidden_layer = new Layer(this.hiddenLayerNodes);

		Matrix sum = this.hidden_layer.summation(this.hidden_layer_weights, network_input, this.hidden_layer_bias);
		this.hidden_layer_output = Layer.sigmoid(sum);

		this.output_layer = new Layer(this.outputLayerNodes);
		sum = hidden_layer.summation(this.output_layer_weights, this.hidden_layer_output, this.output_layer_bias);

		return Layer.sigmoid(sum);
	}
}
