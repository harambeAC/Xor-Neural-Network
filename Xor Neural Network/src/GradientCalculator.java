
public class GradientCalculator {
	Network net;

	public GradientCalculator(Network net) {
		// TODO Auto-generated constructor stub
		this.net = net;
	}

	void calculate_gradient(Matrix network_input, Matrix network_output, Matrix target_output) {
		// calculate gradients of output layer weights
		for (int j = 0; j < net.output_layer_weights.matrix.length; j++) {
			for (int i = 0; i < net.output_layer_weights.matrix[i].length; i++) { // since its flip flopped, a 5X2
																					// matrix. first index is which
																					// weight,second is which node
				double node_output = network_output.matrix[0][i];
				double target = target_output.matrix[0][i];
				double previous_node_output = net.hidden_layer_output.matrix[0][j];
				net.output_layer_weights_derivatives.matrix[j][i] = (node_output - target) * node_output
						* (1 - node_output) * previous_node_output;
			}

		}

		// calculate gradients of output layer bias weights
		for (int i = 0; i < net.output_layer_bias.matrix.length; i++) {
			for (int j = 0; j < net.output_layer_bias.matrix[i].length; j++) {
				double node_output = network_output.matrix[0][j];
				double target = target_output.matrix[0][j];
				double deltaZ = (node_output - target) * node_output * (1 - node_output);
				net.output_layer_bias_derivatives.matrix[i][j] = deltaZ;
			}
		}

		// calculate gradients of hidden layer weights
		for (int i = 0; i < net.hidden_layer_weights.matrix.length; i++) { // 2 options
			for (int j = 0; j < net.hidden_layer_weights.matrix[i].length; j++) { // 5 options-
				double node_output = network_output.matrix[0][i];
				double target = target_output.matrix[0][i];
				double previous_node_output = network_input.matrix[0][i];
				double next_node_output = net.hidden_layer_output.matrix[0][j];

				double deltaZ = (node_output - target) * node_output * (1 - node_output);

				double sum = 0;
				for (int k = 0; k < net.output_layer_weights.matrix.length; k++) {
					sum += deltaZ * net.output_layer_weights.matrix[k][i];
				}

				net.hidden_layer_weights_derivatives.matrix[i][j] = (sum) * previous_node_output
						* (1 - next_node_output) * next_node_output;
			}
		}

		// calculate gradients of hidden layer bias weights
		for (int i = 0; i < net.inputLayerNodes; i++) { // 2
			for (int j = 0; j < net.hidden_layer_bias.matrix[0].length; j++) { // 5
				double node_output = network_output.matrix[0][i];
				double target = target_output.matrix[0][i];
				double next_node_output = net.hidden_layer_output.matrix[0][j];

				double deltaZ = (node_output - target) * node_output * (1 - node_output);

				double sum = 0;
				for (int k = 0; k < net.output_layer_weights.matrix.length; k++) {
					sum += deltaZ * net.output_layer_weights.matrix[k][i];
				}

				net.hidden_layer_bias_derivatives.matrix[0][j] = (sum) * (1 - next_node_output) * next_node_output;
			}
		}
	}
}
