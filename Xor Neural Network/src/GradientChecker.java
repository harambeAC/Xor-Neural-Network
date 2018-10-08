
public class GradientChecker {
	Network net;

	public GradientChecker(Network net) {
		// TODO Auto-generated constructor stub
		this.net = net;
	}

	void check_gradients(double cost, Matrix network_input, Matrix target_output) { // prints relative Error
		double epsilon = 0.0001;
		// check for hidden layer weights
		for (int i = 0; i < net.hidden_layer_weights.matrix.length; i++) {
			for (int j = 0; j < net.hidden_layer_weights.matrix[i].length; j++) {
				net.hidden_layer_weights.matrix[i][j] += epsilon;
				double error2 = CostFunctions.mean_square_error(net.feedforward(network_input), target_output);

				net.hidden_layer_weights.matrix[i][j] -= 2 * epsilon;
				double error1 = CostFunctions.mean_square_error(net.feedforward(network_input), target_output);

				double numerical_gradient = (error1 - error2) / (2 * epsilon);

				net.hidden_layer_weights.matrix[i][j] += epsilon;
			}
		}

		// check for hidden layer bias
		for (int i = 0; i < net.hidden_layer_bias.matrix.length; i++) {
			for (int j = 0; j < net.hidden_layer_bias.matrix[i].length; j++) {
				net.hidden_layer_bias.matrix[i][j] += epsilon;
				double error2 = CostFunctions.mean_square_error(net.feedforward(network_input), target_output);

				net.hidden_layer_bias.matrix[i][j] -= 2 * epsilon;
				double error1 = CostFunctions.mean_square_error(net.feedforward(network_input), target_output);

				double numerical_gradient = (error1 - error2) / (2 * epsilon);
				// System.out.println(cost - numerical_gradient);

				net.hidden_layer_bias.matrix[i][j] += epsilon;
			}
		}

		// check for output layer weights
		for (int i = 0; i < net.output_layer_weights.matrix.length; i++) {
			for (int j = 0; j < net.output_layer_weights.matrix[i].length; j++) {
				net.output_layer_weights.matrix[i][j] += epsilon;
				double error2 = CostFunctions.mean_square_error(net.feedforward(network_input), target_output);

				net.output_layer_weights.matrix[i][j] -= 2 * epsilon;
				double error1 = CostFunctions.mean_square_error(net.feedforward(network_input), target_output);

				double numerical_gradient = (error1 - error2) / (2 * epsilon);
				// System.out.println(cost - numerical_gradient);

				net.output_layer_weights.matrix[i][j] += epsilon;
			}
		}

		// check for output layer bias
		for (int i = 0; i < net.output_layer_bias.matrix.length; i++) {
			for (int j = 0; j < net.output_layer_bias.matrix[i].length; j++) {
				net.output_layer_bias.matrix[i][j] += epsilon;
				double error2 = CostFunctions.mean_square_error(net.feedforward(network_input), target_output);

				net.output_layer_bias.matrix[i][j] -= 2 * epsilon;
				double error1 = CostFunctions.mean_square_error(net.feedforward(network_input), target_output);

				double numerical_gradient = (error1 - error2) / (2 * epsilon);
				// System.out.println(cost - numerical_gradient);

				net.output_layer_bias.matrix[i][j] += epsilon;
			}
		}
	}

}
