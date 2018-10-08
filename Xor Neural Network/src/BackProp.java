import java.util.Arrays;

public class BackProp {
	Network net;
	GradientChecker checker;
	WeightUpdater updater;
	GradientCalculator calculator;

	public BackProp(Network network) {
		// TODO Auto-generated constructor stub
		net = network;
		checker = new GradientChecker(net);
		updater = new WeightUpdater(net);
		calculator = new GradientCalculator(net);
	}

	void backpropagation(Matrix target_output) {
		for (int i = 0; i < net.network_input.matrix.length; i++) {
			Matrix network_output = net.feedforward(new Matrix(net.network_input.matrix[i]));
			double cost = CostFunctions.cross_entropy(network_output, new Matrix(target_output.matrix[i]));

			printInfo(network_output, target_output.matrix[i], net.network_input.matrix[i]);
			calculator.calculate_gradient(new Matrix(net.network_input.matrix[i]), network_output,
					new Matrix(target_output.matrix[i]));
			checker.check_gradients(cost, new Matrix(net.network_input.matrix[i]), new Matrix(target_output.matrix[i]));
			printInfo(network_output, target_output.matrix[i], net.network_input.matrix[i]);

			updater.update_weights();
			// System.out.println(Arrays.deepToString(network_input.matrix));

			System.out.println(cost);
		}
	}

	void printInfo(Matrix network_output, double[] target_output, double[] network_input) {
		System.out.println("------------------------------------------------------");
		System.out.println("network input: " + Arrays.toString(network_input));
		System.out.println("network output: " + Arrays.deepToString(network_output.matrix));
		System.out.println("target output: " + Arrays.toString(target_output));
		System.out.println("hidden layer weights" + Arrays.deepToString(net.hidden_layer_weights.matrix));
		System.out.println("hidden layer bias" + Arrays.deepToString(net.hidden_layer_bias.matrix));
		System.out.println("hidden layer output" + Arrays.deepToString(net.hidden_layer_output.matrix));
		System.out.println("output layer weights" + Arrays.deepToString(net.output_layer_weights.matrix));
		System.out.println("output layer bias" + Arrays.deepToString(net.output_layer_bias.matrix));
		System.out.println(
				"hidden layer weights derivative" + Arrays.deepToString(net.hidden_layer_weights_derivatives.matrix));
		System.out.println(
				"hidden layer bias derivative " + Arrays.deepToString(net.hidden_layer_bias_derivatives.matrix));
		System.out.println(
				"output layer weights derivative" + Arrays.deepToString(net.output_layer_weights_derivatives.matrix));
		System.out.println(
				"output layer bias derivative" + Arrays.deepToString(net.output_layer_bias_derivatives.matrix));
	}
}
