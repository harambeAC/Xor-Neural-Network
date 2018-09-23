
public class WeightUpdater {
	Network net;

	public WeightUpdater(Network net) {
		// TODO Auto-generated constructor stub
		this.net = net;
	}

	void update_weights() {
		// update hidden layer weights
		for (int i = 0; i < net.hidden_layer_weights.matrix.length; i++) {
			for (int j = 0; j < net.hidden_layer_weights.matrix[i].length; j++) {
				net.hidden_layer_weights.matrix[i][j] -= net.learning_rate
						* net.hidden_layer_weights_derivatives.matrix[i][j];
			}
		}

		// update hidden layer bias weights
		for (int i = 0; i < net.hidden_layer_bias.matrix.length; i++) {
			for (int j = 0; j < net.hidden_layer_bias.matrix[i].length; j++) {
				net.hidden_layer_bias.matrix[i][j] -= net.learning_rate
						* net.hidden_layer_bias_derivatives.matrix[i][j];
			}
		}

		// update output layer weights
		for (int i = 0; i < net.output_layer_weights.matrix.length; i++) {
			for (int j = 0; j < net.output_layer_weights.matrix[i].length; j++) {
				net.output_layer_weights.matrix[i][j] -= net.learning_rate
						* net.output_layer_weights_derivatives.matrix[i][j];
			}
		}

		// update output layer bias weights
		for (int i = 0; i < net.output_layer_bias.matrix.length; i++) {
			for (int j = 0; j < net.output_layer_bias.matrix[i].length; j++) {
				net.output_layer_bias.matrix[i][j] -= net.learning_rate
						* net.output_layer_bias_derivatives.matrix[i][j];
			}
		}
	}
}
