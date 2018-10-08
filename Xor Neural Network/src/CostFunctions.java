
public class CostFunctions {

	static double mean_square_error(Matrix network_output, Matrix target_output) {
		// System.out.println("network output: "+
		// Arrays.deepToString(network_output.matrix));
		// System.out.println("target output: "+
		// Arrays.deepToString(target_output.matrix));
		double cost = 0;
		for (int i = 0; i < target_output.matrix[0].length; i++) {
			cost += Math.pow(target_output.matrix[0][i] - network_output.matrix[0][i], 2);
		}
		return (cost) / target_output.matrix[0].length;
	}

	static double squared_error(Matrix network_output, Matrix target_output) {
		double cost = 0;
		for (int i = 0; i < target_output.matrix[0].length; i++) {
			cost += Math.pow(target_output.matrix[0][i] - network_output.matrix[0][i], 2);
		}
		return 0.5 * (cost);
	}

	static double root_mean_square_error(Matrix network_output, Matrix target_output) {
		double cost = 0;
		for (int i = 0; i < target_output.matrix[0].length; i++) {
			cost += Math.pow(target_output.matrix[0][i] - network_output.matrix[0][i], 2);
		}
		return Math.sqrt(cost / target_output.matrix[0].length);
	}

	static double sum_square_error(Matrix network_output, Matrix target_output) {
		double cost = 0;
		for (int i = 0; i < target_output.matrix[0].length; i++) {
			cost += Math.pow(target_output.matrix[0][i] - network_output.matrix[0][i], 2);
		}
		return cost;
	}

	static double cross_entropy(Matrix network_output, Matrix target_output) {
		double cost = 0;
		for (int i = 0; i < target_output.matrix[0].length; i++) {
			cost += -1 * target_output.matrix[0][i] * Math.log(network_output.matrix[0][i]);
		}
		return cost;
	}

}
