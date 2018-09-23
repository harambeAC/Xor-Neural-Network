/**
 * The HelloWorld program implements an application that simply displays "Hello
 * World!" to the standard output.
 *
 * @author Alexander Chen
 * @since 2018-07-20
 */

public class Xor {
	public static void main(String[] args) {
		double[][] input_data = new double[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 }, };

		// index 0 = 0
		// index 1 = 1
		double[][] target_output = new double[][] { { 1, 0 }, { 0, 1 },

				{ 0, 1 }, { 1, 0 } };

		double[] target_output2 = new double[] { 0, 1, 1, 0 };

		Network network = new Network(new Matrix(input_data), new Matrix(target_output));

		double accuracy_numerator = 0;
		double accuracy_denominator = 0;
		double accuracy = 0;

		for (int i = 0; i < input_data.length; i++) {
			if (network.feedforward(new Matrix(input_data[i])).equals(target_output2[i])) {
				accuracy_numerator++;
				accuracy_denominator++;
			} else {
				accuracy_denominator++;
			}
			accuracy = accuracy_numerator / accuracy_denominator;
		}

		System.out.println("Final Accuracy = " + accuracy);
	}
}
