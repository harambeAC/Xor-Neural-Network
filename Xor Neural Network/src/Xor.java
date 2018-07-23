/**
 * The HelloWorld program implements an application that
 * simply displays "Hello World!" to the standard output.
 *
 * @author  Alexander Chen
 * @since   2018-07-20
 */

public class Xor {
    public static void main(String[] args){
        double[][] input_data = new double[][]{
                {0,0},
                {0,1},
                {1,0},
                {1,1}
        };

        //index 0 = 0
        //index 1 = 1
        double[][] target_output = new double[][]{
                {1,0},
                {0,1},
                {0,1},
                {1,0}
        };

        Network network = new Network(new Matrix(input_data),new Matrix(target_output));
        
        double[] test = input_data[0];
        System.out.println(network.test(new Matrix(test)));
        test = input_data[1];
        System.out.println(network.test(new Matrix(test)));
        test = input_data[2];
        System.out.println(network.test(new Matrix(test)));
        test = input_data[3];
        System.out.println(network.test(new Matrix(test)));

    }
}
