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

        Network network = new Network(new Matrix(input_data[0]),new Matrix(target_output[0]));
    }
}
