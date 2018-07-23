import java.util.Arrays;

public class Layer {
    int size;

    Layer(int numNodes){
        this.size = numNodes;
    }

    Matrix summation(Matrix weights, Matrix input, Matrix bias){
        return Matrix.add(Matrix.multiply(weights,input),bias);
    }

    static Matrix sigmoid(Matrix input){
        double[][] returnMatrix = new double[1][input.matrix[0].length];
        for(int i = 0; i<input.matrix.length; i++){
            returnMatrix[0][i] = 1/(Math.exp(-1.0*input.matrix[0][i])+1.0);
        }
        return new Matrix(returnMatrix);
    }

    static Matrix relu(Matrix input) {
        double[][] returnMatrix = new double[1][input.matrix[0].length];
        for(int i = 0; i<input.matrix.length; i++){
            returnMatrix[0][i] = Math.max(input.matrix[0][i],0);;
        }
        return new Matrix(returnMatrix);
    }

    static Matrix step(Matrix input){
        double[][] returnMatrix = new double[1][input.matrix[0].length];
        for(int i = 0; i<input.matrix.length; i++){
            returnMatrix[0][i] = input.matrix[0][i] >= 0 ? 1:0;;
        }
        return new Matrix(returnMatrix);
    }

    static Matrix linear(Matrix input){
        double[][] returnMatrix = new double[1][input.matrix[0].length];
        for(int i = 0; i<input.matrix.length; i++){
            returnMatrix[0][i] = input.matrix[0][i];
        }
        return new Matrix(returnMatrix);
    }

    static Matrix tanh(Matrix input){
        double[][] returnMatrix = new double[1][input.matrix[0].length];
        for(int i = 0; i<input.matrix.length; i++){
            returnMatrix[0][i] = (Math.exp(input.matrix[0][i])-Math.exp(-1*input.matrix[0][i]))/
                                    (Math.exp(-1*input.matrix[0][i])+Math.exp(input.matrix[0][i]));;
        }
        return new Matrix(returnMatrix);
    }


}
