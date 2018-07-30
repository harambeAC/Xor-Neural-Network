import java.util.Arrays;

public class Matrix {
    public int rows;
    public int columns;
    public double[][] matrix;
    public int length;

    /**
     * Constructor for Matrix
     *
     * @param rows    Number of Rows in Matrix Created
     * @param columns Number of Columns in Matrix Created
     */
    Matrix(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
        this.matrix = new double[rows][columns];
    }

    /**
     * Constructor for Matrix
     *
     * @param matrix The matrix to be created
     */
    Matrix(double[][] matrix) {
        this.matrix = matrix;
        rows = matrix.length;
        columns = matrix[0].length;
    }

    Matrix(double[] matrix) {
        this.matrix = new double[][]{matrix};
        rows = this.matrix.length;
        columns = this.matrix[0].length;
    }

    String getDimensions(){
        return rows + "X" + columns;
    }

    static Matrix multiply(Matrix matrix1, Matrix matrix2) {

        //System.out.println(Arrays.deepToString(matrix1.matrix));
        //System.out.println(Arrays.deepToString(matrix2.matrix));

        if(matrix1.columns !=matrix2.rows){
            throw new Error("Could not multiply matrix of size (" + matrix1.rows + "," + matrix1.columns +
                    ") and matrix of size (" + matrix2.rows + "," + matrix2.columns + ")");
        }

        int constant = matrix1.columns;
        double[][] returnMatrix = new double[matrix1.rows][matrix2.columns];

        for(int i = 0; i<returnMatrix.length; i++){
            for(int j = 0; j<returnMatrix[i].length; j++) {
                for (int k = 0; k < constant; k++) {
                    returnMatrix[i][j] += matrix1.matrix[i][k]*matrix2.matrix[k][j];
                }
            }
        }

        //System.out.println(Arrays.deepToString(returnMatrix));
        return new Matrix(returnMatrix);
    }

    static Matrix add(Matrix matrix1, Matrix matrix2) {
        if(matrix1.rows != matrix2.rows || matrix1.columns != matrix2.columns){
            throw new Error("Could not ad matrix of size (" + matrix1.rows + "," + matrix1.columns +
                    ") to matrix of size (" + matrix2.rows + "," + matrix2.columns + ")");
        }

        double[][] returnMatrix = new double[matrix1.rows][matrix2.columns];

        for(int i = 0; i<returnMatrix.length; i++){
            for(int j = 0; j<returnMatrix[i].length; j++) {
                returnMatrix[i][j] = matrix1.matrix[i][j] + matrix2.matrix[i][j];
            }
        }

        return new Matrix(returnMatrix);
    }
    
    /*public static void main(String[] args) {
    		Matrix m1 = new Matrix(new double[][] {{1,2,3},{1,2,3}});
    	    	Matrix m2 = new Matrix(new double[][] {{1,2,3},{1,2,3}}); 

    	    	System.out.println(Arrays.deepToString(Matrix.add(m1, m2).matrix));
    }*/
}