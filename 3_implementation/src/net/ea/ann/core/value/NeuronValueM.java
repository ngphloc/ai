/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value;

import net.ea.ann.core.function.Function;
import net.ea.ann.core.function.FunctionInvertible;

/**
 * This class represents a matrix neuron value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class NeuronValueM extends NeuronValueM0 implements NeuronValue, WeightValue {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with data.
	 * @param data data.
	 */
	private NeuronValueM(double[][] data) {
		super(data);
	}
	
	
	/**
	 * Constructor with numbers of rows and columns along with specified value.
	 * @param rows number of rows.
	 * @param columns numbers of columns.
	 * @param value specified value.
	 */
	protected NeuronValueM(int rows, int columns, double value) {
		super(rows, columns, value);
	}

	
	@Override
	public NeuronValue zero() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public WeightValue zeroW() {
		return (WeightValue)zero();
	}


	@Override
	public NeuronValue unit() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public WeightValue unitW() {
		return (WeightValue)unit();
	}


	@Override
	public int length() {
		return rows()*columns();
	}

	
	@Override
	public int dim() {
		return rows();
	}

	
	@Override
	public NeuronValue resize(int newDim) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue duplicate() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public boolean equals(NeuronValue value) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public WeightValue newWeightValue() {
		return (WeightValue)zero();
	}

	
	@Override
	public WeightValue toWeightValue() {
		return this;
	}

	
	@Override
	public NeuronValue toValue() {
		return this;
	}


	@Override
	public NeuronValue negative() {
		return (NeuronValue)negative0();
	}

	
	@Override
	public NeuronValue abs() {
		int m = data.length, n = data[0].length;
		double[][] abs = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) abs[i][j] = Math.abs(data[i][j]);
		}
		return (NeuronValue)wrap(abs);
	}


	@Override
	public boolean canInvert() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public boolean canInvertWise() {
		throw new RuntimeException("Not implemented yet");
	}


	@Override
	public NeuronValue inverse() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue add(NeuronValue value) {
		return (NeuronValue)add((Matrix)value);
	}

	
	@Override
	public WeightValue addValue(NeuronValue value) {
		return (WeightValue)add(value);
	}


	@Override
	public NeuronValue subtract(NeuronValue value) {
		return (NeuronValue)subtract((Matrix)value);
	}

	
	@Override
	public WeightValue subtractValue(NeuronValue value) {
		return (WeightValue)subtract(value);
	}


	@Override
	public NeuronValue multiply(NeuronValue value) {
		return (NeuronValue)multiply0(value);
	}

	
	@Override
	public NeuronValue multiply(WeightValue value) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue multiply(double value) {
		return (NeuronValue)multiply0(value);
	}

	
	@Override
	public NeuronValue multiplyDerivative(NeuronValue derivative) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue divide(NeuronValue value) {
		return (NeuronValue)divide0(value);
	}

	
	@Override
	public NeuronValue divide(double value) {
		return (NeuronValue)divide0(value);
	}

	
	@Override
	public NeuronValue power(double exponent) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue sqrt() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue exp() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue log() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public double mean() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public double norm() {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue valueOf(double value) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue min(NeuronValue value) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue max(NeuronValue value) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public boolean matrixIsInvertible(NeuronValue[][] matrix) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue matrixDet(NeuronValue[][] matrix) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue[][] matrixInverse(NeuronValue[][] matrix) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue[][] matrixSqrt(NeuronValue[][] matrix) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue[] flatten(int smallerDim) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue[] flatten(NeuronValue[] array, int smallerDim) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue aggregate(NeuronValue[] array) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue[] aggregate(NeuronValue[] array, int largerDim) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue evaluate(Function f) {
		return (NeuronValue)evaluate0(f);
	}

	
	@Override
	public NeuronValue derivative(Function f) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue evaluateInverse(FunctionInvertible f) {
		throw new RuntimeException("Not implemented yet");
	}

	
	@Override
	public NeuronValue derivativeInverse(FunctionInvertible f) {
		throw new RuntimeException("Not implemented yet");
	}


	@Override
	protected Matrix wrap(double[][] data) {
		if (data == null || data.length == 0) return null;
		int n = data[0].length;
		if (n == 0) return null;
		for (int i = 1; i < data.length; i++) {
			if (data[i] == null || data[i].length != n) return null;
		}
		return new NeuronValueM(data);
	}


	@Override
	public Matrix create(int rows, int columns) {
		if (rows <= 0 || columns <= 0)
			return null;
		else
			return new NeuronValueM(rows, columns, 0);
	}


	/**
	 * Creating matrix from data array.
	 * @param data data array.
	 * @return matrix.
	 */
	public static Matrix create(double[][] data) {
		return new NeuronValueM(null).wrap(data);
	}

	
}



/**
 * This class represents basic matrix neuron value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
abstract class NeuronValueM0 implements Matrix {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Internal data.
	 */
	protected double[][] data = null;

	
	/**
	 * Constructor with data.
	 * @param data data.
	 */
	protected NeuronValueM0(double[][] data) {
		this.data = data;
	}
	
	
	/**
	 * Constructor with numbers of rows and columns along with specified value.
	 * @param rows number of rows.
	 * @param columns numbers of columns.
	 * @param value specified value.
	 */
	protected NeuronValueM0(int rows, int columns, double value) {
		if (rows <= 0 || columns <= 0 || Double.isNaN(value)) throw new IllegalArgumentException("Wrong rows, columns, or value");
		data = new double[rows][columns];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) data[i][j] = value;
		}
	}

	
	/**
	 * Wrapping data as matrix.
	 * @param data specified data.
	 * @return wrapped matrix.
	 */
	protected abstract Matrix wrap(double[][] data);

	
	@Override
	public NeuronValue newNeuronValue() {
		return create(1, 1).get(0, 0).zero();
	}


	@Override
	public int rows() {return data.length;}


	@Override
	public int columns() {return data.length > 0 ? data[0].length : 0;}


	@Override
	public NeuronValue get(int row, int column) {
		return new NeuronValue1(data[row][column]);
	}


	@Override
	public void set(int row, int column, NeuronValue value) {
		data[row][column] = ((NeuronValue1)value).v;
	}


	@Override
	public Matrix getRow(int row) {
		int n = columns();
		double[][] newdata = new double[1][n];
		for (int j = 0; j < n; j++) newdata[0][j] = this.data[row][j];
		return wrap(newdata);
	}


	@Override
	public Matrix getColumn(int column) {
		int m = rows();
		double[][] newdata = new double[m][1];
		for (int i = 0; i < m; i++) newdata[i][0] = this.data[i][column];
		return wrap(newdata);
	}


	@Override
	public Matrix getColumns(int column, int range) {
		column = column < 0 ? 0 : column;
		range = column + range <= this.columns() ? range : this.columns() - column;
		if (range <= 0) return null;

		int m = this.rows();
		double[][] newMatrixData = new double[m][range];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < range; j++) {
				newMatrixData[i][j] = this.data[i][column + j];
			}
		}
		return wrap(newMatrixData);
	}

	
	@Override
	public Matrix transpose() {
		if (data == null) return null;
		int m = data.length, n = data[0].length;
		if (m == 1 && n == 1) return this;
		
		double[][] trans = new double[n][m];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) trans[i][j] = data[j][i];
		}
		return wrap(trans);
	}


	@Override
	public Matrix negative0() {
		int m = data.length, n = data[0].length;
		double[][] negated = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) negated[i][j] = -data[i][j];
		}
		return wrap(negated);
	}


	@Override
	public Matrix add(Matrix other) {
		double[][] A = this.data;
		double[][] B = ((NeuronValueM0)other).data;
		int m = A.length, n = A[0].length;
		double[][] C = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) C[i][j] = A[i][j] + B[i][j];
		}
		return wrap(C);
	}


	@Override
	public Matrix subtract(Matrix other) {
		double[][] A = this.data;
		double[][] B = ((NeuronValueM0)other).data;
		int m = A.length, n = A[0].length;
		double[][] C = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) C[i][j] = A[i][j] - B[i][j];
		}
		return wrap(C);
	}


	@Override
	public Matrix multiply(Matrix other) {
		double[][] A = this.data;
		double[][] B = ((NeuronValueM0)other).data;
		double[][] C = multiply(A, B);
		return C != null ? wrap(C) : null;
	}


	@Override
	public Matrix multiply0(NeuronValue value) {
		return multiply0( ((NeuronValue1)value).v );
	}


	@Override
	public Matrix multiply0(double value) {
		int m = this.data.length, n =  this.data[0].length;
		double[][] result = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) result[i][j] = this.data[i][j]*value;
		}
		return wrap(result);
	}


	@Override
	public Matrix multiplyWise(Matrix other) {
		int m = Math.min(this.rows(), other.rows());
		int n = Math.min(this.columns(), other.columns());
		Matrix result = create(m, n);
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				NeuronValue value = this.get(i, j).multiply(other.get(i, j));
				result.set(i, j, value);
			}
		}
		return result;
	}

	
	@Override
	public Matrix divide0(NeuronValue value) {
		return divide0( ((NeuronValue1)value).v );
	}


	@Override
	public Matrix divide0(double value) {
		if (value == 0) return null;
		int m = this.data.length, n =  this.data[0].length;
		double[][] result = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) result[i][j] = this.data[i][j]/value;
		}
		return wrap(result);
	}


	@Override
	public Matrix kroneckerProductRowOf(Matrix other, int rowOfThis) {
		Matrix[] matrices = new Matrix[this.columns()];
		for (int j = 0; j < matrices.length; j++) {
			matrices[j] = other.multiply0(this.get(rowOfThis, j));
		}
		return concatVertical(matrices);
	}

	
	@Override
	public Matrix evaluate0(Function f) {
		if (f == null) return null;
		Matrix result = create(this.rows(), this.columns());
		for (int i = 0; i < this.rows(); i++) {
			for (int j = 0; j < this.columns(); j++) {
				result.set(i, j, this.get(i, j).evaluate(f));
			}
		}
		return result;
	}

	
	@Override
	public Matrix derivativeWise(Function f) {
		Matrix result = create(this.rows(), this.columns());
		for (int i = 0; i < this.rows(); i++) {
			for (int j = 0; j < this.columns(); j++) {
				result.set(i, j, this.get(i, j).derivative(f));
			}
		}
		return result;
	}

	
	@Override
	public Matrix concatHorizontal(Matrix... matrices) {
		if (matrices == null || matrices.length == 0) return null;
		int m = 0, n = matrices[0].columns();
		for (Matrix matrix : matrices) {
			m += matrix.rows();
			n = Math.min(n, matrix.columns());
		}
		
		double[][] newMatrixData = new double[m][n];
		for (int j = 0; j < n; j++) {
			int i = 0;
			for (int k = 0; k < matrices.length; k++) {
				double[][] matrixData = ((NeuronValueM0)matrices[k]).data;
				int rows = matrices[k].rows();
				for (int l = 0; l < rows; l++) {
					newMatrixData[i][j] = matrixData[l][j];
					i++;
				}
			}
		}
		return wrap(newMatrixData);
	}


	@Override
	public Matrix concatVertical(Matrix... matrices) {
		if (matrices == null || matrices.length == 0) return null;
		int m = matrices[0].rows(), n = 0;
		for (Matrix matrix : matrices) {
			m = Math.min(m, matrix.rows());
			n += matrix.columns();
		}
		
		double[][] newMatrixData = new double[m][n];
		for (int i = 0; i < m; i++) {
			int j = 0;
			for (int k = 0; k < matrices.length; k++) {
				double[][] matrixData = ((NeuronValueM0)matrices[k]).data;
				int columns = matrices[k].columns();
				for (int l = 0; l < columns; l++) {
					newMatrixData[i][j] = matrixData[i][l];
					j++;
				}
			}
		}
		return wrap(newMatrixData);
	}


	@Override
	public Matrix vec() {
		if (this.columns() == 1) return this;
		double[][] result = new double[this.rows()*this.columns()][1];
		int k = 0;
		for (int j = 0; j < this.columns(); j++) {
			for (int i = 0; i < this.rows(); i++) {
				result[k][0] = this.data[i][j];
				k++;
			}
		}
		return wrap(result);
	}


	@Override
	public Matrix vecInverse(int rows) {
		if (rows <= 0) return null;
		int columns = this.rows() / rows;
		if (columns == 0) return null;
		
		double[][] result = new double[rows][columns];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				int index = j*rows + i;
				result[i][j] = this.data[index][0];
			}
		}
		return wrap(result);
	}

	
	/**
	 * Multiplying two matrices.
	 * @param A first matrix.
	 * @param B second matrix.
	 * @return multiplied matrix.
	 */
	public static double[][] multiply(double[][] A, double[][] B) {
		if (A == null || B == null) return null;
		int m = A.length, n = B.length, o = B[0].length;
		if (m == 0 || n == 0 || o == 0) return null;
		if (A[0].length != n) return null;
		
		double[][] C = new double[m][o];
		double zero = 0;
		for (int i = 0; i < m; i++) {
			for (int k = 0; k < o; k++) {
				C[i][k] = zero;
				for (int j = 0; j < n; j++) C[i][k] += A[i][j]*B[j][k];
			}
		}
		return C;
	}
	
	
	@Override
	public Matrix createIdentity(int n) {
		double[][] identityData = new double[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				identityData[i][j] = i == j ? 1 : 0;
			}
		}
		return wrap(identityData);
	}

	
	/**
	 * Generate cofactor of matrix, the code is available at https://stackoverflow.com/questions/16602350/calculating-matrix-determinant
	 * @param A specific squared matrix.
	 * @param removedRow removed row.
	 * @param removedColumn removed column.
	 * @return cofactor of matrix.
	 */
	private static double[][] genCofactor(double A[][], int removedRow, int removedColumn) {
		int n = A.length;
		double[][] co = new double[n-1][];
		for (int k = 0; k < n-1; k++) co[k] = new double[n-1];

		int k = 0;
		for (int i = 0; i < n; i++) {
			if (i == removedRow) continue;
			
			int l = 0;
			for (int j = 0; j < n; j++){
				if(j != removedColumn) {
					co[k][l] = A[i][j];
					l++;
				}
			}
			k++;
		}
		
		return co;
	}
	
	
	/**
	 * Calculate determinant recursively, the code is available at https://stackoverflow.com/questions/16602350/calculating-matrix-determinant
	 * @param A specific squared matrix.
	 * @return determinant of specific matrix.
	 */
	private static double det0(double[][] A) {
		int n = A.length;
		if (n == 1) return A[0][0];
		if (n == 2) return A[0][0]*A[1][1] - A[1][0]*A[0][1];
		
		double det = 0;
		for (int j = 0; j < n; j++){
			double[][] co = genCofactor(A, 0, j);
			det += Math.pow(-1.0, j) * A[0][j] * det0(co);
		}
		
		return det;
	}
	
	
	/**
	 * Calculate determinant recursively, the code is available at https://stackoverflow.com/questions/16602350/calculating-matrix-determinant
	 * @param A specific squared matrix.
	 * @return determinant of specific matrix.
	 */
	public static double detNotOptimalYet(double[][] A) {
		if (A == null) return Double.NaN;
		int n = A.length;
		if (n == 0) return Double.NaN;
		
		return det0(A);
	}


	/**
	 * Calculating the inverse of specific matrix.
	 * @param A specific matrix.
	 * @return the inverse of specific matrix.
	 */
	public static double[][] inverseNotOptimalYet(double[][] A) {
		if (A == null) return null;
		int n = A.length;
		if (n == 0) return null;
		
		if (A.length == 1) return A[0][0] != 0 ? new double[][] {{1.0/A[0][0]}} : null;
		
		double[][] B = new double[n][];
		for (int i = 0; i < n; i++) B[i] = new double[n];
		
		double det = detNotOptimalYet(A);
		if (Double.isNaN(det) || det == 0) return null;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				double[][] co = genCofactor(A, i, j);
				B[j][i] = Math.pow(-1.0, i+j) * detNotOptimalYet(co) / det;
			}
		}
		return B;
	}
	
	
	/**
	 * Checking if the given matrix is invertible.
	 * @param A given matrix.
	 * @return if the given matrix is invertible.
	 */
	public static boolean isInvertible(double[][] A) {
		try {
		    return net.ea.ann.adapter.Util.isInvertible(A);
		}
		catch (Throwable e) {
			if (e instanceof ClassNotFoundException) {
				double det = detNotOptimalYet(A);
				return (!Double.isNaN(det)) && (det != 0);
			}
			System.out.println("Checking if matrix is invertible causes error: " + e.getMessage());
		}
		return false;
	}

	
	/**
	 * Checking if the given matrix is invertible.
	 * @param A given matrix.
	 * @return if the given matrix is invertible.
	 */
	public static double det(double[][] A) {
		try {
		    return net.ea.ann.adapter.Util.det(A);
		}
		catch (Throwable e) {
			if (e instanceof ClassNotFoundException) return detNotOptimalYet(A);
			System.out.println("Calculating matrix determinant causes error: " + e.getMessage());
		}
		return Double.NaN;
	}

	
	/**
	 * Calculating inverse of the given matrix.
	 * @param A given matrix.
	 * @return inverse of the given matrix. Return null if the matrix is not invertible.
	 */
	public static double[][] inverse(double[][] A) {
		try {
		    return net.ea.ann.adapter.Util.inverse(A);
		}
		catch (Throwable e) {
			if (e instanceof ClassNotFoundException) return inverseNotOptimalYet(A);
			System.out.println("Calculating matrix inverse causes error: " + e.getMessage());
		}
		return null;
	}
	
	
	/**
	 * Calculating pseudo square root of the given matrix.
	 * @param A given matrix.
	 * @return pseudo square root of the given matrix.
	 */
	public static double[][] sqrtNotOptimalYet(double[][] A) {
		if (A == null || A.length == 0 || A[0].length == 0 || A.length != A[0].length) return null;
		if (A.length == 1) return A[0][0] >= 0 ? new double[][] {{Math.sqrt(A[0][0])}} : null;
		
		for (int i = 0; i < A.length; i++) {
			for (int j = 0; j < A.length; j++) if (A[i][j] < 0) return null;
		}
		
		double[][] S = new double[A.length][A.length];
		for (int i = 0; i < A.length; i++) {
			for (int j = 0; j < A.length; j++) S[i][j] = Math.sqrt(A[i][j]);
		}
		return S;
	}
	
	
	/**
	 * Calculating square root of the given matrix.
	 * @param A given matrix.
	 * @return square root of the given matrix.
	 */
	public static double[][] sqrt(double[][] A) {
		if (A == null || A.length == 0) return null;
		if (A.length == 1) return A[0][0] >= 0 ? new double[][] {{Math.sqrt(A[0][0])}} : null;
		
		boolean specialTechnique = false;
		for (int i = 0; i < A.length; i++) {
			for (int j = 0; j < A.length; j++) {
				specialTechnique = ((i != j) && (A[i][j] != 0)) || ((i == j) && (A[i][j] < 0));
				if (specialTechnique) break;
			}
		}
		
		if (specialTechnique) {
			try {
			    return net.ea.ann.adapter.Util.sqrt(A);
			}
			catch (Throwable e) {
				if (e instanceof ClassNotFoundException)
					return sqrtNotOptimalYet(A);
				System.out.println("Calculating matrix square root causes error: " + e.getMessage());
			}
			return null;
		}
		else {
			double[][] S = new double[A.length][A.length];
			for (int i = 0; i < A.length; i++) {
				for (int j = 0; j < A.length; j++) S[i][j] = i == j ? Math.sqrt(A[i][j]) : 0;
			}
			return S;
		}
	}


}
