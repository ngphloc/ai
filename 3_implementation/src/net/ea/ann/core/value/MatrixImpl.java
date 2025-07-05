/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value;

/**
 * This class implements default matrix.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MatrixImpl implements Matrix {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal data.
	 */
	protected NeuronValue[][] data = null;
	
	
	/**
	 * Constructor with internal data.
	 * @param data internal data.
	 */
	private MatrixImpl(NeuronValue[][] data) {
		this.data = data;
	}
	
	
	/**
	 * Constructor with numbers of rows and columns along with specified value.
	 * @param rows number of rows.
	 * @param columns numbers of columns.
	 * @param value specified value.
	 */
	public MatrixImpl(int rows, int columns, NeuronValue value) {
		data = new NeuronValue[rows][columns];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) data[i][j] = value;
		}
	}

	
	/**
	 * Constructor with numbers of rows and columns along with specified double value.
	 * @param rows number of rows.
	 * @param columns numbers of columns.
	 * @param value specified double value.
	 */
	public MatrixImpl(int rows, int columns, double value) {
		this(rows, columns, new NeuronValue1(value));
	}
	
	
	/**
	 * Constructor with numbers of rows and columns along.
	 * @param rows number of rows.
	 * @param columns numbers of columns.
	 */
	public MatrixImpl(int rows, int columns) {
		this(rows, columns, 0);
	}


	@Override
	public int rows() {return data.length;}
	
	
	@Override
	public int columns() {return data.length > 0 ? data[0].length : 0;}
	
	
	@Override
	public NeuronValue get(int row, int column) {
		return data[row][column];
	}
	
	
	@Override
	public void set(int row, int column, NeuronValue value) {
		data[row][column] = value;
	}

	
	@Override
	public Matrix getRow(int row) {
		int n = columns();
		NeuronValue[][] newdata = new NeuronValue[1][n];
		for (int j = 0; j < n; j++) newdata[0][j] = this.data[row][j];
		return new MatrixImpl(newdata);
	}


	@Override
	public Matrix transpose() {
		return new MatrixImpl(NeuronValue.transpose(data));
	}
	
	
	@Override
	public Matrix negative0() {
		return new MatrixImpl(NeuronValue.negative(data));
	}

	
	@Override
	public Matrix add(Matrix other) {
		return new MatrixImpl(NeuronValue.add(this.data, ((MatrixImpl)other).data));
	}

	
	@Override
	public Matrix subtract(Matrix other) {
		return new MatrixImpl(NeuronValue.subtract(this.data, ((MatrixImpl)other).data));
	}

	
	@Override
	public Matrix multiply(Matrix other) {
		return new MatrixImpl(NeuronValue.multiply(this.data, ((MatrixImpl)other).data));
	}
	
	
	@Override
	public Matrix multiply0(NeuronValue value) {
		return new MatrixImpl(NeuronValue.multiply(this.data, value));
	}


	@Override
	public Matrix multiply0(double value) {
		return new MatrixImpl(NeuronValue.multiply(this.data, value));
	}


	@Override
	public Matrix divide0(NeuronValue value) {
		return new MatrixImpl(NeuronValue.divide(this.data, value));
	}


	@Override
	public Matrix divide0(double value) {
		return new MatrixImpl(NeuronValue.divide(this.data, value));
	}


	@Override
	public Matrix concatHorizontal(Matrix... matrices) {
		if (matrices == null || matrices.length == 0) return null;
		int m = 0, n = matrices[0].columns();
		for (Matrix matrix : matrices) {
			m += matrix.rows();
			n = Math.min(n, matrix.columns());
		}
		
		Matrix newMatrix = new MatrixImpl(m, n);
		for (int j = 0; j < n; j++) {
			int i = 0;
			for (int k = 0; k < matrices.length; k++) {
				Matrix matrix = matrices[k];
				for (int l = 0; l < matrix.rows(); l++) {
					newMatrix.set(i, j, matrix.get(l, j));
					i++;
				}
			}
		}
		return newMatrix;
	}


	@Override
	public Matrix concatVertical(Matrix...matrices) {
		if (matrices == null || matrices.length == 0) return null;
		int m = matrices[0].rows(), n = 0;
		for (Matrix matrix : matrices) {
			m = Math.min(m, matrix.rows());
			n += matrix.columns();
		}
		
		Matrix newMatrix = new MatrixImpl(m, n);
		for (int i = 0; i < m; i++) {
			int j = 0;
			for (int k = 0; k < matrices.length; k++) {
				Matrix matrix = matrices[k];
				for (int l = 0; l < matrix.columns(); l++) {
					newMatrix.set(i, j, matrix.get(i, l));
					j++;
				}
			}
		}
		return newMatrix;
	}
	
	
	@Override
	public Matrix extractVertical(int columnIndex, int range) {
		columnIndex = columnIndex < 0 ? 0 : columnIndex;
		range = columnIndex + range <= this.columns() ? range : this.columns() - columnIndex;
		if (range <= 0) return null;

		Matrix newMatrix = new MatrixImpl(this.rows(), range);
		for (int i = 0; i < newMatrix.rows(); i++) {
			for (int j = 0; j < newMatrix.columns(); j++) {
				newMatrix.set(i, j, this.get(i, columnIndex + j));
			}
		}
		return newMatrix;
	}
	
	
}
