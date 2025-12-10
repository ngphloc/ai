/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value;

import net.ea.ann.conv.Content;
import net.ea.ann.core.function.Function;
import net.ea.ann.raster.Size;

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
	 * @param size size.
	 * @param value specified value.
	 */
	protected MatrixImpl(Size size, NeuronValue value) {
		if (size.height <= 0 || size.width <= 0 || value == null) throw new IllegalArgumentException("Wrong rows, columns, or value");
		data = new NeuronValue[size.height][size.width];
		for (int i = 0; i < size.height; i++) {
			for (int j = 0; j < size.width; j++) data[i][j] = value;
		}
	}

	
	/**
	 * Constructor with numbers of rows and columns along with specified double value.
	 * @param size size.
	 * @param value specified double value.
	 */
	protected MatrixImpl(Size size, double value) {
		this(size, new NeuronValue1(value));
	}
	
	
	/**
	 * Constructor with numbers of rows and columns along.
	 * @param size size.
	 */
	protected MatrixImpl(Size size) {
		this(size, 0);
	}


	@Override
	public NeuronValue newNeuronValue() {
		NeuronValue value = create(new Size(1, 1)).get(0, 0);
		if (value instanceof Matrix)
			return value;
		else if (value instanceof Content)
			return value;
		else
			return value.zero();
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
	public Matrix getColumn(int column) {
		int m = rows();
		NeuronValue[][] newdata = new NeuronValue[m][1];
		for (int i = 0; i < m; i++) newdata[i][0] = this.data[i][column];
		return new MatrixImpl(newdata);
	}


	@Override
	public Matrix getColumns(int column, int range) {
		column = column < 0 ? 0 : column;
		range = column + range <= this.columns() ? range : this.columns() - column;
		if (range <= 0) return null;

		Matrix newMatrix = new MatrixImpl(new Size(range, this.rows()));
		for (int i = 0; i < newMatrix.rows(); i++) {
			for (int j = 0; j < newMatrix.columns(); j++) {
				newMatrix.set(i, j, this.get(i, column + j));
			}
		}
		return newMatrix;
	}


	@Override
	public Matrix transpose() {
		return new MatrixImpl(NeuronValue.transpose(data));
	}
	
	
	@Override
	public Matrix negative0() {
		return new MatrixImpl(NeuronValue.negative(data));
	}

	
	/**
	 * Getting absolute matrix.
	 * @return absolute matrix.
	 */
	public Matrix abs0() {
		return new MatrixImpl(NeuronValue.abs(data));
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
	public Matrix multiplyWise(Matrix other) {
		int m = Math.min(this.rows(), other.rows());
		int n = Math.min(this.columns(), other.columns());
		Matrix result = create(new Size(n, m));
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
		return new MatrixImpl(NeuronValue.divide(this.data, value));
	}


	@Override
	public Matrix divide0(double value) {
		return new MatrixImpl(NeuronValue.divide(this.data, value));
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
		Matrix result = create(new Size(this.columns(), this.rows()));
		for (int i = 0; i < this.rows(); i++) {
			for (int j = 0; j < this.columns(); j++) {
				result.set(i, j, this.get(i, j).evaluate(f));
			}
		}
		return result;
	}


	@Override
	public Matrix derivativeWise(Function f) {
		Matrix result = create(new Size(this.columns(), this.rows()));
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
		
		Matrix newMatrix = new MatrixImpl(new Size(n, m));
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
		
		Matrix newMatrix = new MatrixImpl(new Size(n, m));
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
	public Matrix vec() {
		if (this.columns() == 1) return this;
		Matrix result = create(new Size(1, this.rows()*this.columns()));
		int k = 0;
		for (int j = 0; j < this.columns(); j++) {
			for (int i = 0; i < this.rows(); i++) {
				result.set(k, 0, this.get(i, j));
				k++;
			}
		}
		return result;
	}


	@Override
	public Matrix vecInverse(int rows) {
		if (rows <= 0) return null;
		int columns = this.rows() / rows;
		if (columns == 0) return null;
		
		Matrix result = create(new Size(columns, rows));
		for (int j = 0; j < columns; j++) {
			int columnLength = j*rows;
			for (int i = 0; i < rows; i++) {
				int index = columnLength + i;
				result.set(i, j, this.get(index, 0));
			}
		}
		return result;
	}
	

	@Override
	public Matrix create(Size size) {
		if (size.height <= 0 || size.width <= 0)
			return null;
		else
			return new MatrixImpl(size, this.get(0, 0).zero());
	}


	/**
	 * Creating matrix from data array.
	 * @param data data array.
	 * @return matrix.
	 */
	public static Matrix create(NeuronValue[][] data) {
		if (data == null || data.length == 0) return null;
		int n = data[0].length;
		if (n == 0) return null;
		for (int i = 1; i < data.length; i++) {
			if (data[i] == null || data[i].length != n) return null;
		}
		return new MatrixImpl(data);
	}


	@Override
	public Matrix createIdentity(int n) {
		Matrix matrix = create(new Size(n, n));
		if (matrix == null) return null;
		NeuronValue zero = matrix.get(0, 0).zero();
		NeuronValue unit = matrix.get(0, 0).unit();
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (i == j)
					matrix.set(i, j, unit);
				else
					matrix.set(i, j, zero);
			}
		}
		return matrix;
	}


}
