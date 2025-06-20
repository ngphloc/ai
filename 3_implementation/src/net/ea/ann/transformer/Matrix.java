/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.transformer;

import java.io.Serializable;

import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValue1;
import net.ea.ann.core.value.NeuronValueV;

/**
 * This class represents standard attention.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Matrix implements Cloneable, Serializable {

	
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
	private Matrix(NeuronValue[][] data) {
		this.data = data;
	}
	
	
	/**
	 * Constructor with numbers of rows and columns along with specified value.
	 * @param rows number of rows.
	 * @param columns numbers of columns.
	 * @param value specified value.
	 */
	public Matrix(int rows, int columns, NeuronValue value) {
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
	public Matrix(int rows, int columns, double value) {
		this(rows, columns, new NeuronValue1(value));
	}
	
	
	/**
	 * Constructor with numbers of rows and columns along with specified double array.
	 * @param rows number of rows.
	 * @param columns numbers of columns.
	 * @param array specified double array.
	 */
	public Matrix(int rows, int columns, double[] array) {
		this(rows, columns, new NeuronValueV(array));
	}


	/**
	 * Constructor with numbers of rows and columns along.
	 * @param rows number of rows.
	 * @param columns numbers of columns.
	 */
	public Matrix(int rows, int columns) {
		this(rows, columns, 0);
	}


	/**
	 * Getting the number of rows.
	 * @return the number of rows.
	 */
	public int rows() {return data.length;}
	
	
	/**
	 * Getting the number of columns.
	 * @return the number of columns.
	 */
	public int columns() {return data.length > 0 ? data[0].length : 0;}
	
	
	/**
	 * Getting value at specified row and column.
	 * @param row specified row.
	 * @param column specified column.
	 * @return value at specified row and column.
	 */
	public NeuronValue get(int row, int column) {
		return data[row][column];
	}
	
	
	/**
	 * Setting value at specified row and column.
	 * @param row specified row.
	 * @param column specified column.
	 * @param value specified value.
	 * @return old value.
	 */
	public NeuronValue set(int row, int column, NeuronValue value) {
		NeuronValue oldValue = data[row][column];
		data[row][column] = value;
		return oldValue;
	}

	
	/**
	 * Transposing this matrix.
	 * @return transposed matrix.
	 */
	public Matrix transpose() {
		return new Matrix(NeuronValue.transpose(data));
	}
	
	
	/**
	 * Negating this matrix.
	 * @return negative matrix.
	 */
	public Matrix negative() {
		return new Matrix(NeuronValue.negative(data));
	}

	
	/**
	 * Adding this matrix with other matrix.
	 * @param other other matrix.
	 * @return added matrix.
	 */
	public Matrix add(Matrix other) {
		return new Matrix(NeuronValue.add(this.data, other.data));
	}

	
	/**
	 * Subtracting this matrix with other matrix.
	 * @param other other matrix.
	 * @return subtracted matrix.
	 */
	public Matrix subtract(Matrix other) {
		return new Matrix(NeuronValue.subtract(this.data, other.data));
	}

	
	/**
	 * Multiplying this matrix with other matrix.
	 * @param other other matrix.
	 * @return multiplied matrix.
	 */
	public Matrix multiply(Matrix other) {
		return new Matrix(NeuronValue.multiply(this.data, other.data));
	}
	
	
	/**
	 * Multiplying this matrix with other value.
	 * @param value other value.
	 * @return multiplied matrix.
	 */
	public Matrix multiply(NeuronValue value) {
		return new Matrix(NeuronValue.multiply(this.data, value));
	}


	/**
	 * Multiplying this matrix with other value.
	 * @param value other value.
	 * @return multiplied matrix.
	 */
	public Matrix multiply(double value) {
		return new Matrix(NeuronValue.multiply(this.data, value));
	}


	/**
	 * Dividing this matrix with other value.
	 * @param value other value.
	 * @return multiplied matrix.
	 */
	public Matrix divide(NeuronValue value) {
		return new Matrix(NeuronValue.divide(this.data, value));
	}


	/**
	 * Dividing this matrix with other value.
	 * @param value other value.
	 * @return multiplied matrix.
	 */
	public Matrix divide(double value) {
		return new Matrix(NeuronValue.divide(this.data, value));
	}


	/**
	 * Concatenating many matrices into one matrix, excluding this matrix.
	 * @param matrixs array of matrices.
	 * @return concatenated matrix.
	 */
	public Matrix concatenate(Matrix...matrixs) {
		throw new RuntimeException("Matrix.concatenate(Matrix...) not implemented yet.");
	}
	
	
	/**
	 * Extracting a sub-matrix at specified column.
	 * @param column specified column.
	 * @return sub-matrix extracted at specified column.
	 */
	public Matrix extractVertical(int column) {
		throw new RuntimeException("Matrix.extract(int) not implemented yet.");
	}
	
	
}
