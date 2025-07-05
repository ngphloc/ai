/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value;

import java.io.Serializable;

/**
 * This interface represents matrix.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Matrix extends Cloneable, Serializable {


	/**
	 * Getting the number of rows.
	 * @return the number of rows.
	 */
	int rows();
	
	
	/**
	 * Getting the number of columns.
	 * @return the number of columns.
	 */
	int columns();
	
	
	/**
	 * Getting value at specified row and column.
	 * @param row specified row.
	 * @param column specified column.
	 * @return value at specified row and column.
	 */
	NeuronValue get(int row, int column);
	
	
	/**
	 * Setting value at specified row and column.
	 * @param row specified row.
	 * @param column specified column.
	 * @param value specified value.
	 */
	void set(int row, int column, NeuronValue value);

	
	/**
	 * Getting row as row vector.
	 * @param row row index.
	 * @return row as row vector.
	 */
	Matrix getRow(int row);
	
	
	/**
	 * Transposing this matrix.
	 * @return transposed matrix.
	 */
	Matrix transpose();
	
	
	/**
	 * Negating this matrix.
	 * @return negative matrix.
	 */
	Matrix negative0();

	
	/**
	 * Adding this matrix with other matrix.
	 * @param other other matrix.
	 * @return added matrix.
	 */
	Matrix add(Matrix other);

	
	/**
	 * Subtracting this matrix with other matrix.
	 * @param other other matrix.
	 * @return subtracted matrix.
	 */
	Matrix subtract(Matrix other);

	
	/**
	 * Multiplying this matrix with other matrix.
	 * @param other other matrix.
	 * @return multiplied matrix.
	 */
	Matrix multiply(Matrix other);
	
	
	/**
	 * Multiplying this matrix with other value.
	 * @param value other value.
	 * @return multiplied matrix.
	 */
	Matrix multiply0(NeuronValue value);


	/**
	 * Multiplying this matrix with other value.
	 * @param value other value.
	 * @return multiplied matrix.
	 */
	Matrix multiply0(double value);


	/**
	 * Dividing this matrix with other value.
	 * @param value other value.
	 * @return multiplied matrix.
	 */
	Matrix divide0(NeuronValue value);


	/**
	 * Dividing this matrix with other value.
	 * @param value other value.
	 * @return multiplied matrix.
	 */
	Matrix divide0(double value);


	/**
	 * Concatenating many matrices into one matrix by horizontal, excluding this matrix.
	 * @param matrices array of matrices.
	 * @return concatenated matrix.
	 */
	Matrix concatHorizontal(Matrix...matrices);

	
	/**
	 * Concatenating many matrices into one matrix by vertical, excluding this matrix.
	 * @param matrices array of matrices.
	 * @return concatenated matrix.
	 */
	Matrix concatVertical(Matrix...matrices);
	
	
	/**
	 * Extracting a sub-matrix at specified column index.
	 * @param columnIndex specified column index.
	 * @param range specified range.
	 * @return sub-matrix extracted at specified column index.
	 */
	Matrix extractVertical(int columnIndex, int range);


}
