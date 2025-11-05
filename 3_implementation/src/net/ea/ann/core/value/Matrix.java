/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value;

import java.util.List;
import java.util.Random;

import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.raster.Image;
import net.ea.ann.raster.ImageList;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Raster2DImpl;
import net.ea.ann.raster.Raster3DImpl;
import net.ea.ann.raster.Size;

/**
 * This interface represents matrix.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Matrix extends NeuronValueCreator {


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
	 * Getting column as column vector.
	 * @param column column index.
	 * @return column as column vector.
	 */
	Matrix getColumn(int column);
	
	
	/**
	 * Extracting a sub-matrix at specified column index.
	 * @param column specified column index.
	 * @param range specified range.
	 * @return sub-matrix extracted at specified column index.
	 */
	Matrix getColumns(int column, int range);


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
	 * Wise multiplying this matrix with other matrix.
	 * @param other other matrix.
	 * @return multiplied matrix.
	 */
	Matrix multiplyWise(Matrix other);

	
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
	 * Calculating Kronecker product of this matrix and the other matrix.
	 * @param other other matrix.
	 * @param rowOfThis the row of this matrix.
	 * @return Kronecker product of this matrix and the other matrix at specified row of this matrix.
	 */
	Matrix kroneckerProductRowOf(Matrix other, int rowOfThis);
	
	
	/**
	 * Calculating Kronecker product of the first matrix and the second matrix.
	 * @param first first matrix.
	 * @param second second matrix.
	 * @param rowOfFirst the row of first matrix.
	 * @param rowOfSecond the row of second matrix.
	 * @return Kronecker product of this matrix and the other matrix at specified row of first matrix and row of second matrix.
	 */
	static Matrix kroneckerProductRowOf(Matrix first, Matrix second, int rowOfFirst, int rowOfSecond) {
		return first.kroneckerProductRowOf(second.getRow(rowOfSecond), rowOfFirst);
	}
	
	
	/**
	 * Calculating multiplication of Kronecker product and another matrix.
	 * @param first first matrix.
	 * @param second second matrix.
	 * @param rowOfFirst the row of first matrix.
	 * @param rowOfSecond the row of second matrix.
	 * @param multiplier third matrix.
	 * @return multiplication of Kronecker product and another matrix.
	 */
	static Matrix kroneckerProductMutilply(Matrix first, Matrix second, int rowOfFirst, int rowOfSecond, Matrix multiplier) {
		return first.kroneckerProductRowOf(second.getRow(rowOfSecond), rowOfFirst).multiply(multiplier);
	}
	
	
	/**
	 * Calculating multiplication of Kronecker product and another matrix.
	 * @param first first matrix.
	 * @param second second matrix.
	 * @param rowOfFirst the row of first matrix.
	 * @param multiplier third matrix.
	 * @return multiplication of Kronecker product and another matrix.
	 */
	static Matrix kroneckerProductMutilply(Matrix first, Matrix second, int rowOfFirst, Matrix multiplier) {
		int rows = second.rows();
		Matrix[] results = new Matrix[rows];
		for (int row = 0; row < rows; row++) {
			results[row] = kroneckerProductMutilply(first, second, rowOfFirst, row, multiplier);
		}
		return concatH(results);
	}
	
	
	/**
	 * Calculating multiplication of Kronecker product and another matrix.
	 * @param first first matrix.
	 * @param second second matrix.
	 * @param multiplier third matrix.
	 * @return multiplication of Kronecker product and another matrix.
	 */
	static Matrix kroneckerProductMutilply(Matrix first, Matrix second, Matrix multiplier) {
		int rows = first.rows();
		Matrix[] results = new Matrix[rows];
		for (int row = 0; row < rows; row++) {
			results[row] = kroneckerProductMutilply(first, second, row, multiplier);
		}
		return concatH(results);
	}
	
	
	/**
	 * Evaluating matrix by function.
	 * @param f specific function.
	 * @return evaluated matrix.
	 */
	Matrix evaluate0(Function f);
	
	
	/**
	 * Taking derivative on every element of specified matrix.
	 * @param f function.
	 * @return the matrix whose elements are derivatives.
	 */
	Matrix derivativeWise(Function f);
	
	
	/**
	 * Calculating soft-max function of vector.
	 * @param values vector.
	 * @return soft-max function of vector.
	 */
	static NeuronValue[] softmax(NeuronValue...values) {
		if (values == null || values.length == 0) return null;
		NeuronValue[] softmax = new NeuronValue[values.length];
		NeuronValue sum = values[0].zero();
		NeuronValue max = NeuronValue.max(values);
		for (int i = 0; i < values.length; i++) {
			softmax[i] = values[i].subtract(max).exp();
			sum = sum.add(softmax[i]);
		}
		for (int i = 0; i < values.length; i++) {
			softmax[i] = softmax[i].divide(sum);
		}
		return softmax;
	}
	
	
	/**
	 * Calculating soft-max function of matrix by row.
	 * @param matrix matrix.
	 * @return soft-max function of matrix by row.
	 */
	static Matrix softmaxByRow(Matrix matrix) {
		if (matrix == null) return null;
		Matrix softmax = matrix.create(matrix.rows(), matrix.columns());
		NeuronValue max = valueMax(matrix);
		NeuronValue zero = max.zero();
		for (int row = 0; row < matrix.rows(); row++) {
			NeuronValue[] exps = new NeuronValue[matrix.columns()];
			NeuronValue sum = zero;
			for (int column = 0; column < matrix.columns(); column++) {
				exps[column] = matrix.get(row, column).subtract(max).exp();
				sum = sum.add(exps[column]);
			}
			
			for (int column = 0; column < matrix.columns(); column++) {
				NeuronValue value = exps[column].divide(sum);
				softmax.set(row, column, value);
			}
		}
		
		return softmax;
	}

	
	/**
	 * Calculating inverse soft-max function of matrix by row.
	 * @param matrix matrix.
	 * @return inverse soft-max function of matrix by row.
	 */
	static Matrix softmaxByRowInverse(Matrix matrix) {
		Matrix unitMatrix = matrix.create(matrix.rows(), matrix.columns());
		NeuronValue unit = matrix.get(0, 0).unit();
		Matrix.fill(unitMatrix, unit);
		return softmaxByRow(unitMatrix.subtract(matrix));
	}
	
	
	/**
	 * Calculating soft-max function of matrix by column.
	 * @param matrix matrix.
	 * @return soft-max function of matrix by column.
	 */
	static Matrix softmaxByColumn(Matrix matrix) {
		if (matrix == null) return null;
		Matrix softmax = matrix.create(matrix.rows(), matrix.columns());
		NeuronValue max = valueMax(matrix);
		NeuronValue zero = max.zero();
		for (int column = 0; column < matrix.columns(); column++) {
			NeuronValue[] exps = new NeuronValue[matrix.rows()];
			NeuronValue sum = zero;
			for (int row = 0; row < matrix.rows(); row++) {
				exps[row] = matrix.get(row, column).subtract(max).exp();
				sum = sum.add(exps[row]);
			}
			
			for (int row = 0; row < matrix.rows(); row++) {
				NeuronValue value = exps[row].divide(sum);
				softmax.set(row, column, value);
			}
		}
		
		return softmax;
	}
	
	
	/**
	 * Calculating inverse soft-max function of matrix by column.
	 * @param matrix matrix.
	 * @return inverse soft-max function of matrix by column.
	 */
	static Matrix softmaxByColumnInverse(Matrix matrix) {
		Matrix unitMatrix = matrix.create(matrix.rows(), matrix.columns());
		NeuronValue unit = matrix.get(0, 0).unit();
		Matrix.fill(unitMatrix, unit);
		return softmaxByColumn(unitMatrix.subtract(matrix));
	}

	
	/**
	 * Concatenating many matrices into one matrix by horizontal, excluding this matrix.
	 * @param matrices array of matrices.
	 * @return concatenated matrix.
	 */
	Matrix concatHorizontal(Matrix...matrices);

	
	/**
	 * Concatenating many matrices into one matrix by horizontal.
	 * @param matrices array of matrices.
	 * @return concatenated matrix.
	 */
	static Matrix concatH(Matrix...matrices) {
		if (matrices == null || matrices.length == 0)
			return null;
		else
			return matrices[0].concatHorizontal(matrices);
	}
	
	
	/**
	 * Concatenating many matrices into one matrix by vertical, excluding this matrix.
	 * @param matrices array of matrices.
	 * @return concatenated matrix.
	 */
	Matrix concatVertical(Matrix...matrices);
	
	
	/**
	 * Concatenating many matrices into one matrix by vertical.
	 * @param matrices array of matrices.
	 * @return concatenated matrix.
	 */
	static Matrix concatV(Matrix...matrices) {
		if (matrices == null || matrices.length == 0)
			return null;
		else
			return matrices[0].concatVertical(matrices);
	}

	
	/**
	 * Vectorization of matrix.
	 * @return vectorized vector.
	 */
	Matrix vec();
	
	
	/**
	 * Converting vectorized vector back to matrix.
	 * @param rows rows.
	 * @return matrix.
	 */
	Matrix vecInverse(int rows);
	

	/**
	 * Checking whether two matrices are equal.
	 * @param matrix1 matrix 1.
	 * @param matrix2 matrix 2.
	 * @return true if two matrices are equal.
	 */
	static boolean equals(Matrix matrix1, Matrix matrix2) {
		if (matrix1.rows() != matrix2.rows() || matrix1.columns() != matrix2.columns()) return false;
		int rows = matrix1.rows(), columns = matrix1.columns();
		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < columns; column++) {
				if (!matrix1.get(row, column).equals(matrix2.get(row, column))) return false;
			}
		}
		return true;
	}
	
	
	/**
	 * Checking whether two matrices are equal in elements reference.
	 * @param matrix1 matrix 1.
	 * @param matrix2 matrix 2.
	 * @return true if two matrices are equal in elemental reference.
	 */
	static boolean refEquals(Matrix matrix1, Matrix matrix2) {
		if (matrix1.rows() != matrix2.rows() || matrix1.columns() != matrix2.columns()) return false;
		int rows = matrix1.rows(), columns = matrix1.columns();
		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < columns; column++) {
				if (matrix1.get(row, column) != matrix2.get(row, column)) return false;
			}
		}
		return true;
	}

	
	/**
	 * Calculating value mean of matrices.
	 * @param matrices specified matrices.
	 * @return value mean.
	 */
	static NeuronValue valueSum(Matrix...matrices) {
		if (matrices == null || matrices.length == 0) return null;
		NeuronValue sum = null;
		for (Matrix matrix : matrices) {
			for (int i = 0; i < matrix.rows(); i++) {
				for (int j = 0; j < matrix.columns(); j++) {
					NeuronValue value = matrix.get(i, j);
					if (sum == null)
						sum = value;
					else
						sum = sum.add(value);
				}
			}
		}
		return sum;
	}

	
	/**
	 * Calculating value mean of matrices.
	 * @param matrices specified matrices.
	 * @return value mean.
	 */
	static NeuronValue valueMean(Matrix...matrices) {
		if (matrices == null || matrices.length == 0) return null;
		NeuronValue mean = null;
		int N = 0;
		for (Matrix matrix : matrices) {
			for (int i = 0; i < matrix.rows(); i++) {
				for (int j = 0; j < matrix.columns(); j++) {
					NeuronValue value = matrix.get(i, j);
					if (mean == null)
						mean = value;
					else
						mean = mean.add(value);
					N++;
				}
			}
		}
		return mean.divide((double)N);
	}

	
	/**
	 * Calculating norm sum of matrices.
	 * @param matrices specified matrices.
	 * @return norm sum.
	 */
	static double normSum(Matrix...matrices) {
		if (matrices == null || matrices.length == 0) return 0;
		double sum = 0;
		for (Matrix matrix : matrices) {
			for (int i = 0; i < matrix.rows(); i++) {
				for (int j = 0; j < matrix.columns(); j++) {
					sum += matrix.get(i, j).norm();
				}
			}
		}
		return sum;
	}

	
	/**
	 * Calculating norm mean of matrices.
	 * @param matrices specified matrices.
	 * @return norm mean.
	 */
	static double normMean(Matrix...matrices) {
		if (matrices == null || matrices.length == 0) return 0;
		double mean = 0;
		int N = 0;
		for (Matrix matrix : matrices) {
			for (int i = 0; i < matrix.rows(); i++) {
				for (int j = 0; j < matrix.columns(); j++) {
					mean += matrix.get(i, j).norm();
					N++;
				}
			}
		}
		return mean / (double)N;
	}

	
	/**
	 * Getting max value of matrix.
	 * @param matrix matrix.
	 * @return max value of matrix.
	 */
	static NeuronValue valueMax(Matrix matrix) {
		NeuronValue max = matrix.get(0, 0);
		for (int row = 0; row < matrix.rows(); row++) {
			for (int column = 0; column < matrix.columns(); column++)
				max = max.max(matrix.get(row, column));
		}
		return max;
	}
	
	
	/**
	 * Determining minimum matrix.
	 * @param matrices array of matrices.
	 * @return minimum matrix.
	 */
	static Matrix min(Matrix...matrices) {
		if (matrices == null || matrices.length == 0) return null;
		Matrix min = matrices[0];
		for (int i = 1; i < matrices.length; i++) {
			for (int row = 0; row < min.rows(); row++) {
				for (int column = 0; column < min.columns(); column++) {
					NeuronValue v1 = min.get(row, column);
					NeuronValue v2 = matrices[i].get(row, column);
					min.set(row, column, v1.min(v2));
				}
			}
		}
		return min;
	}
	
	
	/**
	 * Determining minimum matrix.
	 * @param matrices array of matrices.
	 * @return minimum matrix.
	 */
	static Matrix max(Matrix...matrices) {
		if (matrices == null || matrices.length == 0) return null;
		Matrix max = matrices[0];
		for (int i = 1; i < matrices.length; i++) {
			for (int row = 0; row < max.rows(); row++) {
				for (int column = 0; column < max.columns(); column++) {
					NeuronValue v1 = max.get(row, column);
					NeuronValue v2 = matrices[i].get(row, column);
					max.set(row, column, v1.max(v2));
				}
			}
		}
		return max;
	}

	
	/**
	 * Calculating sum matrix.
	 * @param matrices array of matrices.
	 * @return sum matrix.
	 */
	static Matrix sum(Matrix...matrices) {
		if (matrices == null || matrices.length == 0) return null;
		Matrix sum = matrices[0];
		for (int i = 1; i < matrices.length; i++) sum = sum.add(matrices[i]);
		return sum;
	}

	
	/**
	 * Calculating mean matrix.
	 * @param matrices array of matrices.
	 * @return mean matrix.
	 */
	static Matrix mean(Matrix...matrices) {
		if (matrices == null || matrices.length == 0) return null;
		Matrix mean = matrices[0];
		for (int i = 1; i < matrices.length; i++) mean = mean.add(matrices[i]);
		return mean.divide0(matrices.length);
	}
	
	
	/**
	 * Calculating mean absolute value of two matrices.
	 * @param matrix1 matrix 1.
	 * @param matrix2 matrix 2.
	 * @return mean absolute value of two matrices.
	 */
	static NeuronValue mae(Matrix matrix1, Matrix matrix2) {
		int rows = Math.min(matrix1.rows(), matrix2.rows());
		int columns = Math.min(matrix1.columns(), matrix2.columns());
		NeuronValue mae = null;
		int n = 0;
		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < columns; column++) {
				NeuronValue d = matrix1.get(row, column).subtract(matrix2.get(row, column)).abs();
				mae = mae != null ? mae.add(d) : d;
				n++;
			}
		}
		return mae.divide(n);
	}
	
	
	/**
	 * Calculating standard deviation matrix.
	 * @param matrices array of matrices.
	 * @return standard deviation matrix.
	 */
	static Matrix std(Matrix...matrices) {
		if (matrices == null || matrices.length == 0) return null;
		Matrix mean = mean(matrices);
		Matrix[] stds = new Matrix[matrices.length];
		for (int i = 0; i < matrices.length; i++) {
			stds[i] = mean.create(mean.rows(), mean.columns());
			for (int row = 0; row < mean.rows(); row++) {
				for (int column = 0; column < mean.columns(); column++) {
					NeuronValue d = matrices[i].get(row, column).subtract(mean.get(row, column));
					stds[i].set(row, column, d.multiply(d));
				}
			}
		}
		
		Matrix std = mean(stds);
		for (int row = 0; row < std.rows(); row++) {
			for (int column = 0; column < std.columns(); column++) {
				std.set(row, column, std.get(row, column).sqrt());
			}
		}
		return std;
	}
	
	
	/**
	 * Creating matrix.
	 * @param rows rows.
	 * @param columns columns.
	 * @return created matrix.
	 */
	Matrix create(int rows, int columns);
	
	
	/**
	 * Creating new matrix.
	 * @param rows rows.
	 * @param columns columns.
	 * @param value specified value.
	 * @return matrix.
	 */
	static Matrix create(int rows, int columns, Object value) {
		if (rows <= 0 || columns <= 0)
			return null;
		else if (value == null)
			return new MatrixImpl(rows, columns, new NeuronValue1(0).zero());
		else if (value instanceof NeuronValue)
			return new MatrixImpl(rows, columns, (NeuronValue)value);
		else if (value instanceof Number)
			return new NeuronValueM(rows, columns, ((Number)value).doubleValue());
		else
			return null;
	}


	/**
	 * Creating identity matrix.
	 * @param n rows and columns.
	 * @return identity matrix.
	 */
	Matrix createIdentity(int n);
	
	
	/**
	 * Copying source matrix to target matrix.
	 * @param source source matrix.
	 * @param target target matrix.
	 */
	static void copy(Matrix source, Matrix target) {
		if (source == null || target == null) return;
		int rows = Math.min(source.rows(), target.rows());
		int columns = Math.min(source.columns(), target.columns());
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) target.set(i, j, source.get(i, j));
		}
	}

	
	/**
	 * Copying source array to target matrix.
	 * @param source source array.
	 * @param target target matrix.
	 */
	static void copy(NeuronValue[] source, Matrix target) {
		if (source == null || target == null) return;
		int rows = target.rows();
		int columns = target.columns();
		for (int i = 0; i < rows; i++) {
			int rowLength = i*columns;
			for (int j = 0; j < columns; j++) {
				int index = rowLength + j;
				if (index < source.length) target.set(i, j, source[index]);
			}
		}
	}
	
	
	/**
	 * Filling matrix by specified real value.
	 * @param matrix matrix.
	 * @param value value.
	 * @return filled matrix.
	 */
	static void fill(Matrix matrix, NeuronValue value) {
		for (int i = 0; i < matrix.rows(); i++) {
			for (int j = 0; j < matrix.columns(); j++) matrix.set(i, j, value);
		}
	}
	
	
	/**
	 * Filling matrix by specified real value.
	 * @param matrix matrix.
	 * @param v value.
	 * @return filled matrix.
	 */
	static void fill(Matrix matrix, double v) {
		NeuronValue value = matrix.get(0, 0).valueOf(v);
		fill(matrix, value);
	}
	
	
	/**
	 * Filling matrix by random value.
	 * @param matrix matrix.
	 * @param rnd randomizer.
	 */
	static void fill(Matrix matrix, Random rnd) {
		for (int row = 0; row < matrix.rows(); row++) {
			for (int column = 0; column < matrix.columns(); column++) {
				NeuronValue value = matrix.get(row, column).valueOf(NeuronValue.r(rnd));
				matrix.set(row, column, value);
			}
		}
	}

	
	/**
	 * Extracting raster into matrix.
	 * @param rows rows.
	 * @param columns columns.
	 * @param raster raster.
	 * @param neuronChannel neuron channel.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @param ref reference to matrix, which can be null.
	 * @return matrix.
	 */
	static Matrix toMatrix(int rows, int columns, Raster raster, int neuronChannel, boolean isNorm, Matrix ref) {
		NeuronValue[] values = raster.toNeuronValues(neuronChannel, new Size(columns, rows, 1, 1), isNorm);
		Matrix matrix = (ref == null) ? create(rows, columns, values[0]) : ref.create(rows, columns);
		for (int i = 0; i < matrix.rows(); i++) {
			int rowLength = i*matrix.columns();
			for (int j = 0; j < matrix.columns(); j++) {
				int index = rowLength + j;
				matrix.set(i, j, values[index]);
			}
		}
		return matrix;
	}

	
	/**
	 * Extracting raster into matrix.
	 * @param rows rows.
	 * @param columns columns.
	 * @param raster raster.
	 * @param neuronChannel neuron channel.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @return matrix.
	 */
	static Matrix toMatrix(int rows, int columns, Raster raster, int neuronChannel, boolean isNorm) {
		return toMatrix(rows, columns, raster, neuronChannel, isNorm, null);
	}
	
	
	/**
	 * Extracting values of matrix as vector.
	 * @param matrix specified matrix.
	 * @return values of matrix as vector.
	 */
	static NeuronValue[] extractValues(Matrix matrix) {
		NeuronValue[] values = new NeuronValue[matrix.rows()*matrix.columns()];
		for (int i = 0; i < matrix.rows(); i++) {
			int rowLength = i*matrix.columns();
			for (int j = 0; j < matrix.columns(); j++) {
				int index = rowLength + j;
				values[index] = matrix.get(i, j);
			}
		}
		return values;
	}
	
	
	/**
	 * Create raster from neuron values.
	 * @param matrix matrix.
	 * @param neuronChannel neuron channel.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @param defaultAlpha default alpha channel.
	 * @return raster.
	 */
	static Raster toRaster(Matrix matrix, int neuronChannel, boolean isNorm, int defaultAlpha) {
		NeuronValue[] values = extractValues(matrix);
		return Raster2DImpl.create(values, neuronChannel, new Size(matrix.columns(), matrix.rows(), 1, 1), isNorm, defaultAlpha);
	}

	
	/**
	 * Create raster from neuron values.
	 * @param matrices matrix array.
	 * @param neuronChannel neuron channel.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @param defaultAlpha default alpha channel.
	 * @return raster.
	 */
	static Raster toRaster(Matrix matrices[], int neuronChannel, boolean isNorm, int defaultAlpha) {
		if (matrices == null || matrices.length == 0) return null;
		if (matrices.length == 1) return toRaster(matrices[0], neuronChannel, isNorm, defaultAlpha);
		
		List<Image> images = Util.newList(matrices.length);
		for (Matrix matrix : matrices) {
			NeuronValue[] values = extractValues(matrix);
			Raster2DImpl raster = Raster2DImpl.create(values, neuronChannel, new Size(matrix.columns(), matrix.rows(), 1, 1), isNorm, defaultAlpha);
			if (raster != null) images.add(raster.getImage());
		}
		if (images.size() == 0) return null;
		
		ImageList imageList = ImageList.create(images);
		return Raster3DImpl.create(imageList);
	}
	
	
}
