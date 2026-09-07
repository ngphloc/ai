/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value;

import java.io.Serializable;

import net.ea.ann.mane.Kernel;
import net.ea.ann.raster.Size;

/**
 * This class provides utility methods to manipulate matrix.
 * This class is the replacement of the class {@link MatrixUtil} for recursion of matrix stack.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MatrixUtilExt implements Cloneable, Serializable {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public MatrixUtilExt() {}
	


	
	/**
	 * Getting depth of matrix.
	 * @param matrix matrix.
	 * @return depth of matrix.
	 */
	public static int depth(Matrix matrix) {
		return matrix instanceof MatrixStack ? ((MatrixStack)matrix).depth() : 1;
	}
	
	
	
	/**
	 * Getting capacity.
	 * @return capacity.
	 */
	public static int capacity(Matrix matrix) {
		int depth = depth(matrix);
		return matrix.rows()*matrix.columns()*depth;
	}
	
	
	/**
	 * Creating new matrix.
	 * @param size size.
	 * @return matrix.
	 */
	private static Matrix create0(Size size, Object value) {
		if (size.height <= 0 || size.width <= 0)
			return null;
		else if (value == null)
			return Kernel.SPEED_MODE ? new MatrixReal(size, 0) : new MatrixImpl(size, new NeuronValue1(0).zero());
		else if (value instanceof NeuronValue)
			return Kernel.speedMode((NeuronValue)value) ? new MatrixReal(size, ((NeuronValue1)value).get()) : new MatrixImpl(size, (NeuronValue)value);
		else if (value instanceof Number)
			return new MatrixReal(size, ((Number)value).doubleValue());
		else
			return null;
	}


	/**
	 * Creating new matrix.
	 * @param size size.
	 * @param value specified value.
	 * @return matrix.
	 */
	public static Matrix create(Size size, Object value) {
		int depth = size.depth < 1 ? 1 : size.depth;
		if (depth == 1) return create0(size, value);
		Matrix[] matrices = new Matrix[depth];
		for (int i = 0; i < matrices.length; i++) matrices[i] = create0(size, value);
		return new MatrixStack(matrices);
	}

	
	/**
	 * Splitting matrix.
	 * @param matrix matrix.
	 * @return array of matrices.
	 */
	public static Matrix[] split(Matrix matrix) {
		if (matrix == null)
			return null;
		else
			return matrix instanceof MatrixStack ? ((MatrixStack)matrix).matrices() : new Matrix[] {matrix};
	}
	
	
	/**
	 * Joining matrices.
	 * @param matrices matrices.
	 * @return joined matrix.
	 */
	public static Matrix join(Matrix...matrices) {
		if (matrices == null || matrices.length == 0)
			return null;
		else
			return matrices.length > 1 ? new MatrixStack(matrices) : matrices[0];
	}
	
	
	/**
	 * Calculating norm of stacks.
	 * @param matrices stacks.
	 * @return norm of stacks.
	 */
	public static double norm(MatrixStack...stacks) {
		assert (stacks != null && stacks.length > 0);
		
		double norm = 0;
		for (MatrixStack stack : stacks) {
			Matrix[] matrices = stack.matrices();
			int rows = matrices[0].rows(), columns = matrices[0].columns();
			for (int d = 0; d < matrices.length; d++) {
				for (int row = 0; row < rows; row++) {
					for (int column = 0; column < columns; column++) {
						double v = matrices[d].getv(row, column);
						norm += v*v;
					}
				}
			}
		}
		return Math.sqrt(norm);
	}

	
	/**
	 * Calculating norm of stacks.
	 * @param matrices stacks.
	 * @return norm of stacks.
	 */
	public static NeuronValue normV(MatrixStack...stacks) {
		assert (stacks != null && stacks.length > 0);
		
		NeuronValue norm = stacks[0].get().get(0, 0).zero();
		for (MatrixStack stack : stacks) {
			Matrix[] matrices = stack.matrices();
			int rows = matrices[0].rows(), columns = matrices[0].columns();
			for (int d = 0; d < matrices.length; d++) {
				for (int row = 0; row < rows; row++) {
					for (int column = 0; column < columns; column++) {
						NeuronValue v = matrices[d].get(row, column);
						norm = norm.add(v.multiply(v));
					}
				}
			}
		}
		return norm.sqrt();
	}

	
	/**
	 * Calculating norm of matrices.
	 * @param matrices matrices.
	 * @return norm of matrices.
	 */
	public static double norm(Matrix...matrices) {
		assert (matrices != null && matrices.length > 0);
		
		int rows = matrices[0].rows(), columns = matrices[0].columns();
		double norm = 0;
		for (int d = 0; d < matrices.length; d++) {
			for (int row = 0; row < rows; row++) {
				for (int column = 0; column < columns; column++) {
					double v = matrices[d].getv(row, column);
					norm += v*v;
				}
			}
		}
		return Math.sqrt(norm);
	}
	
	
	/**
	 * Calculating norm of matrices.
	 * @param matrices matrices.
	 * @return norm of matrices.
	 */
	public static NeuronValue normV(Matrix...matrices) {
		assert (matrices != null && matrices.length > 0);
		
		int rows = matrices[0].rows(), columns = matrices[0].columns();
		NeuronValue norm = matrices[0].get(0, 0).zero();
		for (int d = 0; d < matrices.length; d++) {
			for (int row = 0; row < rows; row++) {
				for (int column = 0; column < columns; column++) {
					NeuronValue v = matrices[d].get(row, column);
					norm = norm.add(v.multiply(v));
				}
			}
		}
		return norm.sqrt();
	}

	
	/**
	 * Calculating norm of values.
	 * @param values values.
	 * @return norm of values.
	 */
	public static double norm(NeuronValue...values) {
		assert (values != null && values.length > 0);
		
		double norm = 0;
		for (int d = 0; d < values.length; d++) {
			double v = ((NeuronValue1)values[d]).get();
			norm += v*v;
		}
		return Math.sqrt(norm);
	}

	
	/**
	 * Calculating norm of matrices.
	 * @param matrices matrices.
	 * @return norm of matrices.
	 */
	public static NeuronValue normV(NeuronValue...values) {
		assert (values != null && values.length > 0);
		
		NeuronValue norm = values[0].zero();
		for (int d = 0; d < values.length; d++) {
			NeuronValue v = values[d];
			norm = norm.add(v.multiply(v));
		}
		return norm.sqrt();
	}

	
}
