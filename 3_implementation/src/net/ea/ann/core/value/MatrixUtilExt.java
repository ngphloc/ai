/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value;

import java.io.Serializable;

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
			return new MatrixImpl(size, new NeuronValue1(0).zero());
		else if (value instanceof NeuronValue)
			return new MatrixImpl(size, (NeuronValue)value);
		else if (value instanceof Number)
			return new NeuronValueM(size, ((Number)value).doubleValue());
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
	
	
}
