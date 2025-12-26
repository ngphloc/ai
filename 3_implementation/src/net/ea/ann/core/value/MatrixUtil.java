package net.ea.ann.core.value;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import net.ea.ann.core.Util;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Raster2DImpl;
import net.ea.ann.raster.Raster3DImpl;
import net.ea.ann.raster.Raster4DImpl;
import net.ea.ann.raster.Size;

/**
 * This class provides utility methods to manipulate matrix.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public final class MatrixUtil implements Cloneable, Serializable {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public MatrixUtil() {

	}

	
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
	 * Calculating value sum.
	 * @param matrix matrix.
	 * @return value sum.
	 */
	public static NeuronValue valueSum1(Matrix matrix) {
		return matrix instanceof MatrixStack ? MatrixStack.valueSum((MatrixStack)matrix): Matrix.valueSum(matrix);
	}
	
	
	/**
	 * Calculating value sum.
	 * @param matrix matrix.
	 * @return value sum.
	 */
	public static NeuronValue valueMean1(Matrix matrix) {
		return matrix instanceof MatrixStack ? MatrixStack.valueMean((MatrixStack)matrix): Matrix.valueMean(matrix);
	}
	
	
	/**
	 * Calculating sum matrix.
	 * @param matrices array of matrices.
	 * @return sum matrix.
	 */
	public static Matrix sum(Matrix...matrices) {
		return Matrix.sum(matrices);
	}

	
	/**
	 * Calculating mean matrix.
	 * @param matrices array of matrices.
	 * @return mean matrix.
	 */
	public static Matrix mean(Matrix...matrices) {
		return Matrix.mean(matrices);
	}

	
	/**
	 * Copying source matrix to target matrix.
	 * @param source source matrix.
	 * @param target target matrix.
	 */
	public static void copy(Matrix source, Matrix target) {
		if (source == null || target == null)
			return;
		else if ((source instanceof MatrixStack) && (target instanceof MatrixStack))
			MatrixStack.copy((MatrixStack)source, (MatrixStack)target);
		else if (!(source instanceof MatrixStack) && (target instanceof MatrixStack)) {
			source = MatrixUtil.flattenByChannel(source, ((MatrixStack)target).get().get(0, 0).dim());
			if (source instanceof MatrixStack)
				MatrixStack.copy((MatrixStack)source, (MatrixStack)target);
			else
				Matrix.copy(source, ((MatrixStack)target).get());
		}
		else if ((source instanceof MatrixStack) && !(target instanceof MatrixStack)) {
			target = MatrixUtil.flattenByChannel(target, ((MatrixStack)source).get().get(0, 0).dim());
			if (target instanceof MatrixStack)
				MatrixStack.copy((MatrixStack)source, (MatrixStack)target);
			else
				Matrix.copy(((MatrixStack)source).get(), target);
		}
		else
			Matrix.copy(source, target);
	}
	
	
	/**
	 * Filling matrix by specified real value.
	 * @param matrix matrix.
	 * @param value value.
	 * @return filled matrix.
	 */
	public static void fill(Matrix matrix, NeuronValue value) {
		if (matrix instanceof MatrixStack)
			MatrixStack.fill((MatrixStack)matrix, value);
		else
			Matrix.fill(matrix, value);
	}
	
	
	/**
	 * Filling matrix by specified real value.
	 * @param matrix matrix.
	 * @param v value.
	 * @return filled matrix.
	 */
	public static void fill(Matrix matrix, double v) {
		if (matrix instanceof MatrixStack)
			MatrixStack.fill((MatrixStack)matrix, v);
		else
			Matrix.fill(matrix, v);
	}
	
	
	/**
	 * Filling matrix by random value.
	 * @param matrix matrix.
	 * @param rnd randomizer.
	 */
	public static void fill(Matrix matrix, Random rnd) {
		if (matrix instanceof MatrixStack)
			MatrixStack.fill((MatrixStack)matrix, rnd);
		else
			Matrix.fill(matrix, rnd);
	}

	
	/**
	 * Flattening array of matrices according to smaller channel.
	 * @param matrices array of matrices.
	 * @param smallerChannel smaller channel.
	 * @return flattened array of matrices.
	 */
	private static Matrix[] flattenByChannel(Matrix[] matrices, int smallerChannel) {
		List<Matrix> result = Util.newList(0);
		for (Matrix matrix : matrices) {
			Matrix[] flatten = matrix instanceof MatrixStack ?
				MatrixStack.flattenByChannel(new MatrixStack[] {(MatrixStack)matrix}, smallerChannel) :
				Matrix.flattenByChannel(new Matrix[] {matrix}, smallerChannel);
			result.addAll(Arrays.asList(flatten));
		}
		return result.toArray(new Matrix[] {});
	}
	
	
	/**
	 * Flattening matrix.
	 * @param matrix matrix.
	 * @param smallerChannel smaller channel.
	 * @return flattened matrix.
	 */
	public static Matrix flattenByChannel(Matrix matrix, int smallerChannel) {
		Matrix[] result = flattenByChannel(new Matrix[] {matrix}, smallerChannel);
		return result.length == 1 ? result[0] : new MatrixStack(result);
	}
	
	
	/**
	 * Checking whether to be flatten.
	 * @param depth depth.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @return whether to be flatten.
	 */
	public static boolean isFlatten(int depth, int neuronChannel, int rasterChannel) {
		return depth <= 1 && neuronChannel < rasterChannel && neuronChannel <= 1;
	}
	
	
	/**
	 * Extracting raster into matrix.
	 * @param size size.
	 * @param rows rows.
	 * @param columns columns.
	 * @param raster raster.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @param ref reference to matrix, which can be null.
	 * @return matrix.
	 */
	public static Matrix toMatrix(Size size, Raster raster, int neuronChannel, int rasterChannel, boolean isNorm, Matrix ref) {
		int depth = raster.getDepth() < 1 ? 1 : raster.getDepth();
		NeuronValue[] values = null;
		boolean flatten = isFlatten(depth, neuronChannel, rasterChannel);
		if (flatten)
			values = raster.toNeuronValues(rasterChannel, size, isNorm);
		else
			values = raster.toNeuronValues(neuronChannel, size, isNorm);

		if (ref != null && depth(ref) != depth) ref = null;
		Matrix stack = (ref == null) ? MatrixUtil.create(size, values[0]) : ref.create(size);
		
		for (int i = 0; i < depth; i++) {
			Matrix matrix = stack instanceof MatrixStack ? ((MatrixStack)stack).get(i) : stack;
			for (int j = 0; j < matrix.rows(); j++) {
				int rowLength = j*matrix.columns();
				for (int k = 0; k < matrix.columns(); k++) {
					int index = rowLength + k;
					matrix.set(j, k, values[index]);
				}
			}
		}
		return flatten ? flattenByChannel(stack, neuronChannel) : stack;
	}

	
	/**
	 * Extracting raster into matrix.
	 * @param size size.
	 * @param raster raster.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @return matrix.
	 */
	public static Matrix toMatrix(Size size, Raster raster, int neuronChannel, int rasterChannel, boolean isNorm) {
		return toMatrix(size, raster, neuronChannel, rasterChannel, isNorm, null);
	}
	
	
	/**
	 * Extracting values of matrix as vector.
	 * @param matrix specified matrix.
	 * @return values of matrix as vector.
	 */
	public static NeuronValue[] extractValues(Matrix matrix) {
		int depth = depth(matrix);
		NeuronValue[] values = new NeuronValue[depth*matrix.rows()*matrix.columns()];
		for (int i = 0; i < depth; i++) {
			Matrix matrix0 = matrix instanceof MatrixStack ? ((MatrixStack)matrix).get(i) : matrix;
			int depthLength = i*matrix0.rows()*matrix0.columns();
			for (int j = 0; j < matrix0.rows(); j++) {
				int rowLength = depthLength + j*matrix0.columns();
				for (int k = 0; k < matrix0.columns(); k++) {
					int index = rowLength + k;
					values[index] = matrix0.get(j, k);
				}
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
	public static Raster toRaster(Matrix matrix, int neuronChannel, boolean isNorm, int defaultAlpha) {
		NeuronValue[] values = extractValues(matrix);
		int depth = depth(matrix);
		return depth <= 1 ? Raster2DImpl.create(values, neuronChannel, new Size(matrix.columns(), matrix.rows(), 1, 1), isNorm, defaultAlpha) :
			Raster3DImpl.create(values, neuronChannel, new Size(matrix.columns(), matrix.rows(), depth, 1), isNorm, defaultAlpha);
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
		
		List<Raster> rasters = Util.newList(matrices.length);
		for (Matrix matrix : matrices) {
			Raster raster = toRaster(matrix, neuronChannel, isNorm, defaultAlpha);
			if (raster != null) rasters.add(raster);
		}
		if (rasters.size() == 0) return null;
		
		return rasters.get(0).getDepth() <= 1 ? Raster3DImpl.createByRasters(rasters) : Raster4DImpl.createByRasters(rasters); 
	}
	

}
