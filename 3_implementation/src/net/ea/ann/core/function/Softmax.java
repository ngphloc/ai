/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.function;

import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueV;
import net.ea.ann.raster.Size;

/**
 * This interface represents soft-max function.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Softmax extends Probability, FunctionDelay {

	
	/**
	 * Calculating softmax of specified value and array. 
	 * @param all array.
	 * @param x value.
	 * @return soft-max of specified value and array.
	 */
	static NeuronValue softmax(NeuronValue[] all, NeuronValue x) {
		if (x == null || all == null || all.length == 0) return null;
		
//		NeuronValue max = NeuronValue.max(all);
//		max = max != null ? max : all[0].zero();
		NeuronValue[] exps = new NeuronValue[all.length];
		NeuronValue sum = x.zero();
		for (int i = 0; i < all.length; i++) {
//			exps[i] = all[i].subtract(max).exp();
			exps[i] = all[i].exp();
			sum = sum.add(exps[i]);
		}
//		return x.subtract(max).exp().divide(sum);
		return x.exp().divide(sum);
	}

	
	/**
	 * Calculating soft-max of specified array.
	 * @param all array.
	 * @return soft-max of specified array.
	 */
	static NeuronValue[] softmaxArray(NeuronValue[] all) {
		if (all == null || all.length == 0) return null;
		
		NeuronValue[] softmax = new NeuronValue[all.length];
		NeuronValue sum = all[0].zero();
		for (int i = 0; i < all.length; i++) {
			softmax[i] = all[i].exp();
			sum = sum.add(softmax[i]);
		}
		for (int i = 0; i < all.length; i++) softmax[i] = softmax[i].divide(sum);
		return softmax;
	}
	
	
	/**
	 * Calculating soft-max function of vector.
	 * @param values vector.
	 * @return soft-max function of vector.
	 */
	static NeuronValue[] softmax(NeuronValue...values) {
		if (values == null || values.length == 0) return null;
		
		NeuronValue[] softmax = new NeuronValue[values.length];
		NeuronValue sum = values[0].zero();
		for (int i = 0; i < values.length; i++) {
			softmax[i] = values[i].exp();
			sum = sum.add(softmax[i]);
		}
		for (int i = 0; i < values.length; i++) softmax[i] = softmax[i].divide(sum);
		return softmax;
	}
	
	
	/**
	 * Calculating soft-max function of vector.
	 * @param values vector.
	 * @return soft-max function of vector.
	 */
	static double[] softmax(double... values) {
		if (values == null || values.length == 0) return null;
		
		double[] softmax = new double[values.length];
		double sum = 0;
		for (int i = 0; i < values.length; i++) {
			softmax[i] = Math.exp(values[i]);
			sum += softmax[i];
		}
		for (int i = 0; i < values.length; i++) softmax[i] = softmax[i]/sum;
		return softmax;
	}
	
	
	/**
	 * Calculating soft-max function of vector.
	 * @param values vector.
	 * @return soft-max function of vector.
	 */
	static NeuronValueV softmax(NeuronValueV vector) {
		double[] softmmax = softmax(vector.v());
		return new NeuronValueV(softmmax);
	}
	
	
	/**
	 * Calculating soft-max function of matrix by row.
	 * @param matrix matrix.
	 * @return soft-max function of matrix by row.
	 */
	static Matrix softmaxByRow(Matrix matrix) {
		if (matrix instanceof MatrixStack) return softmaxByRow((MatrixStack)matrix);
		
		Matrix softmax = matrix.create(new Size(matrix.columns(), matrix.rows()));
		NeuronValue zero = matrix.get(0, 0).zero();
		for (int row = 0; row < matrix.rows(); row++) {
			NeuronValue[] exps = new NeuronValue[matrix.columns()];
			NeuronValue sum = zero;
			for (int column = 0; column < matrix.columns(); column++) {
				exps[column] = matrix.get(row, column).exp();
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
	 * Calculating soft-max function of matrix by row.
	 * @param matrix matrix stack.
	 * @return soft-max function of matrix by row.
	 */
	static MatrixStack softmaxByRow(MatrixStack matrix) {
		int depth = matrix.depth();
		Matrix[] result = new Matrix[depth];
		for (int d = 0; d < depth; d++) result[d] = softmaxByRow(matrix.get(d));
		return new MatrixStack(result);
	}
	
	
	/**
	 * Calculating soft-max function of matrix by row.
	 * @param matrix matrix.
	 * @param row specified row.
	 * @return soft-max function of matrix by row.
	 */
	@Deprecated
	@SuppressWarnings("unused")
	private static NeuronValue[] softmaxByRow(Matrix matrix, int row) {
		NeuronValue[] softmax = new NeuronValue[matrix.columns()];
		NeuronValue sum = matrix.get(0, 0).zero();
		for (int column = 0; column < matrix.columns(); column++) {
			softmax[column] = matrix.get(row, column).exp();
			sum = sum.add(softmax[column]);
		}
		for (int column = 0; column < matrix.columns(); column++) softmax[column] = softmax[column].divide(sum);
		return softmax;
	}
	
	
	/**
	 * Calculating inverse soft-max function of matrix by row.
	 * @param matrix matrix.
	 * @return inverse soft-max function of matrix by row.
	 */
	@SuppressWarnings("unused")
	@Deprecated
	private static Matrix softmaxByRowInverse(Matrix matrix) {
		Matrix unitMatrix = matrix.create(new Size(matrix.columns(), matrix.rows()));
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
		if (matrix instanceof MatrixStack) return softmaxByColumn((MatrixStack)matrix);

		Matrix softmax = matrix.create(new Size(matrix.columns(), matrix.rows()));
		NeuronValue zero = matrix.get(0, 0).zero();
		for (int column = 0; column < matrix.columns(); column++) {
			NeuronValue[] exps = new NeuronValue[matrix.rows()];
			NeuronValue sum = zero;
			for (int row = 0; row < matrix.rows(); row++) {
				exps[row] = matrix.get(row, column).exp();
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
	 * Calculating soft-max function of matrix by column.
	 * @param matrix matrix stack.
	 * @return soft-max function of matrix by column.
	 */
	static MatrixStack softmaxByColumn(MatrixStack matrix) {
		int depth = matrix.depth();
		Matrix[] result = new Matrix[depth];
		for (int d = 0; d < depth; d++) result[d] = softmaxByColumn(matrix.get(d));
		return new MatrixStack(result);
	}

	
	/**
	 * Calculating soft-max function of matrix by column.
	 * @param matrix matrix.
	 * @return soft-max function of matrix by column.
	 */
	@Deprecated
	@SuppressWarnings("unused")
	private static NeuronValue[] softmaxByColumn(Matrix matrix, int column) {
		NeuronValue[] softmax = new NeuronValue[matrix.rows()];
		NeuronValue sum = matrix.get(0, 0).zero();
		for (int row = 0; row < matrix.rows(); row++) {
			softmax[row] = matrix.get(row, column).exp();
			sum = sum.add(softmax[row]);
		}
		for (int row = 0; row < matrix.rows(); row++) softmax[row] = softmax[row].divide(sum);
		return softmax;
	}
	
	
	/**
	 * Calculating inverse soft-max function of matrix by column.
	 * @param matrix matrix.
	 * @return inverse soft-max function of matrix by column.
	 */
	@Deprecated
	@SuppressWarnings("unused")
	private static Matrix softmaxByColumnInverse(Matrix matrix) {
		Matrix unitMatrix = matrix.create(new Size(matrix.columns(), matrix.rows()));
		NeuronValue unit = matrix.get(0, 0).unit();
		Matrix.fill(unitMatrix, unit);
		return softmaxByColumn(unitMatrix.subtract(matrix));
	}

	
	/**
	 * Calculating soft-max derivative of specified array.
	 * @param all specified array.
	 * @return soft-max derivative of specified array.
	 */
	@Deprecated
	@SuppressWarnings("unused")
	private static Matrix softmaxDerivative(NeuronValue[] all, Matrix hint) {
		NeuronValue[] softmax = softmaxArray(all);
		if (softmax == null || softmax.length == 0) return null;

		Matrix gradient = hint.create(new Size(all.length, all.length));
		NeuronValue unit = gradient.get(0, 0).unit();
		for (int row = 0; row < gradient.rows(); row++) {
			for (int column = 0; column < gradient.columns(); column++) {
				NeuronValue derivative = null;
				if (column == row)
					derivative = softmax[row].multiply(unit.subtract(softmax[column]));
				else
					derivative = softmax[row].multiply(softmax[column].negative());
				gradient.set(row, column, derivative);
			}
		}
		return gradient;
	}
	
	
	/**
	 * Calculating soft-max derivative on diagonal of specified array.
	 * @param all specified array.
	 * @return diagonal soft-max derivative of specified array.
	 */
	@Deprecated
	@SuppressWarnings("unused")
	private static NeuronValue[] softmaxDerivativeDiagonal(NeuronValue[] all) {
		NeuronValue[] softmax = softmaxArray(all);
		if (softmax == null || softmax.length == 0) return null;
		
		NeuronValue unit = all[0].unit();
		NeuronValue[] gradient = new NeuronValue[softmax.length];
		for (int i = 0; i < softmax.length; i++) {
			gradient[i] = softmax[i].multiply(unit.subtract(softmax[i]));
		}
		return gradient;
	}
	
	
	/**
	 * Creating soft-max function.
	 * @param neuronChannel neuron channel.
	 * @param layer layer.
	 * @return soft-max function.
	 */
	static Softmax create(int neuronChannel, LayerStandard layer) {
		if (neuronChannel < 1)
			return null;
		else if (neuronChannel == 1)
			return Softmax1.create(layer);
		else
			return SoftmaxV.create(layer);
	}

	
}
