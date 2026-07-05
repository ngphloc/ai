/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.weight;

import java.util.Random;

import net.ea.ann.core.TextParsable;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.Kernel;
import net.ea.ann.mane.Weight;
import net.ea.ann.raster.Size;

/**
 * This class implements normalization weight.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class NormWeight implements Weight, TextParsable {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Epsilon.
	 */
	private final static double EPSILON = Float.MIN_VALUE;
	
	
	/**
	 * This kernel consists of the linear weight.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static class WKernel implements Kernel {
		
		/**
		 * Serial version UID for serializable class.
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * The linear weight.
		 */
		protected MatrixStack W = null;
		
		/**
		 * Constructor with the linear weight.
		 * @param W the linear weight.
		 */
		public WKernel(MatrixStack W) {
			if (!checkValid(W)) throw new IllegalArgumentException();
			this.W = W;
		}

		/**
		 * Checking the linear weight.
		 * @param W the linear weight.
		 * @return true if the linear weight is valid.
		 */
		private static boolean checkValid(MatrixStack W) {
			if (W == null /*|| W.columns() != 1*/) return false;
			return true;
		}

		@Override
		public WKernel add(Kernel kernel) {
			MatrixStack W0 = (MatrixStack)this.W.add(((WKernel)kernel).W);
			return new WKernel(W0);
		}

		@Override
		public WKernel multiply(double value) {
			MatrixStack W0 = (MatrixStack)this.W.multiply0(value);
			return new WKernel(W0);
		}

		@Override
		public WKernel divide(double value) {
			MatrixStack W0 = (MatrixStack)this.W.divide0(value);
			return new WKernel(W0);
		}

	}

	
	/**
	 * The kernel.
	 */
	protected WKernel kernel = null;
	
	
	/**
	 * Constructor with the kernel.
	 * @param kernel the kernel.
	 */
	public NormWeight(WKernel kernel) {
		this.kernel = kernel;
	}

	
	/**
	 * Getting the weight.
	 * @return the weight.
	 */
	private MatrixStack W() {return kernel != null ? kernel.W : null;}

	
	@Override
	public Weight accumKernel(Kernel dKernel, double factor) {
		this.kernel = (WKernel)this.kernel.add(dKernel.multiply(factor));
		return this;
	}

	
	@Override
	public Weight accumKernel(Kernel dKernel, double factor, double decay) {
		this.kernel = (WKernel)this.kernel.multiply(decay).add(dKernel.multiply(factor));
		return this;
	}

	
	@Override
	public Matrix evaluate(Matrix input, Matrix bias) {
		if (input.rows() != W().rows() || input.columns() != W().columns() || MatrixUtil.depth(input) != W().depth()) throw new IllegalArgumentException();
		if (bias != null) {
			if (bias.rows() != W().rows() || bias.columns() != W().columns() || MatrixUtil.depth(bias) != W().depth()) throw new IllegalArgumentException();
		}

		int rows = input.rows(), columns = input.columns(), depth = W().depth();
		MatrixStack inputs = input instanceof MatrixStack ? (MatrixStack)input : new MatrixStack(input);
		Matrix[] outputs = new Matrix[depth];
		for (int d = 0; d < depth; d++) {
			Matrix input0 = inputs.get(d);
			NeuronValue zero = input0.get(0, 0).zero();
			NeuronValue epsilon = zero.valueOf(EPSILON);
			NeuronValue mean = MatrixUtil.valueMean(input0);
			NeuronValue std = MatrixUtil.valueVariance(input0).add(epsilon).sqrt();
			
			outputs[d] = input0.create(new Size(columns, rows));
			for (int row = 0; row < rows; row++) {
				for (int column = 0; column < columns; column++) {
					NeuronValue z = inputs.get(d).get(row, column).subtract(mean).divide(std);
					outputs[d].set(row, column, z);
				}
			}
			outputs[d] = W().get(d).multiplyWise(outputs[d]);
		}
		
		Matrix output = outputs.length == 1 ? outputs[0] : new MatrixStack(outputs);
		if (bias != null) output = output.add(bias);
		return output;
	}

	
	/**
	 * Calculate gradient of previous layers.
	 * @param prevInputs previous inputs.
	 * @param prevOutputs previous outputs.
	 * @param thisErrors current errors.
	 * @param prevActivateRef previous activation function.
	 * @return gradient of previous layers.
	 */
	private MatrixStack dValue(MatrixStack prevInputs, MatrixStack prevOutputs, MatrixStack thisErrors, Function prevActivateRef) {
		int rows = prevOutputs.rows(), columns = prevOutputs.columns(), depth = W().depth();
		Matrix[] dValues = new Matrix[depth];
		for (int d = 0; d < depth; d++) {
			Matrix prevOutput = prevOutputs.get(d);
			NeuronValue zero = prevOutput.get(0, 0).zero();
			NeuronValue epsilon = zero.valueOf(EPSILON);
			NeuronValue mean = MatrixUtil.valueMean(prevOutput);
			NeuronValue std = MatrixUtil.valueVariance(prevOutput).add(epsilon).sqrt();

			Matrix norm = prevOutput.create(new Size(columns, rows));
			for (int row = 0; row < rows; row++) {
				for (int column = 0; column < columns; column++) {
					NeuronValue z = prevOutput.get(row, column).subtract(mean).divide(std);
					norm.set(row, column, z);
				}
			}
			norm = W().get(d).multiplyWise(norm);

			Matrix w = W().get(d);
			NeuronValue errorSum = zero, normErrorSum = zero;
			for (int row = 0; row < rows; row++) {
				for (int column = 0; column < columns; column++) {
					NeuronValue error = thisErrors.get(d).get(row, column).multiply(w.get(row, column));
					errorSum = errorSum.add(error);
					NeuronValue normError = error.multiply(norm.get(row, column));
					normErrorSum = normErrorSum.add(normError);
				}
			}
			
			int N = rows*columns;
			NeuronValue factor = std.multiply(N);
			dValues[d] = prevOutput.create(new Size(columns, rows));
			for (int row = 0; row < rows; row++) {
				for (int column = 0; column < columns; column++) {
					NeuronValue error = thisErrors.get(d).get(row, column).multiply(w.get(row, column));
					NeuronValue bias = error.multiply(N)
						.subtract(errorSum)
						.subtract(norm.get(row, column).multiply(normErrorSum))
						.divide(factor);
					dValues[d].set(row, column, bias);
				}
			}
		}
		
		MatrixStack dValue = new MatrixStack(dValues);
		MatrixStack derivative = prevActivateRef != null ? (MatrixStack)prevInputs.derivativeWise(prevActivateRef) : null;
		return derivative != null ? (MatrixStack)derivative.multiplyWise(dValue) : dValue;
	}

	
	@Override
	public Matrix dValue(Matrix prevInput, Matrix prevOutput, Matrix thisError, Function prevActivateRef) {
		if (prevInput.rows() != W().rows() || prevInput.columns() != W().columns() || MatrixUtil.depth(prevInput) != W().depth()) throw new IllegalArgumentException();
		if (prevOutput.rows() != W().rows() || prevOutput.columns() != W().columns() || MatrixUtil.depth(prevOutput) != W().depth()) throw new IllegalArgumentException();
		if (thisError.rows() != W().rows() || thisError.columns() != W().columns() || MatrixUtil.depth(thisError) != W().depth()) throw new IllegalArgumentException();

		MatrixStack prevInputs = prevInput instanceof MatrixStack ? (MatrixStack)prevInput : new MatrixStack(prevInput);
		MatrixStack prevOutputs = prevOutput instanceof MatrixStack ? (MatrixStack)prevOutput : new MatrixStack(prevOutput);
		MatrixStack thisErrors = thisError instanceof MatrixStack ? (MatrixStack)thisError : new MatrixStack(thisError);
		MatrixStack dValue = dValue(prevInputs, prevOutputs, thisErrors, prevActivateRef);
		return dValue.depth() == 1 ? dValue.get() : dValue;
	}

	
	@Override
	public Kernel dKernel(Matrix prevOutput, Matrix thisError) {
		if (prevOutput.rows() != W().rows() || prevOutput.columns() != W().columns() || MatrixUtil.depth(prevOutput) != W().depth()) throw new IllegalArgumentException();
		if (thisError.rows() != W().rows() || thisError.columns() != W().columns() || MatrixUtil.depth(thisError) != W().depth()) throw new IllegalArgumentException();

		MatrixStack prevOutputs = prevOutput instanceof MatrixStack ? (MatrixStack)prevOutput : new MatrixStack(prevOutput);
		MatrixStack thisErrors = thisError instanceof MatrixStack ? (MatrixStack)thisError : new MatrixStack(thisError);
		return new WKernel((MatrixStack)prevOutputs.multiplyWise(thisErrors));
	}

	
	@Override
	public void initParams(double v) {
		MatrixStack W = W();
		if (W != null) MatrixUtil.fill(W, v);
	}
	

	@Override
	public void initParams(Random rnd) {
		MatrixStack W = W();
		if (W != null) MatrixUtil.fill(W, rnd, 1);
	}


	@Override
	public int sizeOfParams() {
		MatrixStack W = W();
		return W != null ? MatrixUtil.capacity(W) : 0;
	}
	

	@Override
	public String toText() {
		MatrixStack W = W();
		if (W == null) return "{}";
		
		StringBuffer buffer = new StringBuffer();
		buffer.append("{");
		buffer.append("W = " + W.toText() + "");
		buffer.append("}");
		return buffer.toString();
	}


	/**
	 * Creating weight.
	 * @param prevSize previous size.
	 * @param size current size.
	 * @param hint hint value.
	 * @return weight.
	 */
	public static NormWeight create(Size prevSize, Size size, NeuronValue hint) {
		if (prevSize.width != size.width || prevSize.height != size.height || prevSize.depth != size.depth) throw new IllegalArgumentException();
		Matrix W = MatrixUtil.create(new Size(size.width, size.height, size.depth, 1), hint); 
		WKernel kernel = new WKernel(W instanceof MatrixStack ? (MatrixStack)W : new MatrixStack(W));
		return new NormWeight(kernel);
	}
	
	
}
