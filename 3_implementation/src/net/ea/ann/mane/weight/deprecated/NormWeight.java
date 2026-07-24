/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.weight.deprecated;

import net.ea.ann.core.TextParsable;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.Kernel;
import net.ea.ann.mane.MatrixLayerAbstract;
import net.ea.ann.mane.Weight;
import net.ea.ann.mane.train.AdamOptimizer;
import net.ea.ann.mane.train.Optimizer;
import net.ea.ann.raster.Size;

/**
 * This class implements normalization weight.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@Deprecated
public class NormWeight implements Weight, TextParsable {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Epsilon.
	 */
	public final static double EPSILON = 1E-5;
	
	
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
		 * Linear bias.
		 */
		protected MatrixStack bias = null;
		
		/**
		 * Optimizer.
		 */
		private Optimizer optimizer = null;
		
		/**
		 * Constructor with the linear weight and bias.
		 * @param W the linear weight.
		 * @param bias the linear bias.
		 */
		public WKernel(MatrixStack W, MatrixStack bias) {
			if (!checkValid(W, bias)) throw new IllegalArgumentException();
			this.W = W;
			this.bias = bias;
		}

		/**
		 * Checking the linear weight.
		 * @param W the linear weight.
		 * @return true if the linear weight is valid.
		 */
		private static boolean checkValid(MatrixStack W, MatrixStack bias) {
			if (W == null) return false;
			if (bias != null) {
				if (bias.rows() != W.rows() || bias.columns() != W.columns() || MatrixUtil.depth(bias) != W.depth()) return false;
			}
			return true;
		}

		@Override
		public WKernel add(Kernel kernel) {
			this.W = this.W != null ? (MatrixStack)this.W.add(((WKernel)kernel).W) : null;
			this.bias = this.bias != null ? (MatrixStack)this.bias.add(((WKernel)kernel).bias) : null;
			return this;
		}

		@Override
		public WKernel multiply(double value) {
			this.W = this.W != null ? (MatrixStack)this.W.multiply0(value) : null;
			this.bias = this.bias != null ? (MatrixStack)this.bias.multiply0(value) : null;
			return this;
		}

		@Override
		public WKernel divide(double value) {
			this.W = this.W != null ? (MatrixStack)this.W.divide0(value) : null;
			this.bias = this.bias != null ? (MatrixStack)this.bias.divide0(value) : null;
			return this;
		}

		@Override
		public Optimizer getOptimizer() {return optimizer;}

		@Override
		public void setOptimizer(Optimizer optimizer) {this.optimizer = optimizer;}
		
		@Override
		public WKernel optimize() {
			if (this.optimizer == null) System.out.println("WARNING: norm weight has no optimizer");
			if ((this.optimizer == null) || !(this.optimizer instanceof AdamOptimizer)) return (WKernel)Kernel.super.optimize();
			if (this.W == null) return (WKernel)Kernel.super.optimize();
			
			AdamOptimizer adam = (AdamOptimizer)this.optimizer;
			int time = adam.incTime();
			if (this.W != null) {
				Matrix W0 = adam.recalcGradient(this.W, time);
				this.W = W0 instanceof MatrixStack ? (MatrixStack)W0 : new MatrixStack(W0);
			}
			
			if (this.bias != null) {
				Matrix bias0 = adam.recalcGradient(this.bias, time);
				this.bias = bias0 instanceof MatrixStack ? (MatrixStack)bias0 : new MatrixStack(bias0);
			}
			
			return this;
		}
		
		
		/**
		 * Making L2 regularization.
		 * @param decay decay factor.
		 * @return this kernel.
		 */
		public WKernel L2(double decay) {
			assert (decay > 0 && decay <= 1);
			if (REGULAR) this.W = this.W != null ? (MatrixStack)this.W.multiply0(decay) : null;
			return this;
		}
		
		
	}

	
	/**
	 * The kernel.
	 */
	protected WKernel kernel = null;
	
	
	/**
	 * Referred layer.
	 */
	private MatrixLayerAbstract layer = null;
	
	
	/**
	 * Filter mode.
	 */
	private boolean depthMode = false;
	
	
	/**
	 * Constructor with the kernel.
	 * @param kernel the kernel.
	 */
	public NormWeight(WKernel kernel) {
		assert (kernel != null);
		this.kernel = kernel;
		if (kernel.W != null) {
			NeuronValue unit = kernel.W.get().get(0, 0).unit();
			MatrixUtil.fill(kernel.W, unit);
		}
		if (kernel.bias != null) {
			NeuronValue zero = kernel.bias.get().get(0, 0).zero();
			MatrixUtil.fill(kernel.bias, zero);
		}
		if (Kernel.OPTIMIZER) this.kernel.setOptimizer(this.kernel.createOptimizer());
	}

	
	/**
	 * Getting depth mode.
	 * @return depth mode.
	 */
	public boolean getDepthMode() {return depthMode;}
	
	
	/**
	 * Setting filter mode.
	 * @param depthMode depth mode.
	 */
	public void setDepthMode(boolean depthMode) {this.depthMode = depthMode;}
	
	
	/**
	 * Getting layer.
	 * @return layer.
	 */
	public MatrixLayerAbstract getLayer() {return layer;}
	
	
	/**
	 * Setting layer.
	 * @param layer layer.
	 */
	public void setLayer(MatrixLayerAbstract layer) {this.layer = layer;}
	
	
	/**
	 * Getting the weight.
	 * @return the weight.
	 */
	private MatrixStack W() {return kernel != null ? kernel.W : null;}

	
	/**
	 * Getting bias.
	 * @return bias.
	 */
	private MatrixStack bias() {return kernel != null ? kernel.bias : null;}
	
	
	@Override
	public WKernel kernel() {return this.kernel;}


	@Override
	public NormWeight accumKernel(Kernel dKernel, double factor) {
		assert (factor > 0 && factor < 1);
		if (dKernel == this.kernel) throw new IllegalArgumentException();
		if (dKernel.getOptimizer() == null) dKernel.setOptimizer(this.kernel.getOptimizer());
		if (dKernel.getOptimizer() == this.kernel.getOptimizer()) dKernel = dKernel.optimize();
		
		this.kernel = this.kernel.add(dKernel.multiply(factor));
		return this;
	}

	
	@Override
	public NormWeight accumKernel(Kernel dKernel, double factor, double decay) {
		assert (factor > 0 && factor < 1);
		if (dKernel == this.kernel) throw new IllegalArgumentException();
		if (dKernel.getOptimizer() == null) dKernel.setOptimizer(this.kernel.getOptimizer());
		if (dKernel.getOptimizer() == this.kernel.getOptimizer()) dKernel = dKernel.optimize();
		
		this.kernel = this.kernel/*.L2(decay)*/.add(dKernel.multiply(factor)); //L2 regularization should not be applied into linear norm weight.
		return this;
	}

	
	/**
	 * Calculating means and standard deviations.
	 * @param outputs outputs.
	 * @return array of means, standard deviations, and norms.
	 */
	public static Matrix[] meanStds(Matrix[] outputs) {
		int rows = outputs[0].rows(), columns = outputs[0].columns(), depth = outputs.length;
		NeuronValue zero = outputs[0].get(0, 0).zero();
		NeuronValue epsilon = zero.valueOf(EPSILON);
		Matrix means = outputs[0].create(new Size(columns, rows));
		Matrix stds = outputs[0].create(new Size(columns, rows));
		MatrixUtil.fill(means, zero);
		MatrixUtil.fill(stds, zero);

		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < columns; column++) {
				NeuronValue mean = zero;
				for (int d = 0; d < depth; d++) mean = mean.add(outputs[d].get(row, column));
				means.set(row, column, mean.divide(depth));
			}
		}
		
		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < columns; column++) {
				NeuronValue std = zero, mean = means.get(row, column);
				for (int d = 0; d < depth; d++) {
					NeuronValue dev = outputs[d].get(row, column).subtract(mean);
					std = std.add(dev.multiply(dev));
				}
				stds.set(row, column, std.divide(depth).add(epsilon).sqrt());
			}
		}
		
		return new Matrix[] {means, stds};
	}

	
	@Override
	public Matrix evaluate(Matrix input, Matrix bias) {
		if (input.rows() != W().rows() || input.columns() != W().columns() || MatrixUtil.depth(input) != W().depth()) throw new IllegalArgumentException();
		if (bias != null) {
			if (bias.rows() != W().rows() || bias.columns() != W().columns() || MatrixUtil.depth(bias) != W().depth()) throw new IllegalArgumentException();
		}
		if (this.bias() != null) {
			if (this.bias().rows() != W().rows() || this.bias().columns() != W().columns() || MatrixUtil.depth(this.bias()) != W().depth()) throw new IllegalArgumentException();
		}
		if (this.bias() != null && bias != null) {assert (this.bias() != bias);}

		int rows = input.rows(), columns = input.columns(), depth = W().depth();
		MatrixStack inputs = input instanceof MatrixStack ? (MatrixStack)input : new MatrixStack(input);
		NeuronValue zero = inputs.get(0).get(0, 0).zero();
		boolean acrossDepth = depthMode && depth > 1;

		//Calculating means and standard deviations.
		Matrix means = null, stds = null;
		NeuronValue[] mean0 = null, std0 = null;
		if (acrossDepth) {
			Matrix[] meanStds = meanStds(MatrixUtil.split(inputs));
			means = meanStds[0];
			stds = meanStds[1];
		}
		else {
			NeuronValue epsilon = zero.valueOf(EPSILON);
			mean0 = new NeuronValue[depth];
			std0 = new NeuronValue[depth];
			for (int d = 0; d < depth; d++) {
				mean0[d] = MatrixUtil.valueMean(inputs.get(d));
				std0[d] = MatrixUtil.valueVariance(inputs.get(d)).add(epsilon).sqrt();
			}
		}

		//Normalizing.
		Matrix[] prevOutputs = new Matrix[depth];
		Matrix[] outputs = new Matrix[depth];
		for (int d = 0; d < depth; d++) {
			prevOutputs[d] = inputs.get(d).create(new Size(columns, rows));
			for (int row = 0; row < rows; row++) {
				for (int column = 0; column < columns; column++) {
					NeuronValue mean = acrossDepth ? means.get(row, column) : mean0[d];
					NeuronValue std = acrossDepth ? stds.get(row, column) : std0[d];
					NeuronValue z = inputs.get(d).get(row, column).subtract(mean).divide(std);
					prevOutputs[d].set(row, column, z);
				}
			}
			outputs[d] = W().get(d).multiplyWise(prevOutputs[d]);
		}
		
		//Storing normalized previous output.
		if (this.layer != null) {
			this.layer.setPrevOutput(prevOutputs.length == 1 ? prevOutputs[0] : new MatrixStack(prevOutputs));
		}
		
		//Adding bias.
		Matrix output = outputs.length == 1 ? outputs[0] : new MatrixStack(outputs);
		Matrix thisBias = this.bias() != null ? (this.bias().depth() == 1 ? this.bias().get(0) : this.bias()) : null;
		Matrix bias0 = null;
		if (thisBias != null && bias != null)
			bias0 = Kernel.GLOBAL_BIAS ? thisBias.add(bias) : thisBias;
		else if (thisBias != null)
			bias0 = thisBias;
		else if (bias != null)
			bias0 = bias;
		return bias0 != null ? output.add(bias0) : output;
	}

	
	/**
	 * Calculate gradient of previous layers. Please pay attention to this method.
	 * @param prevOutputs previous outputs.
	 * @param thisErrors current errors.
	 * @param W current weight matrices.
	 * @return gradient of previous layers.
	 */
	private static MatrixStack dValue(MatrixStack prevOutputs, MatrixStack thisErrors, MatrixStack W, boolean filterMode) {
		int rows = prevOutputs.rows(), columns = prevOutputs.columns(), depth = W.depth();
		NeuronValue zero = prevOutputs.get(0).get(0, 0).zero();
		boolean acrossDepth = filterMode && depth > 1;

		//Calculating means and standard deviations.
		Matrix means = null, stds = null;
		NeuronValue[] mean0 = null, std0 = null;
		if (acrossDepth) {
			Matrix[] meanStds = meanStds(MatrixUtil.split(prevOutputs));
			means = meanStds[0];
			stds = meanStds[1];
		}
		else {
			NeuronValue epsilon = zero.valueOf(EPSILON);
			mean0 = new NeuronValue[depth];
			std0 = new NeuronValue[depth];
			for (int d = 0; d < depth; d++) {
				mean0[d] = MatrixUtil.valueMean(prevOutputs.get(d));
				std0[d] = MatrixUtil.valueVariance(prevOutputs.get(d)).add(epsilon).sqrt();
			}
		}
		
		//Calculating value gradient.
		Matrix[] dValues = new Matrix[depth];
		for (int d = 0; d < depth; d++) {
			Matrix prevOutput = prevOutputs.get(d);
			Matrix norm = prevOutput.create(new Size(columns, rows));
			for (int row = 0; row < rows; row++) {
				for (int column = 0; column < columns; column++) {
					NeuronValue mean = acrossDepth ? means.get(row, column) : mean0[d];
					NeuronValue std = acrossDepth ? stds.get(row, column) : std0[d];
					NeuronValue z = prevOutput.get(row, column).subtract(mean).divide(std);
					norm.set(row, column, z);
				}
			}
			norm = W.get(d).multiplyWise(norm);

			Matrix w = W.get(d);
			NeuronValue errorSum = zero, normErrorSum = zero;
			for (int row = 0; row < rows; row++) {
				for (int column = 0; column < columns; column++) {
					NeuronValue error = thisErrors.get(d).get(row, column).multiply(w.get(row, column));
					errorSum = errorSum.add(error);
					NeuronValue normError = error.multiply(norm.get(row, column));
					normErrorSum = normErrorSum.add(normError);
				}
			}
			
			int N = acrossDepth ? depth : rows*columns;
			dValues[d] = prevOutput.create(new Size(columns, rows));
			for (int row = 0; row < rows; row++) {
				for (int column = 0; column < columns; column++) {
					NeuronValue std = acrossDepth ? stds.get(row, column) : std0[d];
					NeuronValue factor = std.multiply(N);
					
					NeuronValue error = thisErrors.get(d).get(row, column).multiply(w.get(row, column));
					NeuronValue bias = error.multiply(N)
						.subtract(errorSum)
						.subtract(norm.get(row, column).multiply(normErrorSum))
						.divide(factor);
					dValues[d].set(row, column, bias);
				}
			}
		}
		
		return new MatrixStack(dValues);
	}

	
	/**
	 * Calculate gradient of previous layers.
	 * @param prevOutputs previous outputs.
	 * @param thisErrors current errors.
	 * @return gradient of previous layers.
	 */
	private MatrixStack dValue(MatrixStack prevOutputs, MatrixStack thisErrors) {
		return dValue(prevOutputs, thisErrors, W(), depthMode);
	}

	
	@Override
	public Matrix dValue(Matrix prevOutput, Matrix thisError) {
		if (prevOutput.rows() != W().rows() || prevOutput.columns() != W().columns() || MatrixUtil.depth(prevOutput) != W().depth()) throw new IllegalArgumentException();
		if (thisError.rows() != W().rows() || thisError.columns() != W().columns() || MatrixUtil.depth(thisError) != W().depth()) throw new IllegalArgumentException();

		MatrixStack prevOutputs = prevOutput instanceof MatrixStack ? (MatrixStack)prevOutput : new MatrixStack(prevOutput);
		MatrixStack thisErrors = thisError instanceof MatrixStack ? (MatrixStack)thisError : new MatrixStack(thisError);
		MatrixStack dValue = dValue(prevOutputs, thisErrors);
		return dValue.depth() == 1 ? dValue.get() : dValue;
	}

	
	@Override
	public Kernel dKernel(Matrix prevOutput, Matrix thisError) {
		if (prevOutput.rows() != W().rows() || prevOutput.columns() != W().columns() || MatrixUtil.depth(prevOutput) != W().depth()) throw new IllegalArgumentException();
		if (thisError.rows() != W().rows() || thisError.columns() != W().columns() || MatrixUtil.depth(thisError) != W().depth()) throw new IllegalArgumentException();
		if (this.bias() != null) {
			if (this.bias().rows() != W().rows() || this.bias().columns() != W().columns() || MatrixUtil.depth(this.bias()) != W().depth()) throw new IllegalArgumentException();
		}
		assert (this.bias() != null);

		MatrixStack prevOutputs = prevOutput instanceof MatrixStack ? (MatrixStack)prevOutput : new MatrixStack(prevOutput);
		MatrixStack thisErrors = thisError instanceof MatrixStack ? (MatrixStack)thisError : new MatrixStack(thisError);
		Matrix dW = prevOutputs.multiplyWise(thisErrors);
		Matrix dBias = thisErrors;
		
		WKernel dKernel = new WKernel(dW instanceof MatrixStack ? (MatrixStack)dW : new MatrixStack(dW),
			this.bias() != null ? (dBias instanceof MatrixStack ? (MatrixStack)dBias : new MatrixStack(dBias)) : null);
		if (this.kernel() != null) dKernel.setOptimizer(this.kernel().getOptimizer());
		return dKernel;
	}

	
//	@Override
//	public void initParams(double v) {
//		MatrixStack W = this.W(), bias = this.bias();
//		if (W != null) MatrixUtil.fill(W, v);
//		if (bias != null) MatrixUtil.fill(bias, v);
//	}
//	
//
//	@Override
//	public void initParams(Random rnd) {
//		MatrixStack W = this.W(), bias = this.bias();
//		if (W != null) MatrixUtil.fill(W, rnd, 1);
//		if (bias != null) MatrixUtil.fill(bias, rnd);
//	}


	@Override
	public int sizeOfParams() {
		MatrixStack W = this.W(), bias = this.bias();
		return (W != null ? MatrixUtil.capacity(W) : 0) + (bias != null ? MatrixUtil.capacity(bias) : 0);
	}
	

	@Override
	public String toText() {
		MatrixStack W = this.W(), bias = this.bias();
		if (W == null && bias == null) return "{}";
		
		StringBuffer buffer = new StringBuffer();
		buffer.append("{");
		buffer.append("W = " + (W!=null?W.toText():"") + "");
		buffer.append("bias = " + (bias!=null?bias.toText():"") + "");
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
		Matrix W = MatrixUtil.create(new Size(size.width, size.height, size.depth, 1), hint.unit());
		Matrix bias = MatrixUtil.create(new Size(size.width, size.height, size.depth, 1), hint.zero());
		WKernel kernel = new WKernel(W instanceof MatrixStack ? (MatrixStack)W : new MatrixStack(W),
			bias instanceof MatrixStack ? (MatrixStack)bias : new MatrixStack(bias));
		return new NormWeight(kernel);
	}
	
	
}
