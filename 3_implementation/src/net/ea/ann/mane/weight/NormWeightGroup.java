/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.weight;

import java.io.Serializable;

import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValue1;
import net.ea.ann.mane.Kernel;
import net.ea.ann.raster.Size;

/**
 * This class implement group norm weight developed by Yuxin Wu and Kaiming He.
 * @author Yuxin Wu, Kaiming He, developed by Loc Nguyen
 * @version 1.0
 *
 */
public class NormWeightGroup extends NormWeight {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * The default number of layers per group.
	 */
	final static int LAYERS_PER_GROUP_DEFAULT = 8;

	
	/**
	 * Constructor with the kernel.
	 * @param kernel the kernel.
	 */
	public NormWeightGroup(WKernel kernel) {
		super(kernel);
	}

	
	/**
	 * This class is the set of mean, standard deviation, and size.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	static class MeanStd implements Cloneable, Serializable {
		
		/**
		 * Serial version UID for serializable class.
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Means and standard deviations.
		 */
		public NeuronValue mean = null;
		
		/**
		 * Standard deviation.
		 */
		public NeuronValue std = null;
		
		/**
		 * Size.
		 */
		public int size = 0;
		
		/**
		 * Constructor with mean, standard deviation, and size.
		 * @param mean mean.
		 * @param std standard deviation.
		 * @param size size.
		 */
		public MeanStd(NeuronValue mean, NeuronValue std, int size) {
			this.mean = mean;
			this.std = std;
			this.size = size;
		}
		
	}
	
	
	/**
	 * Calculating means and standard deviations.
	 * @param matrices matrices.
	 * @param layersPerGroup layers per group.
	 * @return arrays of means and standard deviations where each array belongs to a group.
	 */
	private static MeanStd[] meanStds(Matrix[] matrices, int layersPerGroup) {
		int rows = matrices[0].rows(), columns = matrices[0].columns(), depth = matrices.length;
		if (layersPerGroup <= 0 || layersPerGroup > depth) throw new IllegalArgumentException();
		NeuronValue zero = matrices[0].get(0, 0).zero();
		NeuronValue epsilon = zero.valueOf(EPSILON);
		
		int groups = depth / layersPerGroup;
		MeanStd[] result = new MeanStd[groups];
		
		if (Kernel.speedMode(zero)) {
			for (int group = 0; group < groups; group++) {
				double mean = 0;
				double std = 0;
		
				int dStart = group*layersPerGroup;
				int dEnd = group < group-1 ? dStart+layersPerGroup : depth;
				int N = 0;
				for (int row = 0; row < rows; row++) {
					for (int column = 0; column < columns; column++) {
						for (int d = dStart; d < dEnd; d++) {
							mean += matrices[d].getv(row, column);
							N++;
						}
					}
				}
				mean = mean/(double)N;
				
				for (int row = 0; row < rows; row++) {
					for (int column = 0; column < columns; column++) {
						for (int d = dStart; d < dEnd; d++) {
							double dev = matrices[d].getv(row, column) - mean;
							std += dev*dev;
						}
					}
				}
				std = Math.sqrt((std/(double)N) + EPSILON);
				
				result[group] = new MeanStd(zero.valueOf(mean), zero.valueOf(std), N);
			}
		}
		else {
			for (int group = 0; group < groups; group++) {
				NeuronValue mean = zero;
				NeuronValue std = zero;
		
				int dStart = group*layersPerGroup;
				int dEnd = group < group-1 ? dStart+layersPerGroup : depth;
				int N = 0;
				for (int row = 0; row < rows; row++) {
					for (int column = 0; column < columns; column++) {
						for (int d = dStart; d < dEnd; d++) {
							mean = mean.add(matrices[d].get(row, column));
							N++;
						}
					}
				}
				mean = mean.divide(N);
				
				for (int row = 0; row < rows; row++) {
					for (int column = 0; column < columns; column++) {
						for (int d = dStart; d < dEnd; d++) {
							NeuronValue dev = matrices[d].get(row, column).subtract(mean);
							std = std.add(dev.multiply(dev));
						}
					}
				}
				std = std.divide(N).add(epsilon).sqrt();
				
				result[group] = new MeanStd(mean, std, N);
			}
		}
		
		return result;
	}


	/**
	 * Calculating number of layers per group.
	 * @param depth depth.
	 * @return number of layers per group.
	 */
	private static int calcLayersPerGroup(int depth) {
		return Math.min(depth, LAYERS_PER_GROUP_DEFAULT);
	}
	
	
	@Override
	public Matrix evaluate(Matrix input, Matrix bias) {
		assert (this.layer != null);
		if (W().rows() != 1 || W().columns() != 1 || MatrixUtil.depth(input) != W().depth()) throw new IllegalArgumentException();
		if (bias != null) {
			if (bias.rows() != W().rows() || bias.columns() != W().columns() || MatrixUtil.depth(bias) != W().depth()) throw new IllegalArgumentException();
		}
		if (this.bias() != null) {
			if (this.bias().rows() != W().rows() || this.bias().columns() != W().columns() || MatrixUtil.depth(this.bias()) != W().depth()) throw new IllegalArgumentException();
		}
		if (this.bias() != null && bias != null) {assert (this.bias() != bias);}

		int rows = input.rows(), columns = input.columns(), depth = W().depth();
		Matrix[] inputs = MatrixUtil.split(input);

		//Calculating means and standard deviations.
		NeuronValue[] means = new NeuronValue[depth], stds = new NeuronValue[depth];
		int layersPerGroup = calcLayersPerGroup(depth);
		MeanStd[] meanStds = meanStds(inputs, layersPerGroup);
		for (int d = 0; d < depth; d++) {
			int g = Math.min(d/layersPerGroup, depth/layersPerGroup - 1);
			means[d] = meanStds[g].mean;
			stds[d] = meanStds[g].std;
		}

		//Normalizing.
		Matrix[] prevOutputs = new Matrix[depth];
		Matrix[] outputs = new Matrix[depth];
		if (Kernel.speedMode(inputs[0].get(0, 0))) {
			for (int d = 0; d < depth; d++) {
				prevOutputs[d] = inputs[d].create(new Size(columns, rows));
				double mean = ((NeuronValue1)means[d]).get();
				double std = ((NeuronValue1)stds[d]).get();
				for (int row = 0; row < rows; row++) {
					for (int column = 0; column < columns; column++) {
						double z = (inputs[d].getv(row, column)-mean) / std;
						prevOutputs[d].setv(row, column, z);
					}
				}
				outputs[d] = prevOutputs[d].multiply0(((NeuronValue1)W(d)).get());
			}
		}
		else {
			for (int d = 0; d < depth; d++) {
				prevOutputs[d] = inputs[d].create(new Size(columns, rows));
				NeuronValue mean = means[d];
				NeuronValue std = stds[d];
				for (int row = 0; row < rows; row++) {
					for (int column = 0; column < columns; column++) {
						NeuronValue z = inputs[d].get(row, column).subtract(mean).divide(std);
						prevOutputs[d].set(row, column, z);
					}
				}
				outputs[d] = prevOutputs[d].multiply0(W(d));
			}
		}
		
		//Storing normalized previous output.
		if (this.layer != null) {
			this.layer.setPrevOutput(prevOutputs.length == 1 ? prevOutputs[0] : new MatrixStack(prevOutputs));
		}
		
		//Adding bias.
		Matrix thisBias = this.bias() != null ? (this.bias().depth() == 1 ? this.bias().get(0) : this.bias()) : null;
		Matrix bias0 = null;
		if (thisBias != null && bias != null)
			bias0 = Kernel.GLOBAL_BIAS ? thisBias.add(bias) : thisBias;
		else if (thisBias != null)
			bias0 = thisBias;
		else if (bias != null)
			bias0 = bias;
		
		if (bias0 != null) addBias(outputs, bias0);
		return outputs.length == 1 ? outputs[0] : new MatrixStack(outputs);
	}


	@Override
	MatrixStack dValue(MatrixStack prevOutputs, MatrixStack thisErrors) {
		assert (W().rows() == 1 && W().columns() == 1);
		int rows = prevOutputs.rows(), columns = prevOutputs.columns(), depth = W().depth();
		NeuronValue zero = prevOutputs.get(0).get(0, 0).zero();

		//Calculating means and standard deviations.
		NeuronValue[] means = new NeuronValue[depth], stds = new NeuronValue[depth];
		int[] sizes = new int[depth];
		int layersPerGroup = calcLayersPerGroup(depth);
		MeanStd[] meanStds = meanStds(MatrixUtil.split(prevOutputs), layersPerGroup);
		for (int d = 0; d < depth; d++) {
			int g = Math.min(d/layersPerGroup, depth/layersPerGroup - 1);
			means[d] = meanStds[g].mean;
			stds[d] = meanStds[g].std;
			sizes[d] = meanStds[g].size;
		}
		
		//Calculating value gradient.
		Matrix[] dValues = new Matrix[depth];
		if (Kernel.speedMode(zero)) {
			for (int d = 0; d < depth; d++) {
				Matrix prevOutput = prevOutputs.get(d);
				Matrix norm = prevOutput.create(new Size(columns, rows));
				double mean = ((NeuronValue1)means[d]).get();
				double std = ((NeuronValue1)stds[d]).get();
				for (int row = 0; row < rows; row++) {
					for (int column = 0; column < columns; column++) {
						double z = (prevOutput.getv(row, column)-mean) / std;
						norm.setv(row, column, z);
					}
				}
				norm = norm.multiply0(((NeuronValue1)W(d)).get());

				double w = ((NeuronValue1)W(d)).get();
				double errorSum = 0, normErrorSum = 0;
				for (int row = 0; row < rows; row++) {
					for (int column = 0; column < columns; column++) {
						double error = thisErrors.get(d).getv(row, column) * w;
						errorSum += error;
						double normError = error*norm.getv(row, column);
						normErrorSum += normError;
					}
				}
				
				int N = sizes[d];
				dValues[d] = prevOutput.create(new Size(columns, rows));
				double factor = std*N;
				for (int row = 0; row < rows; row++) {
					for (int column = 0; column < columns; column++) {
						double error = thisErrors.get(d).getv(row, column) * w;
						double bias = (error*N - errorSum - (norm.getv(row, column)*normErrorSum)) / factor;
						dValues[d].setv(row, column, bias);
					}
				}
			}
		}
		else {
			for (int d = 0; d < depth; d++) {
				Matrix prevOutput = prevOutputs.get(d);
				Matrix norm = prevOutput.create(new Size(columns, rows));
				NeuronValue mean = means[d];
				NeuronValue std = stds[d];
				for (int row = 0; row < rows; row++) {
					for (int column = 0; column < columns; column++) {
						NeuronValue z = prevOutput.get(row, column).subtract(mean).divide(std);
						norm.set(row, column, z);
					}
				}
				norm = norm.multiply0(W(d));
	
				NeuronValue w = W(d);
				NeuronValue errorSum = zero, normErrorSum = zero;
				for (int row = 0; row < rows; row++) {
					for (int column = 0; column < columns; column++) {
						NeuronValue error = thisErrors.get(d).get(row, column).multiply(w);
						errorSum = errorSum.add(error);
						NeuronValue normError = error.multiply(norm.get(row, column));
						normErrorSum = normErrorSum.add(normError);
					}
				}
				
				int N = sizes[d];
				dValues[d] = prevOutput.create(new Size(columns, rows));
				NeuronValue factor = std.multiply(N);
				for (int row = 0; row < rows; row++) {
					for (int column = 0; column < columns; column++) {
						NeuronValue error = thisErrors.get(d).get(row, column).multiply(w);
						NeuronValue bias = error.multiply(N)
							.subtract(errorSum)
							.subtract(norm.get(row, column).multiply(normErrorSum))
							.divide(factor);
						dValues[d].set(row, column, bias);
					}
				}
			}
		}
		
		return new MatrixStack(dValues);
	}


	@Override
	public Object clone() throws CloneNotSupportedException {
		WKernel clonedKernel = (WKernel)this.kernel.clone();
		NormWeightGroup cloned = new NormWeightGroup(clonedKernel);
		cloned.layer = this.layer;
		return cloned;
	}


	/**
	 * Creating norm weight.
	 * @param prevSize previous size.
	 * @param size current size.
	 * @param hint hint value.
	 * @return norm weight.
	 */
	public static NormWeightGroup create(Size prevSize, Size size, NeuronValue hint) {
		if (prevSize.width != size.width || prevSize.height != size.height || prevSize.depth != size.depth) throw new IllegalArgumentException();
		Matrix W = MatrixUtil.create(new Size(1, 1, size.depth, 1), hint.unit());
		Matrix bias = MatrixUtil.create(new Size(1, 1, size.depth, 1), hint.zero());
		WKernel kernel = new WKernel(W instanceof MatrixStack ? (MatrixStack)W : new MatrixStack(W),
			bias instanceof MatrixStack ? (MatrixStack)bias : new MatrixStack(bias));
		return new NormWeightGroup(kernel);
	}


}
