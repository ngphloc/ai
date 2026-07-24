/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.layers;

import java.util.Arrays;
import java.util.List;

import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.CustomLayer;
import net.ea.ann.mane.Error;
import net.ea.ann.mane.Error.LayerInput;
import net.ea.ann.mane.Kernel;
import net.ea.ann.mane.MatrixNetworkAbstract;
import net.ea.ann.mane.weight.NormWeight;
import net.ea.ann.mane.weight.NullWeight;
import net.ea.ann.raster.Size;

/**
 * This class represents batch normalization layer.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class BatchNormLayer extends CustomLayer {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Momentum.
	 */
	public static final double MOMENTUM = 0.9;
	
	
	/**
	 * This kernel consists of the linear weight.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	protected class WKernel extends net.ea.ann.mane.weight.NormWeight.WKernel {
		
		/**
		 * Serial version UID for serializable class.
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Constructor with the linear weight and bias.
		 * @param W the linear weight.
		 * @param bias the linear bias.
		 */
		public WKernel(MatrixStack W, MatrixStack bias) {
			super(W, bias);
		}
		
		/**
		 * Getting the weight.
		 * @return the weight.
		 */
		public MatrixStack W() {return this.W;}
		
		/**
		 * Getting bias.
		 * @return bias.
		 */
		public MatrixStack bias() {return this.bias;}
		
	}
	
	
	/**
	 * Kernel.
	 */
	protected WKernel kernel = null;
	
	
	/**
	 * Running mean.
	 */
	protected Matrix runningMean = null;
	
	
	/**
	 * Running standard deviation.
	 */
	protected Matrix runningStd = null;
	
	
	/**
	 * Batch of recent outputs.
	 */
	protected List<Matrix> batchOutputs = Util.newList(0);
	
	
	/**
	 * Batch of recent errors.
	 */
	protected List<Matrix> batchErrors = Util.newList(0);

	
	/**
	 * Backward information: Accumulation of gradients of norm weights.
	 */
	WKernel dWKernelNormAccum = null;

	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public BatchNormLayer(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public BatchNormLayer(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public BatchNormLayer(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public BatchNormLayer(int neuronChannel) {this(neuronChannel, null, null, null);}


	/**
	 * Getting the kernel weight.
	 * @return the kernel weight.
	 */
	private MatrixStack kerW() {return kernel != null ? kernel.W() : null;}

	
	/**
	 * Getting weight at depth.
	 * @param depth depth.
	 * @return weight at depth.
	 */
	private NeuronValue kerW(int depth) {
		MatrixStack W = kerW();
		return W != null ? W.get(depth).get(0, 0) : null;
	}

	
	/**
	 * Getting kernel bias.
	 * @return kernel bias.
	 */
	private MatrixStack kerBias() {return kernel != null ? kernel.bias() : null;}

	
	/**
	 * Accumulating kernel for L2 regularization.
	 * @param dKernel kernel bias.
	 * @param factor factor.
	 * @param decay decay which is factor of L2 regularization.
	 * @return this weight.
	 */
	public WKernel accumKernel(WKernel dKernel, double factor, double decay) {
		assert (factor > 0 && factor < 1);
		if (dKernel == this.kernel) throw new IllegalArgumentException();
		if (dKernel.getOptimizer() == null) dKernel.setOptimizer(this.kernel.getOptimizer());
		if (dKernel.getOptimizer() == this.kernel.getOptimizer()) dKernel = (WKernel)dKernel.optimize();
		
		this.kernel = (WKernel)this.kernel/*.L2(decay)*/.add(dKernel.multiply(factor)); //L2 regularization should not be applied into linear norm weight.
		return this.kernel;
	}

	
	/**
	 * Getting batch size.
	 * @return batch size.
	 */
	@SuppressWarnings("unused")
	@Deprecated
	private int paramGetBatchSize() {
		MatrixNetworkAbstract network = getNetwork();
		return network != null ? network.paramGetBatchSize() : NetworkAbstract.BATCH_SIZE_DEFAULT;
	}
	
	
	/**
	 * Getting batch outputs.
	 * @return batch outputs.
	 */
	private MatrixStack[] getBatchOutputs() {
		if (this.batchOutputs.size() == 0) return null;
		MatrixStack[] batchOutputStacks = new MatrixStack[this.batchOutputs.size()];
		for (int i = 0; i < this.batchOutputs.size(); i++) {
			batchOutputStacks[i] = this.batchOutputs.get(i) instanceof MatrixStack ? (MatrixStack)this.batchOutputs.get(i) : new MatrixStack(this.batchOutputs.get(i));
		}
		return batchOutputStacks;
	}
	
	
	@Override
	public void reset() {
		super.reset();
		this.kernel = null;
		this.runningMean = null;
		this.runningStd = null;
	}


	@Override
	protected void resetBackwardInfo() {
		super.resetBackwardInfo();
		this.batchOutputs.clear();
		this.batchErrors.clear();
		this.dWKernelNormAccum = null;
	}


	@Override
	public boolean initialize(Size size, Size prevSize, LayerSpec layerSpec) {
		if (size == null || prevSize == null) return false;
		if (size.width != prevSize.width || size.height != prevSize.height || size.depth != prevSize.depth) return false;

		this.prevInput = this.prevOutput = null;
		this.input = this.output = null;
		this.weight = null;
		this.bias = null;
		this.filter = null;
		this.filterBias = null;
		this.kernel = null;
		this.runningMean = null;
		this.runningStd = null;
		this.setPrevLayer(null);
		this.setNextLayer(null);
		this.resetBackwardInfo();

		this.input = newMatrix(new Size(size.width, size.height, size.depth));
		this.output = newMatrix(new Size(size.width, size.height, size.depth));
		this.bias = newMatrix(new Size(size.width, size.height, size.depth));

		NeuronValue zero = this.output instanceof MatrixStack ? ((MatrixStack)this.output).get().get(0, 0).zero() : this.output.get(0, 0).zero();
		NeuronValue unit = zero.unit();
		Matrix kerW = MatrixUtil.create(new Size(1, 1, size.depth, 1), zero);
		MatrixUtil.fill(kerW, unit);
		Matrix kerBias = MatrixUtil.create(new Size(1, 1, size.depth, 1), zero);
		MatrixUtil.fill(kerBias, zero);
		this.kernel = new WKernel(kerW instanceof MatrixStack ? (MatrixStack)kerW : new MatrixStack(kerW),
			kerBias instanceof MatrixStack ? (MatrixStack)kerBias : new MatrixStack(kerBias));
		if (Kernel.OPTIMIZER) this.kernel.setOptimizer(this.kernel.createOptimizer());

		this.runningMean = newMatrix(new Size(size.width, size.height, size.depth));
		this.runningStd = newMatrix(new Size(size.width, size.height, size.depth));
		MatrixUtil.fill(this.runningMean, zero);
		MatrixUtil.fill(this.runningStd, unit);
		return true;
	}


	/**
	 * Calculate norms and adjusted norms.
	 * @param output output.
	 * @param means means.
	 * @param stds standard deviations.
	 * @return norms and adjusted norms.
	 */
	private static Matrix[] norms(Matrix output, Matrix mean, Matrix std, NeuronValue w) {
		int rows = output.rows(), columns = output.columns();
		Matrix norm = output.create(new Size(columns, rows));
		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < columns; column++) {
				NeuronValue mean0 = mean.get(row, column), std0 = std.get(row, column);
				NeuronValue z = output.get(row, column).subtract(mean0).divide(std0);
				norm.set(row, column, z);
			}
		}
		Matrix adjustedNorm = norm.multiply0(w);
		return new Matrix[] {norm, adjustedNorm};
	}
	
	
	/**
	 * Calculating means, standard deviations, and norms.
	 * @param batchOutputs batch of outputs.
	 * @param output one output.
	 * @param w current weight matrix.
	 * @return array of means, standard deviations, and norms.
	 */
	@Deprecated
	private static Matrix[] meanStdNorms(Matrix[] batchOutputs, Matrix output, NeuronValue w) {
		if (batchOutputs == null || batchOutputs.length == 0) throw new IllegalArgumentException();
		Matrix[] meanStds = NormWeight.meanStds(batchOutputs);
		Matrix means = meanStds[0], stds = meanStds[1];
		Matrix[] norms = norms(output, means, stds, w);
		return new Matrix[] {means, stds, norms[0], norms[1]};
	}
	
	
	/**
	 * Calculate means and standard deviations.
	 * @param batchOutputStacks batch of outputs.
	 * @return means and standard deviations.
	 */
	private static MatrixStack[] meanStds(MatrixStack[] batchOutputStacks) {
		Matrix[] means = new Matrix[batchOutputStacks[0].depth()];
		Matrix[] stds = new Matrix[batchOutputStacks[0].depth()];
		for (int d = 0; d < batchOutputStacks[0].depth(); d++) {
			Matrix[] batchOutputs = batchOutputStacks != null && batchOutputStacks.length > 0 ? new Matrix[batchOutputStacks.length] : null;
			if (batchOutputs != null) {
				for (int i = 0; i < batchOutputStacks.length; i++) batchOutputs[i] = batchOutputStacks[i].get(d);
			}
			Matrix[] meanStds = NormWeight.meanStds(batchOutputs);
			means[d] = meanStds[0];
			stds[d] = meanStds[1];
		}
		return new MatrixStack[] {new MatrixStack(means), new MatrixStack(stds)};
	}
	
	
	/**
	 * Calculate means and standard deviations.
	 * @return means and standard deviations.
	 */
	private Matrix[] meanStds() {
		MatrixStack[] meanStds = meanStds(getBatchOutputs());
		Matrix[] result = new Matrix[meanStds.length];
		for (int i = 0; i < meanStds.length; i++) {
			result[i] = meanStds[i] instanceof MatrixStack && ((MatrixStack)meanStds[i]).depth() == 1 ? ((MatrixStack)meanStds[i]).get(0) : meanStds[i];
		}
		return result;
	}
	
	
	/**
	 * Calculating norms and adjusted norms.
	 * @param outputs outputs.
	 * @param means means.
	 * @param stds standard deviations.
	 * @return norms and adjusted norms.
	 */
	private MatrixStack[] norms(MatrixStack outputs, MatrixStack means, MatrixStack stds) {
		Matrix[] norms = new Matrix[outputs.depth()];
		Matrix[] adjustedNorms = new Matrix[outputs.depth()];
		for (int d = 0; d < outputs.depth(); d++) {
			Matrix[] normArray = norms(outputs.get(d), means.get(d), stds.get(d), kerW(d));
			norms[d] = normArray[0];
			adjustedNorms[d] = normArray[1];
		}
		return new MatrixStack[] {new MatrixStack(norms), new MatrixStack(adjustedNorms)};
	}
	
	
	/**
	 * Calculating norm and adjusted norms.
	 * @param output output.
	 * @param mean mean.
	 * @param std standard deviations.
	 * @return norm and adjusted norm.
	 */
	private Matrix[] norms(Matrix output, Matrix mean, Matrix std) {
		MatrixStack outputs = output instanceof MatrixStack ? (MatrixStack)output : new MatrixStack(output);
		MatrixStack means = mean instanceof MatrixStack ? (MatrixStack)mean : new MatrixStack(mean);
		MatrixStack stds = std instanceof MatrixStack ? (MatrixStack)std : new MatrixStack(std);
		MatrixStack[] norms = norms(outputs, means, stds);
		Matrix[] result = new Matrix[norms.length];
		for (int i = 0; i < norms.length; i++) {
			result[i] = norms[i] instanceof MatrixStack && ((MatrixStack)norms[i]).depth() == 1 ? ((MatrixStack)norms[i]).get(0) : norms[i];
		}
		return result;
	}
	
	
	/**
	 * Calculate means, standard deviations, and norms.
	 * @param batchOutputStacks batch of outputs.
	 * @param outputs outputs.
	 * @param W current weight matrix.
	 * @return means, standard deviations, and norms.
	 */
	@Deprecated
	private MatrixStack[] meanStdNorms(MatrixStack[] batchOutputStacks, MatrixStack outputs) {
		Matrix[] means = new Matrix[outputs.depth()];
		Matrix[] stds = new Matrix[outputs.depth()];
		Matrix[] norms = new Matrix[outputs.depth()];
		Matrix[] adjustedNorms = new Matrix[outputs.depth()];
		for (int d = 0; d < outputs.depth(); d++) {
			Matrix[] batchOutputs = batchOutputStacks != null && batchOutputStacks.length > 0 ? new Matrix[batchOutputStacks.length] : null;
			if (batchOutputs != null) for (int i = 0; i < batchOutputStacks.length; i++) batchOutputs[i] = batchOutputStacks[i].get(d);
			Matrix[] meanStdNorms = meanStdNorms(batchOutputs, outputs.get(d), kerW(d));
			means[d] = meanStdNorms[0];
			stds[d] = meanStdNorms[1];
			norms[d] = meanStdNorms[2];
			adjustedNorms[d] = meanStdNorms[3];
		}
		return new MatrixStack[] {new MatrixStack(means), new MatrixStack(stds), new MatrixStack(norms), new MatrixStack(adjustedNorms)};
	}

	
	/**
	 * Calculate means, standard deviations, and norms.
	 * @param output output.
	 * @return array of means, standard deviations, and norms.
	 */
	@SuppressWarnings("unused")
	@Deprecated
	private Matrix[] meanStdNorms(Matrix output) {
		MatrixStack[] meanStdNorms = meanStdNorms(getBatchOutputs(), output instanceof MatrixStack ? (MatrixStack)output : new MatrixStack(output));
		Matrix[] result = new Matrix[meanStdNorms.length];
		for (int i = 0; i < meanStdNorms.length; i++) {
			result[i] = meanStdNorms[i] instanceof MatrixStack && ((MatrixStack)meanStdNorms[i]).depth() == 1 ? ((MatrixStack)meanStdNorms[i]).get(0) : meanStdNorms[i];
		}
		return result;
	}
	
	
	/**
	 * Getting bias.
	 * @return bias.
	 */
	private Matrix obtainBias() {
		Matrix kerBias = this.kerBias() != null ? (this.kerBias().depth() == 1 ? this.kerBias().get(0) : this.kerBias()) : null;
		Matrix bias0 = null;
		if (kerBias != null && this.bias != null)
			bias0 = Kernel.GLOBAL_BIAS ? kerBias.add(this.bias) : kerBias;
		else if (kerBias != null)
			bias0 = kerBias;
		else if (this.bias != null)
			bias0 = this.bias;
		return bias0;
	}
	
	
	@Override
	public Matrix evaluate(Object... params) {
		if (this.kernel == null || this.prevLayer == null) throw new IllegalArgumentException();
		Matrix prevLayerOutput = this.prevLayer.queryOutput();
		if (prevLayerOutput.rows() != this.output.rows() || prevLayerOutput.columns() != this.output.columns() || MatrixUtil.depth(prevLayerOutput) != MatrixUtil.depth(this.output)) throw new IllegalArgumentException();

		if (Error.extractTrainingFlag(params)) {
			//Storing batch.
			this.batchOutputs.add(prevLayerOutput);
		}
		else {
			if (this.batchOutputs.size() > 0 || this.batchErrors.size() > 0) throw new IllegalArgumentException();
		}
		
		//Setting input and output.
		Matrix[] norms = norms(prevLayerOutput, this.runningMean, this.runningStd);
		this.prevOutput = norms[0]; //Norm.
		this.input = norms[1]; //Adjusted norm.
		
		//Adding bias.
		Matrix bias0 = obtainBias();
		if (bias0 != null) NormWeight.addBias(MatrixUtil.split(this.input), bias0);
		this.output = (this.getWeightActivateRef() != null) && !(this.weight instanceof NullWeight) ?
			this.input.evaluate0(this.getWeightActivateRef()) : this.input;
		
		Error.addLayerOInput2(this, params);
		return this.output;
	}


	/**
	 * Evaluating each member of batch.
	 * @param each each member of batch.
	 * @param mean mean.
	 * @param std standard deviation.
	 * @return evaluated output.
	 */
	private Matrix evaluateTrainingEach(Matrix each, Matrix mean, Matrix std) {
		if (this.batchOutputs.size() == 0 || this.batchErrors.size() == 0 || this.batchOutputs.size() != this.batchErrors.size()) throw new IllegalArgumentException();

		//Setting input and output.
		Matrix[] norms = norms(each, mean, std);
		this.prevOutput = norms[0]; //Norm.
		this.input = norms[1]; //Adjusted norm.
		
		//Adding bias.
		Matrix bias0 = obtainBias();
		if (bias0 != null) NormWeight.addBias(MatrixUtil.split(this.input), bias0);
		this.output = (this.getWeightActivateRef() != null) && !(this.weight instanceof NullWeight) ?
			this.input.evaluate0(this.getWeightActivateRef()) : this.input;
		return this.output;
	}
	
	
	/**
	 * Calculate gradient. Please pay attention to this method.
	 * @param outputError error.
	 * @param w current weight matrix.
	 * @param std standard deviation.
	 * @param norm norm.
	 * @return gradient.
	 */
	private static Matrix dValue(Matrix outputError, NeuronValue w, Matrix std, Matrix norm, int batchSize) {
		NeuronValue zero = outputError.get(0, 0).zero();
		NeuronValue errorSum = zero, normErrorSum = zero;
		int rows = outputError.rows(), columns = outputError.columns();
		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < columns; column++) {
				NeuronValue error = outputError.get(row, column).multiply(w);
				errorSum = errorSum.add(error);
				NeuronValue normError = error.multiply(norm.get(row, column));
				normErrorSum = normErrorSum.add(normError);
			}
		}
		
		int N = batchSize;
		Matrix factors = std.multiply0(N);
		Matrix dValue = outputError.create(new Size(columns, rows));
		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < columns; column++) {
				NeuronValue error = outputError.get(row, column).multiply(w);
				NeuronValue bias = error.multiply(N)
					.subtract(errorSum)
					.subtract(norm.get(row, column).multiply(normErrorSum))
					.divide(factors.get(row, column));
				dValue.set(row, column, bias);
			}
		}
		
		return dValue;
	}

	
	/**
	 * Calculate gradient.
	 * @param errors errors.
	 * @param stds standard deviations.
	 * @param norms norms.
	 * @return gradient.
	 */
	private MatrixStack dValue(MatrixStack errors, MatrixStack stds, MatrixStack norms, int batchSize) {
		Matrix[] dValues = new Matrix[errors.depth()];
		for (int d = 0; d < errors.depth(); d++) {
			dValues[d] = dValue(errors.get(d), kerW(d), stds.get(d), norms.get(d), batchSize);
		}
		return new MatrixStack(dValues);
	}

	
	/**
	 * Calculate gradient. Please pay attention to this method.
	 * @param batchOutputs batch of current outputs.
	 * @param outputError error.
	 * @param w current weight matrix.
	 * @return gradient.
	 */
	@Deprecated
	private static Matrix dValue(Matrix[] batchOutputs, Matrix outputError, NeuronValue w) {
		Matrix[] meanStdNorms = meanStdNorms(batchOutputs, outputError, w);
		Matrix std = meanStdNorms[1];
		Matrix norm = meanStdNorms[2];
		return dValue(outputError, w, std, norm, batchOutputs.length);
	}
	
	
	/**
	 * Calculate gradient.
	 * @param batchOutputStacks batch of outputs.
	 * @param errors errors.
	 * @return gradient.
	 */
	@Deprecated
	private MatrixStack dValue(MatrixStack[] batchOutputStacks, MatrixStack errors) {
		Matrix[] dValues = new Matrix[errors.depth()];
		for (int d = 0; d < errors.depth(); d++) {
			Matrix[] batchOutputs = batchOutputStacks.length > 0 ? new Matrix[batchOutputStacks.length] : null;
			for (int i = 0; i < batchOutputStacks.length; i++) batchOutputs[i] = batchOutputStacks[i].get(d);
			dValues[d] = dValue(batchOutputs, errors.get(d), kerW(d));
		}
		return new MatrixStack(dValues);
	}

	
	/**
	 * Calculate gradient of previous layer.
	 * @param prevLayerOutput outputs of previous layer.
	 * @param thisError current errors.
	 * @return gradient of previous layer.
	 */
	@SuppressWarnings("unused")
	@Deprecated
	private Matrix dValue(Matrix thisError) {
		if (kerW().rows() != 1 || kerW().columns() != 1 || MatrixUtil.depth(thisError) != kerW().depth()) throw new IllegalArgumentException();
		MatrixStack[] batchOutputs = getBatchOutputs();
		if (batchOutputs[0].rows() != thisError.rows() || batchOutputs[0].columns() != thisError.columns() || MatrixUtil.depth(batchOutputs[0]) != MatrixUtil.depth(thisError)) throw new IllegalArgumentException();
		
		MatrixStack thisErrors = thisError instanceof MatrixStack ? (MatrixStack)thisError : new MatrixStack(thisError);
		MatrixStack dValue = dValue(batchOutputs, thisErrors);
		return dValue.depth() == 1 ? dValue.get() : dValue;
	}

	
	/**
	 * Calculate gradient of previous layer.
	 * @param thisError current errors.
	 * @param std standard deviation.
	 * @param norm norm.
	 * @return gradient of previous layer.
	 */
	Matrix dValue(Matrix thisError, Matrix std, Matrix norm, int batchSize) {
		if (kerW().rows() != 1 || kerW().columns() != 1 || MatrixUtil.depth(thisError) != kerW().depth()) throw new IllegalArgumentException();
		MatrixStack[] batchOutputs = getBatchOutputs();
		if (batchOutputs[0].rows() != thisError.rows() || batchOutputs[0].columns() != thisError.columns() || MatrixUtil.depth(batchOutputs[0]) != MatrixUtil.depth(thisError)) throw new IllegalArgumentException();
		
		MatrixStack thisErrors = thisError instanceof MatrixStack ? (MatrixStack)thisError : new MatrixStack(thisError);
		MatrixStack stds = std instanceof MatrixStack ? (MatrixStack)std : new MatrixStack(std);
		MatrixStack norms = norm instanceof MatrixStack ? (MatrixStack)norm : new MatrixStack(norm);
		MatrixStack dValue = dValue(thisErrors, stds, norms, batchSize);
		return dValue.depth() == 1 ? dValue.get() : dValue;
	}

	
	/**
	 * Calculating gradient of the current weight.
	 * @param prevOutput previous output. Previous output and this error are in the same current layer.
	 * @param thisError current error.
	 * @return gradient of the current weight (at this error).
	 */
	WKernel dKernel(Matrix prevOutput, Matrix thisError) {
		if (kerW().rows() != 1 || kerW().columns() != 1 || MatrixUtil.depth(prevOutput) != kerW().depth()) throw new IllegalArgumentException();
		if (thisError.rows() != prevOutput.rows() || thisError.columns() != prevOutput.columns() || MatrixUtil.depth(thisError) != kerW().depth()) throw new IllegalArgumentException();
		if (this.kerBias() != null) {
			if (this.kerBias().rows() != kerW().rows() || this.kerBias().columns() != kerW().columns() || MatrixUtil.depth(this.kerBias()) != kerW().depth()) throw new IllegalArgumentException();
		}
		assert (this.kerBias() != null);

		MatrixStack prevOutputs = prevOutput instanceof MatrixStack ? (MatrixStack)prevOutput : new MatrixStack(prevOutput);
		MatrixStack thisErrors = thisError instanceof MatrixStack ? (MatrixStack)thisError : new MatrixStack(thisError);
		MatrixStack dWStack = (MatrixStack)prevOutputs.multiplyWise(thisErrors);
		MatrixStack dBiasStack = thisErrors;
		
		Matrix[] dWs = new Matrix[dWStack.depth()];
		Matrix[] dBiases = new Matrix[dBiasStack.depth()];
		for (int d = 0; d < dWStack.depth(); d++) {
			dWs[d] = dWStack.get(d).create(Size.unit());
			dWs[d].set(0, 0, Matrix.valueSum(dWStack.get(d)));
			
			dBiases[d] = dBiasStack.get(d).create(Size.unit());
			dBiases[d].set(0, 0, Matrix.valueSum(dBiasStack.get(d)));
		}
		
		WKernel dKernel = new WKernel(new MatrixStack(dWs),
			this.kerBias() != null ? new MatrixStack(dBiases) : null);
		if (this.kernel != null) dKernel.setOptimizer(this.kernel.getOptimizer());
		return dKernel;
	}

	
	@Override
	protected void browseErrors(Matrix[] errors, Error[] ERRORs, boolean learning, double learningRate) {
		super.browseErrors(errors, ERRORs, learning, learningRate);
		assert (errors.length == 1); //Please remove this code line in next version.
		this.batchErrors.addAll(Arrays.asList(errors));
	}


	@Override
	protected void backwardPost(Error[] outputErrors, double learningRate) {
		super.backwardPost(outputErrors, learningRate);
		if (this.prevLayer == null) return;
		if (outputErrors.length != this.batchOutputs.size() || outputErrors.length != this.batchErrors.size()) throw new IllegalArgumentException();
		
		Matrix[] meanStds = meanStds();
		Matrix mean = meanStds[0], std = meanStds[1];
		this.runningMean = this.runningMean.multiply0(MOMENTUM).add(mean.multiply0(1.0-MOMENTUM));
		this.runningStd = this.runningStd.multiply0(MOMENTUM).add(std.multiply0(1.0-MOMENTUM));
		
		this.dWKernelNormAccum = null;
		for (int i = 0; i < outputErrors.length; i++) {
			evaluateTrainingEach(this.batchOutputs.get(i), mean, std);
			
			LayerInput layerInput = outputErrors[i].layerOInput(this);
			layerInput.oinputPrevPrev = this.prevLayer.queryOutput();
			layerInput.ooutputPrev = getPrevOutput();
			layerInput.oinputActual = layerInput.oinput = getInput();
			layerInput.ooutput = getOutput();

			//Calculating kernel gradient.
			Matrix error = layerInput.backwardError.error();
			assert (error == this.batchErrors.get(i));
			WKernel dWKernel = dKernel(getPrevOutput(), error);
			this.dWKernelNormAccum = this.dWKernelNormAccum != null ? (WKernel)this.dWKernelNormAccum.add(dWKernel) : dWKernel; 

			//Calculating value gradient.
			Matrix dValue = dValue(error, std, getPrevOutput(), this.batchOutputs.size());
			outputErrors[i].errorSet(dValue);
		}
		WKernel dWKernelMean0 = (WKernel)this.dWKernelNormAccum.divide(outputErrors.length);
		this.kernel = accumKernel(dWKernelMean0, learningRate, decay(learningRate, outputErrors.length));
		this.dWKernelNormAccum = null;
		
		this.batchOutputs.clear();
		this.batchErrors.clear();
	}


}
