/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import net.ea.ann.core.Id;
import net.ea.ann.core.Network;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.Weight.Kernel;
import net.ea.ann.mane.filter.Filter;
import net.ea.ann.mane.filter.FilterSpec;
import net.ea.ann.mane.filter.KernelFilter;
import net.ea.ann.mane.filter.PoolFilter;
import net.ea.ann.mane.filter.ProductFilter;
import net.ea.ann.raster.Size;

/**
 * This class implements layer in matrix neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MatrixLayerImpl extends MatrixLayerAbstract {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Parametric weight.
	 */
	protected Weight weight = null;
	
	
	/**
	 * Bias.
	 */
	protected Matrix bias = null;

	
	/**
	 * Previous input value.
	 */
	protected Matrix prevInput = null;
	
	
	/**
	 * Previous output value.
	 */
	protected Matrix prevOutput = null;

	
	/**
	 * Input value.
	 */
	protected Matrix input = null;
	
	
	/**
	 * Output value.
	 */
	protected Matrix output = null;
			
	
	/**
	 * Convolutional filter.
	 */
	protected Filter filter = null;
	
	
	/**
	 * Convolutional filter bias.
	 */
	protected NeuronValue filterBias = null;
	
	
	/**
	 * Backward information: Accumulation of gradients of weights.
	 */
	private Kernel dWKernelAccum = null;
	
	
	/**
	 * Backward information: Accumulation of gradients of biases.
	 */
	private Matrix dWBiasAccum = null;
	
	
	/**
	 * Backward information: Accumulation of gradients of filter kernels.
	 */
	private net.ea.ann.mane.filter.KernelFilter.Kernel dFilterKernelAccum = null;
	
	
	/**
	 * Backward information: Accumulation of gradients of filter biases.
	 */
	private NeuronValue dFilterBiasAccum = null;
	
	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public MatrixLayerImpl(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}
	

	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public MatrixLayerImpl(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public MatrixLayerImpl(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public MatrixLayerImpl(int neuronChannel) {
		this(neuronChannel, null, null, null);
	}

	
	/**
	 * Resetting layer.
	 */
	public void reset() {
		this.prevInput = this.prevOutput = null;
		this.input = this.output = null;
		this.weight = null;
		this.bias = null;
		this.filter = null;
		this.filterBias = null;
		this.setPrevLayer(null);
		this.setNextLayer(null);
		
		resetBackwardInfo();
	}
	
	
	/**
	 * Resetting backward information.
	 */
	void resetBackwardInfo() {
		this.dWKernelAccum = null;
		this.dWBiasAccum = null;
		this.dFilterKernelAccum = null;
		this.dFilterBiasAccum = null;
	}
	
	
	/**
	 * Initializing layer with size, previous layer size, and filter.
	 * @param size this size.
	 * @param prevSize previous layer size. It can be null.
	 * @param filterSpec filter specification.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size, Size prevSize, FilterSpec filterSpec) {
		if (size == null || size.height <= 0 || size.width <= 0) return false;
		this.prevInput = this.prevOutput = null;
		this.input = this.output = null;
		this.weight = null;
		this.bias = null;
		this.filter = null;
		this.filterBias = null;
		this.setPrevLayer(null);
		this.setNextLayer(null);

		//Initialize filter and weights.
		if (prevSize != null) {
			if (prevSize.height <= 0 || prevSize.width <= 0) return false;
			Filter filter = null;
			if (filterSpec != null) {
				filter = newFilter(new Size(filterSpec.width(), filterSpec.height(), prevSize.depth, size.depth), filterSpec);
			}
			
			Weight weight = null;
			if (prevSize.height == size.height && prevSize.width == size.width) {
				weight = newWeight(
					new Size(size.height, size.height, prevSize.depth, size.depth),
					null);
			}
			else if (prevSize.height != size.height && prevSize.width == size.width) {
				weight = newWeight(
					new Size(prevSize.height, size.height, prevSize.depth, size.depth),
					null);
			}
			else if (prevSize.height == size.height && prevSize.width != size.width) {
				weight = newWeight(
					null,
					new Size(size.width, prevSize.width, prevSize.depth, size.depth));
			}
			else {
				weight = newWeight(
					new Size(prevSize.height, size.height, prevSize.depth, size.depth),
					new Size(size.width, prevSize.width, prevSize.depth, size.depth));
			}
			
			this.filter = filter;
			this.weight = weight;
			if (filterSpec != null && !filterSpec.coweight && this.filter != null) this.weight = null;
		}
		else {
			//Do nothing because the input layer has no parameters.
		}
		
		//Initialize ones related to filter.
		if (this.filter != null) {
			this.filterBias = newNeuronValue();
			this.prevInput = newMatrix(new Size(size.width, size.height, size.depth));
			this.prevOutput = newMatrix(new Size(size.width, size.height, size.depth));
		}
		
		//Initialize ones related to weights.
		if (this.weight != null) {
			this.bias = newMatrix(new Size(size.width, size.height, size.depth));
			this.input = newMatrix(new Size(size.width, size.height, size.depth));
			this.output = newMatrix(new Size(size.width, size.height, size.depth));
		}
		else if (this.filter == null) {
			this.output = this.input = newMatrix(new Size(size.width, size.height, size.depth)); //Only for input layer where both filter and weights are null and so, its input and output must be initialized.
		}
		
		return true;
	}
	
	
	/**
	 * Initializing layer with size, previous layer, and filter.
	 * @param size this size.
	 * @param prevSize previous size.
	 * @param prevLayer previous layer. It can be null.
	 * @param filter filter flag.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size, Size prevSize, MatrixLayerAbstract prevLayer, FilterSpec filter) {
		if (prevLayer == null) return initialize(size, (Size)null, filter);
		if (prevSize == null) {
			Matrix prevInput = prevLayer.queryOutput();
			if (prevInput == null) return false;
			prevSize = new Size(prevInput.columns(), prevInput.rows(), Matrix.depth(prevInput));
		}
		if (!initialize(size, prevSize, filter)) return false;
		
		//Connecting two layers.
		this.setPrevLayer(prevLayer);
		
		return true;
	}
	
	
	@Override
	protected Matrix getPrevInput() {
		return this.prevInput;
	}


	@Override
	protected void setPrevInput(Matrix prevInput) {
		this.prevInput = prevInput;
	}


	@Override
	protected Matrix getPrevOutput() {
		return this.prevOutput;
	}


	@Override
	protected void setPrevOutput(Matrix prevOutput) {
		this.prevOutput = prevOutput;
	}


	@Override
	public Matrix getInput() {
		return input;
	}

	
	@Override
	protected void setInput(Matrix input) {
		this.input = input;
	}

	
	@Override
	public Matrix getOutput() {
		return output;
	}

	
	@Override
	protected void setOutput(Matrix output) {
		this.output = output;
	}

	
	@Override
	protected Matrix getBias() {
		return bias;
	}


	@Override
	protected void setBias(Matrix bias) {
		this.bias = bias;
	}


	@Override
	protected Weight getWeight() {
		return weight;
	}

	
	@Override
	protected void setWeight(Weight weight) {
		this.weight = weight;
	}


	@Override
	protected NeuronValue getFilterBias() {
		return filterBias;
	}


	@Override
	protected void setFilterBias(NeuronValue filterBias) {
		this.filterBias = filterBias;
	}


	@Override
	protected boolean removeWeights(FilterSpec filterSpec) {
		if (this.weight == null || this.filter != null || this.prevLayer == null) return false;
		Matrix prevLayerOutput = this.prevLayer.queryOutput();
		if (prevLayerOutput == null) return false;
		Matrix output = this.queryOutput();
		if (output == null) return false;

		Size prevSize = new Size(prevLayerOutput.columns(), prevLayerOutput.rows(), Matrix.depth(prevLayerOutput));
		Size size = new Size(output.columns(), output.rows(), Matrix.depth(output));
		if (size.height <= 0 || size.width <= 0) return false;
		if (prevSize.height <= 0 || prevSize.width <= 0) return false;
		int filterHeight = prevSize.height/size.height;
		int filterWidth = prevSize.width/size.width;
		if (filterHeight == 0 || filterWidth == 0) return false;
		Size filterSize = new Size(filterWidth, filterHeight, prevSize.depth, Matrix.depth(output));
		
		this.weight = null;
		this.bias = this.input = this.output = null;
		
		this.filter = null;
		this.filterBias = null;
		this.prevInput = this.prevOutput = null;
		this.filter = getNetwork().newFilter(filterSize, filterSpec);
		this.filterBias = newNeuronValue();
		this.prevInput = newMatrix(new Size(size.width, size.height, size.depth));
		this.prevOutput = newMatrix(new Size(size.width, size.height, size.depth));
		
		return true;
	}
	
	
	@Override
	protected Filter getFilter() {
		return filter;
	}


	@Override
	protected void setFilter(Filter filter) {
		this.filter = filter;
	}


	@Override
	protected boolean removeFilter() {
		if (this.filter == null || this.prevLayer == null) return false;
		Matrix prevLayerOutput = this.prevLayer.queryOutput();
		if (prevLayerOutput == null) return false;
		Matrix output = this.queryOutput();
		if (output == null) return false;
		
		Size prevSize = new Size(prevLayerOutput.columns(), prevLayerOutput.rows(), Matrix.depth(prevLayerOutput));
		Size size = new Size(output.columns(), output.rows(), Matrix.depth(output));
		if (size.height <= 0 || size.width <= 0) return false;
		if (prevSize.height <= 0 || prevSize.width <= 0) return false;
		
		if (prevSize.height == size.height && prevSize.width == size.width) {
			this.weight = newWeight(
				new Size(size.height, size.height, prevSize.depth, size.depth),
				null);
		}
		else if (prevSize.height != size.height && prevSize.width == size.width) {
			this.weight = newWeight(
				new Size(prevSize.height, size.height, prevSize.depth, size.depth),
				null);
		}
		else if (prevSize.height == size.height && prevSize.width != size.width) {
			this.weight = newWeight(
				null,
				new Size(size.width, prevSize.width, prevSize.depth, size.depth));
		}
		else {
			this.weight = newWeight(
				new Size(prevSize.height, size.height, prevSize.depth, size.depth),
				new Size(size.width, prevSize.width, prevSize.depth, size.depth));
		}
		
		this.filter = null;
		this.filterBias = null;
		this.prevInput = this.prevOutput = null;

		this.bias = this.input = this.output = null;
		this.bias = newMatrix(new Size(size.width, size.height, size.depth));
		this.input = newMatrix(new Size(size.width, size.height, size.depth));
		this.output = newMatrix(new Size(size.width, size.height, size.depth));
		
		return true;
	}
	
	
	/**
	 * Calculating gradient of filter value.
	 * @param error current error.
	 * @return gradient of filter value.
	 */
	private Matrix dFilterValue(Matrix error) {
		if (this.filter == null) return null;
		Matrix thisPrevInputConv = matrixToConvLayer(getPrevInput());
		Matrix prevLayerOutputConv = this.prevLayer.matrixToConvLayer(this.prevLayer.queryOutput());
		Matrix errorConv = matrixToConvLayer(error);
		
		Matrix dValues = null;
		if (this.filter instanceof KernelFilter) {
			dValues = ((KernelFilter)this.filter).dValue(prevLayerOutputConv, thisPrevInputConv, errorConv, this.convActivateRef);
		}
		else if (this.filter instanceof PoolFilter) {
			dValues = ((PoolFilter)this.filter).dValue(prevLayerOutputConv, thisPrevInputConv, errorConv);
		}
		return dValues != null ? this.prevLayer.convLayerToMatrix(dValues) : null;
	}
	
	
	/**
	 * Calculating gradient of filter kernel.
	 * @param error current error.
	 * @return gradient of filter kernel.
	 */
	private net.ea.ann.mane.filter.KernelFilter.Kernel dFilterKernel(Matrix error) {
		if ((this.filter == null) || !(this.filter instanceof KernelFilter)) return null;
		Matrix prevLayerOutputConv = this.prevLayer.matrixToConvLayer(this.prevLayer.queryOutput());
		Matrix thisPrevInputConv = matrixToConvLayer(getPrevInput());
		Matrix errorConv = matrixToConvLayer(error);
		
		return ((KernelFilter)this.filter).dKernel(prevLayerOutputConv, thisPrevInputConv, errorConv, this.convActivateRef);
	}

	
	/**
	 * Accumulating filter kernel.
	 * @param dFilterKernel filter kernel gradient.
	 * @param learningRate learning rate.
	 */
	private void accumFilterKernel(net.ea.ann.mane.filter.KernelFilter.Kernel dFilterKernel, double learningRate) {
		if ((this.filter == null) || !(this.filter instanceof KernelFilter)) return;
		((KernelFilter)this.filter).accumKernel(dFilterKernel, learningRate);
	}
	
	
	/**
	 * Evaluating by filtering.
	 * @return filtered matrix.
	 */
	private Matrix evaluateByFilter() {
		if (this.filter == null || this.prevLayer == null) return null;
		Matrix prevLayerOutput = this.prevLayer.queryOutput();
		if (prevLayerOutput == null) return null;
		
		if (this.filter instanceof ProductFilter) {
			ProductFilter filter = (ProductFilter)this.filter;
			Matrix prevLayerOutputConv = this.prevLayer.matrixToConvLayer(prevLayerOutput);
			Matrix thisPrevInputConv = matrixToConvLayer(this.prevInput);
			Matrix thisInputConv = this.input != null ? matrixToConvLayer(this.input) :
				thisPrevInputConv.create(new Size(thisPrevInputConv.columns(), thisPrevInputConv.rows()));
			filter.forward(prevLayerOutputConv, thisPrevInputConv, thisInputConv, this.filterBias, this.convActivateRef);
			this.prevInput = convLayerToMatrix(thisPrevInputConv);
			return this.prevOutput = convLayerToMatrix(thisInputConv);
		}
		else {
			return null;
		}
	}
	
	
	@Override
	public Matrix evaluate(Object...params) {
		if (this.prevLayer == null) return null;
		Matrix prevOutput = this.filter != null ? evaluateByFilter() : null;
		if (this.weight == null) return prevOutput;
		
		this.input = prevOutput != null ? prevOutput : this.prevLayer.queryOutput();
		this.input = this.weight.evaluate(this.input, this.bias);
		this.output = this.activateRef != null ? this.input.evaluate0(this.activateRef) : this.input;
		return this.output;
	}


	@Override
	public Matrix forward(Record...inputs) {
		Matrix input = inputs != null && inputs.length > 0 ? inputs[0].input() : null;
		if (this.prevLayer == null) {
			if (input != null) Matrix.copy(input, this.input);
			if (this.input != this.output) this.output = this.input;
			if (this.nextLayer == null) return input;
		}
		
		MatrixLayer nextLayer = null;
		Matrix result = null;
		while ((nextLayer = this.getNextLayer()) != null) {
			result = nextLayer.evaluate();
		}
		return result;
	}


	/**
	 * Learning matrix neural network layer.
	 * @param outputErrors core last errors which are core last biases.
	 * @param learningRate learning rate.
	 * @return training error.
	 */
	protected Error[] learn(Error[] outputErrors, double learningRate) {
		 return backward(outputErrors, this, true, learningRate);
	}

	
	@Override
	public Error[] backward(Error[] outputErrors, MatrixLayer focus, boolean learning, double learningRate) {
		if (outputErrors == null || outputErrors.length == 0) return null;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? Network.LEARN_RATE_DEFAULT : learningRate;
		if (focus == null) learning = true;
		
		Matrix[] errors = new Matrix[outputErrors.length];
		Kernel[] dWKernels = new Kernel[outputErrors.length];
		NeuronValue[] dFilterBiases = new NeuronValue[outputErrors.length];
		net.ea.ann.mane.filter.KernelFilter.Kernel[] dFilterKernels = new net.ea.ann.mane.filter.KernelFilter.Kernel[outputErrors.length];

		//Browsing errors.
		for (int i = 0; i < outputErrors.length; i++) {
			//Calculating value errors from next layer.
			if (this.nextLayer == null) {
				errors[i] = outputErrors[i].error(); //Getting errors from environment.
			}
			else if (this.nextLayer.getFilter() == null) {
				Matrix input = queryInput(), output = queryOutput(); //X^k-1 = input, Xk = output.
				Function thisActivateRef = input == getInput() ? this.activateRef : (input == getPrevInput() ? this.convActivateRef : null); //Getting right-most activation function.
				errors[i] = this.nextLayer.getWeight().dValue(input, output, outputErrors[i].error(), thisActivateRef);
			}
			else {
				errors[i] = outputErrors[i].error(); //Getting errors from next layer.
			}
				
			//Calculating weight gradient.
			if (this.weight != null) {
				Matrix prevOutput = getPrevOutput();
				prevOutput = prevOutput != null ? prevOutput : this.prevLayer.queryOutput(); //Xk-1
				dWKernels[i] = this.weight.dKernel(prevOutput, errors[i]);
			}

			//Calculating filter gradient.
			if (this.filter != null) {
				if (this.weight != null) {
					//Calculating value errors at this layer.
					Matrix input = getPrevInput(), output = getPrevOutput(); //X^k-1 = input, Xk = output.
					errors[i] = this.weight.dValue(input, output, errors[i], this.convActivateRef);
				}
				dFilterBiases[i] = Filter.CALC_ERROR_MEAN ? Matrix.valueMean(errors[i]) : Matrix.valueSum(errors[i]); //Filter errors.
				dFilterKernels[i] = dFilterKernel(errors[i]);
				outputErrors[i].errorSet(dFilterValue(errors[i])); //Please pay attention to this code line to assign current errors to output errors.
			}
			else {
				outputErrors[i].errorSet(errors[i]); //Please pay attention to this code line to assign current errors to output errors.
			}
		} //End browsing errors.
		
		//Update weight bias, first weight, and second weight.
		if (this.bias != null) {
			Matrix dBiasMean = Matrix.mean(errors);
			if (learning) {
				Matrix bias = this.bias.add(dBiasMean.multiply0(learningRate));
				this.setBias(bias);
			}
			else
				this.dWBiasAccum = this.dWBiasAccum != null ? this.dWBiasAccum.add(dBiasMean) : dBiasMean;
		}
		if (this.weight != null) {
			Kernel dWKernelMean = Kernel.mean(dWKernels);
			if (learning)
				this.weight.accumKernel(dWKernelMean, learningRate);
			else 
				this.dWKernelAccum = this.dWKernelAccum != null ? this.dWKernelAccum.add(dWKernelMean) : dWKernelMean;
		}
		
		//Update filter and filter bias.
		if (this.filter != null && isLearnFilter()) {
			NeuronValue dFilterBiasMean = NeuronValue.valueMean(dFilterBiases);
			if (learning) {
				NeuronValue filterBias = this.filterBias.add(dFilterBiasMean.multiply(learningRate));
				setFilterBias(filterBias);
			}
			else
				this.dFilterBiasAccum = this.dFilterBiasAccum != null ? this.dFilterBiasAccum.add(dFilterBiasMean) : dFilterBiasMean;
			
			net.ea.ann.mane.filter.KernelFilter.Kernel dFilterKernelMean = net.ea.ann.mane.filter.KernelFilter.Kernel.mean(dFilterKernels);
			if (learning)
				accumFilterKernel(dFilterKernelMean, learningRate);
			else
				this.dFilterKernelAccum = this.dFilterKernelAccum != null ? this.dFilterKernelAccum.add(dFilterKernelMean) : dFilterKernelMean;
		}
		
		//Returning output errors if there is no previous layers or this layer is focused layer.
		if (this.prevLayer == null || this == focus) return outputErrors;
		
		errors = null;
		dWKernels = null;
		dFilterBiases = null;
		dFilterKernels = null;
		//Browsing backward layers.
		return this.prevLayer.backward(outputErrors, focus, learning, learningRate);
	}


	/**
	 * Back-warding layer as learning matrix neural network.
	 * @param outputErrors core last errors which are core last biases.
	 * @param learningRate learning rate.
	 * @return training error.
	 */
	Error[] backwardThisLayerWithoutLearning(Error[] outputErrors, double learningRate) {
		resetBackwardInfo();
		if (outputErrors == null || outputErrors.length == 0) return null;
		Error[] errors = new Error[outputErrors.length];
		for (int i = 0; i < outputErrors.length; i++) {
			errors[i] = backward(new Error[] {outputErrors[i]}, this, false, learningRate)[0];
		}
		return errors;
	}

	
	/**
	 * Updating parameters from backward information.
	 * @param recordCount count of records in sample.
	 * @param learningRate learning rate.
	 */
	void updateParametersFromBackwardInfo(int recordCount, double learningRate) {
		//Update weight bias, first weight, and second weight.
		if (this.bias != null && this.dWBiasAccum != null) {
			Matrix dBiasMean = this.dWBiasAccum.divide0(recordCount);
			Matrix bias = this.getBias().add(dBiasMean.multiply0(learningRate));
			this.setBias(bias);
			this.dWBiasAccum = null;
		}
		if (this.weight != null && this.dWKernelAccum != null) {
			Kernel dWMean = this.dWKernelAccum.divide(recordCount);
			this.weight.accumKernel(dWMean, learningRate);
			this.dWKernelAccum = null;
		}

		//Update filter and filter bias.
		if (this.filter != null && this.isLearnFilter()) {
			if (this.dFilterBiasAccum != null) {
				NeuronValue dFilterBiasMean = this.dFilterBiasAccum.divide(recordCount);
				NeuronValue filterBias = this.filterBias.add(dFilterBiasMean.multiply(learningRate));
				this.setFilterBias(filterBias);
				this.dFilterBiasAccum = null;
			}
			
			if (this.dFilterKernelAccum != null) {
				net.ea.ann.mane.filter.KernelFilter.Kernel dFilterKernelMean = this.dFilterKernelAccum.divide(recordCount);
				accumFilterKernel(dFilterKernelMean, learningRate);
				this.dFilterKernelAccum = null;
			}
		}
		
		resetBackwardInfo();
	}
	
	
}
