/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.util.List;

import net.ea.ann.core.Id;
import net.ea.ann.core.Network;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.function.IdentityDefault;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.mane.Error.LayerInput;
import net.ea.ann.mane.filter.NetworkFilter;
import net.ea.ann.mane.weight.ActivateFWeight;
import net.ea.ann.mane.weight.NetworkWeight;
import net.ea.ann.mane.weight.NullWeight;
import net.ea.ann.raster.Size;

/**
 * This class implements layer in matrix neural network.
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
	Kernel dWKernelAccum = null;
	
	
	/**
	 * Backward information: Accumulation of gradients of biases.
	 */
	Matrix dWBiasAccum = null;
	
	
	/**
	 * Backward information: Accumulation of gradients of filter kernels.
	 */
	Kernel dFKernelAccum = null;
	
	
	/**
	 * Backward information: Accumulation of gradients of filter biases.
	 */
	NeuronValue dFBiasAccum = null;
	
	
	/**
	 * Temporal account for validating accumulation. It should be removed in improved version.
	 * Please pay attention that the accumulation only occurs in method {@link #backward(Error[], MatrixLayer, boolean, double)}.
	 */
	int tempAccum = 0;
	
	
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
	public MatrixLayerImpl(int neuronChannel) {this(neuronChannel, null, null, null);}

	
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
	protected void resetBackwardInfo() {
		this.dWKernelAccum = null;
		this.dWBiasAccum = null;
		this.dFKernelAccum = null;
		this.dFBiasAccum = null;
		
		this.tempAccum = 0;
		
		if (this.weight != null && this.weight instanceof NetworkWeight) ((NetworkWeight)this.weight).resetBackwardInfo();
		if (this.filter != null && this.filter instanceof NetworkFilter) ((NetworkFilter)this.filter).resetBackwardInfo();
	}
	
	
	/**
	 * Initializing layer with size, previous layer size, and filter.
	 * Please pay attention that filter is before weight in the same layer.
	 * @param size this size.
	 * @param prevSize previous layer size. It can be null.
	 * @param layerSpec layer specification.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size, Size prevSize, MatrixLayerAbstract.LayerSpec layerSpec) {
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
			
			FilterSpec filterSpec = layerSpec != null ? layerSpec.filterSpec : null;
			if (filterSpec != null) {
				this.filter = newFilter(new Size(filterSpec.width(), filterSpec.height(), prevSize.depth, size.depth), layerSpec);
				if (filterSpec.coweight && layerSpec.weightSpec != null)
					this.weight = newWeight(size, size, layerSpec);
			}
			else
				this.weight = newWeight(prevSize, size, layerSpec);
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
		
		/*
		//Setting identity activation function for null filter and network filter.
		if ((this.filter != null) && (this.filter instanceof NullFilter || this.filter instanceof NetworkFilter))
			this.setFilterActivateRef(IdentityDefault.identity());
		*/
		//Setting identity activation function for null weight and network weight.
		if ((this.weight != null) && (this.weight instanceof NullWeight || this.weight instanceof NetworkWeight))
			this.setWeightActivateRef(IdentityDefault.identity());
		//Setting filter activation function for filter activate weight.
		if (this.weight != null && this.weight instanceof ActivateFWeight && this.getFilterActivateRef() != null)
			this.setWeightActivateRef(this.getFilterActivateRef());
		
		resetBackwardInfo(); //Fixing date: 2026.06.05
		return true;
	}
	
	
	/**
	 * Initializing layer with size, previous layer, and filter.
	 * @param size this size.
	 * @param prevSize previous size.
	 * @param prevLayer previous layer. It can be null.
	 * @param layerSpec layer specification.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size, Size prevSize, MatrixLayerAbstract prevLayer, MatrixLayerAbstract.LayerSpec layerSpec) {
		if (prevLayer == null) return initialize(size, (Size)null, layerSpec);
		if (prevSize == null) {
			Matrix prevInput = prevLayer.queryOutput();
			if (prevInput == null) return false;
			prevSize = new Size(prevInput.columns(), prevInput.rows(), MatrixUtil.depth(prevInput));
		}
		if (!initialize(size, prevSize, layerSpec)) return false;
		
		//Connecting two layers.
		this.setPrevLayer(prevLayer);
		
		return true;
	}
	
	
	@Override
	public Matrix getPrevInput() {return this.prevInput;}


	@Override
	protected void setPrevInput(Matrix prevInput) {this.prevInput = prevInput;}


	@Override
	public Matrix getPrevOutput() {return this.prevOutput;}


	@Override
	protected void setPrevOutput(Matrix prevOutput) {this.prevOutput = prevOutput;}


	@Override
	public Matrix getInput() {return input;}

	
	@Override
	protected void setInput(Matrix input) {this.input = input;}

	
	@Override
	public Matrix getOutput() {return output;}

	
	@Override
	protected void setOutput(Matrix output) {this.output = output;}

	
	@Override
	protected Matrix getBias() {return bias;}


	@Override
	protected void setBias(Matrix bias) {this.bias = bias;}


	@Override
	public Weight getWeight() {return weight;}

	
	@Override
	protected void setWeight(Weight weight) {this.weight = weight;}


	@Override
	protected NeuronValue getFilterBias() {return filterBias;}


	@Override
	protected void setFilterBias(NeuronValue filterBias) {this.filterBias = filterBias;}


	@Override
	protected boolean removeWeights(LayerSpec layerSpec) {
		if (this.weight == null || this.filter != null || this.prevLayer == null) return false;
		Matrix prevLayerOutput = this.prevLayer.queryOutput();
		if (prevLayerOutput == null) return false;
		Matrix output = this.queryOutput();
		if (output == null) return false;

		Size prevSize = new Size(prevLayerOutput.columns(), prevLayerOutput.rows(), MatrixUtil.depth(prevLayerOutput));
		Size size = new Size(output.columns(), output.rows(), MatrixUtil.depth(output));
		if (size.height <= 0 || size.width <= 0) return false;
		if (prevSize.height <= 0 || prevSize.width <= 0) return false;
		int filterHeight = prevSize.height/size.height;
		int filterWidth = prevSize.width/size.width;
		if (filterHeight == 0 || filterWidth == 0) return false;
		Size filterSize = new Size(filterWidth, filterHeight, prevSize.depth, MatrixUtil.depth(output));
		
		this.weight = null;
		this.bias = this.input = this.output = null;
		
		this.filter = null;
		this.filterBias = null;
		this.prevInput = this.prevOutput = null;
		this.filter = getNetwork().newFilter(filterSize, layerSpec);
		this.filterBias = newNeuronValue();
		this.prevInput = newMatrix(new Size(size.width, size.height, size.depth));
		this.prevOutput = newMatrix(new Size(size.width, size.height, size.depth));
		
		return true;
	}
	
	
	@Override
	public Filter getFilter() {return filter;}


	@Override
	protected void setFilter(Filter filter) {this.filter = filter;}


	@Override
	protected boolean removeFilter() {
		if (this.filter == null || this.prevLayer == null) return false;
		Matrix prevLayerOutput = this.prevLayer.queryOutput();
		if (prevLayerOutput == null) return false;
		Matrix output = this.queryOutput();
		if (output == null) return false;
		
		Size prevSize = new Size(prevLayerOutput.columns(), prevLayerOutput.rows(), MatrixUtil.depth(prevLayerOutput));
		Size size = new Size(output.columns(), output.rows(), MatrixUtil.depth(output));
		if (size.height <= 0 || size.width <= 0) return false;
		if (prevSize.height <= 0 || prevSize.width <= 0) return false;
		
		this.weight = newWeight(prevSize, size, null);
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
	 * @param learning learning mode.
	 * @param learningRate learning rate.
	 * @param refError error reference which can be null.
	 * @return gradient of filter value.
	 */
	Matrix dFilterValue(Matrix error, boolean learning, double learningRate, Error refError) {
		if (this.filter == null) return null;
		Matrix actualPrevLayerOutput = refError != null ? refError.oinputPrevPrevOfLayer(this) : null; //Actual previous previous input.
		actualPrevLayerOutput = actualPrevLayerOutput != null ? actualPrevLayerOutput : this.prevLayer.queryOutput();
		assert (actualPrevLayerOutput == this.prevLayer.queryOutput()); //This assertion should be removed in next version.
		Matrix prevLayerOutputConv = this.prevLayer.matrixToConvLayer(actualPrevLayerOutput);
		//
		Matrix actualPrevInput = refError != null ? refError.oinputPrevOfLayer(this) : null; //Actual previous input.
		actualPrevInput = actualPrevInput != null ? actualPrevInput : getPrevInput();
		assert (actualPrevInput == getPrevInput()); //This assertion should be removed in next version.
		Matrix thisPrevInputConv = matrixToConvLayer(actualPrevInput);
		//
		Matrix errorConv = matrixToConvLayer(error);
		
		Matrix dValue = null;
		if (this.filter instanceof NetworkFilter)
			dValue = ((NetworkFilter)this.filter).dValue(prevLayerOutputConv, thisPrevInputConv, errorConv, this.getFilterActivateRef(), learning, learningRate);
		else
			dValue = this.filter.dValue(prevLayerOutputConv, thisPrevInputConv, errorConv, this.getFilterActivateRef());
		return dValue != null ? this.prevLayer.convLayerToMatrix(dValue) : null;
	}
	
	
	/**
	 * Calculating gradient of filter kernel.
	 * @param error current error.
	 * @param refError error reference which can be null.
	 * @return gradient of filter kernel.
	 */
	Kernel dFilterKernel(Matrix error, Error refError) {
		if (this.filter == null) return null;
		Matrix actualPrevLayerOutput = refError != null ? refError.oinputPrevPrevOfLayer(this) : null; //Actual previous previous input.
		actualPrevLayerOutput = actualPrevLayerOutput != null ? actualPrevLayerOutput : this.prevLayer.queryOutput();
		assert (actualPrevLayerOutput == this.prevLayer.queryOutput()); //This assertion should be removed in next version.
		Matrix prevLayerOutputConv = this.prevLayer.matrixToConvLayer(actualPrevLayerOutput);
		//
		Matrix actualPrevInput = refError != null ? refError.oinputPrevOfLayer(this) : null; //Actual previous input.
		actualPrevInput = actualPrevInput != null ? actualPrevInput : getPrevInput();
		assert (actualPrevInput == getPrevInput()); //This assertion should be removed in next version.
		Matrix thisPrevInputConv = matrixToConvLayer(actualPrevInput);
		//
		Matrix errorConv = matrixToConvLayer(error);
		
		if (this.filter instanceof NetworkFilter)
			return null;
		else
			return this.filter.dKernel(prevLayerOutputConv, thisPrevInputConv, errorConv, this.getFilterActivateRef());
	}

	
	/**
	 * Accumulating filter kernel.
	 * @param dFKernel filter kernel gradient.
	 * @param learningRate learning rate.
	 * @param recordCount record count.
	 * @return return accumulated filter. 
	 */
	Filter accumFilterKernel(Kernel dFKernel, double learningRate, int recordCount) {
		if (this.filter == null) return this.filter;
		return this.filter.accumKernel(dFKernel, learningRate, decay(learningRate, recordCount));
	}
	
	
	/**
	 * Evaluating by filtering. This method should be private.
	 * @return filtered matrix.
	 */
	Matrix evaluateByFilter() {
		if (this.filter == null || this.prevLayer == null) return null;
		Matrix prevLayerOutputConv = this.prevLayer.matrixToConvLayer(this.prevLayer.queryOutput());
		Matrix thisPrevInputConv = matrixToConvLayer(this.prevInput);
		Matrix thisPrevOutputConv = matrixToConvLayer(this.prevOutput);
		
		this.filter.forward(prevLayerOutputConv, thisPrevInputConv, thisPrevOutputConv, this.filterBias, this.getFilterActivateRef());
		this.prevInput = convLayerToMatrix(thisPrevInputConv);
		return this.prevOutput = convLayerToMatrix(thisPrevOutputConv);
	}
	
	
	@Override
	public Matrix evaluate(Object...params) {
		assert (this.prevLayer != null); //Do not evaluate the input layer.
		if (this.prevLayer == null) return null;
		Matrix prevOutput0 = this.filter != null ? evaluateByFilter() : null;
		if (this.weight == null) {
			Error.addLayerOInput(this, params);
			assert (prevOutput0 != null && prevOutput0 == this.prevOutput);
			return prevOutput0;
		}
		
		this.input = prevOutput0 != null ? prevOutput0 : this.prevLayer.queryOutput();
		this.input = this.weight.evaluate(this.input, this.bias);
		this.output = (this.getWeightActivateRef() != null) && !(this.weight instanceof NullWeight) ?
			this.input.evaluate0(this.getWeightActivateRef()) : this.input;
		
		Error.addLayerOInput(this, params);
		return this.output;
	}
	
	
	@Override
	public Matrix forward(Record...inputs) {
		Matrix input0 = inputs != null && inputs.length > 0 ? inputs[0].input() : null;
		assert (input0 != null);
		Object[] params = inputs[0].params.toArray(new Object[] {});
		Matrix result = null;
		if (this.prevLayer == null) {
			assert (getPrevInput() == null && getPrevOutput() == null);
			if (input0 != null) MatrixUtil.copy(input0, this.input);
			assert (this.input == this.output);
			if (this.input != this.output) this.output = this.input;
			result = this.output;
		}
		else {
			if (input0 != null) MatrixUtil.copy(input0, this.queryInput());
			result = this.evaluate(params);
		}
		
		MatrixLayer nextLayer0 = null;
		while ((nextLayer0 = this.getNextLayer()) != null) {
			result = nextLayer0.evaluate(params);
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

	
	/*
	 * Please pay attention that filter is before weight in the same layer.
	 * This method is the core of matrix neural network.
	 */
	@Override
	public Error[] backward(Error[] outputErrors, MatrixLayer focus, boolean learning, double learningRate) {
		if (outputErrors == null || outputErrors.length == 0) return null;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? Network.LEARN_RATE_DEFAULT : learningRate;
		if (focus == null) learning = true;
		
		Matrix[] errors = new Matrix[outputErrors.length];
		Kernel[] dWKernels = new Kernel[outputErrors.length];
		NeuronValue[] dFBiases = new NeuronValue[outputErrors.length];
		Kernel[] dFKernels = new Kernel[outputErrors.length];

		//Browsing errors. Please pay attention that filter is before weight in the same layer.
		for (int i = 0; i < outputErrors.length; i++) {
			Matrix errPrevPrevInput = outputErrors[i].oinputPrevPrevOfLayer(this); //Previous previous input.
			Matrix errPrevInput = outputErrors[i].oinputPrevOfLayer(this); //Previous input.
			Matrix errPrevOutput = outputErrors[i].ooutputPrevOfLayer(this); //Previous output.
			Matrix actualErrInput = outputErrors[i].oinputOfLayerActual(this); //Actual input.
			Matrix actualErrOutput = outputErrors[i].ooutputOfLayer(this);
			actualErrOutput = actualErrOutput != null ? actualErrOutput : errPrevOutput; //Actual output.
			Matrix mask = outputErrors[i].oinputDropoutMaskOfLayer(this);
			if (outputErrors.length == 1) {
				if (this.prevLayer != null) assert (errPrevPrevInput == this.prevLayer.queryOutput());
				assert (errPrevInput == getPrevInput());
				assert (errPrevOutput == getPrevOutput());
				assert (actualErrInput == queryInput());
				assert (actualErrOutput == queryOutput());
				if (this instanceof DropoutLayer) assert (mask == ((DropoutLayer)this).dropoutMask);
			}
			/*
			 * These inputs and outputs associated with error are not so important for calculating gradients because the accuracy will be improved with a large enough number of batches.
			 * Moreover, keeping the current inputs and outputs in a small enough batch does not affect significantly on the accuracy in training.
			 * However the inputs and outputs associated with error make the logic calculation is more precise. 
			 */
			
			//Getting errors from environment or next layer.
			errors[i] = outputErrors[i].error(); 
			
			//Calculating errors from weight next layer.
			if (this.nextLayer != null && this.nextLayer.getFilter() == null /*&& this.nextLayer.getWeight() != null*/ && this.nextLayer.getWeight().backwardErrorMode()) {
				Function thisActivateRef = this.weight != null ? (!(this.weight instanceof NullWeight) ? this.getWeightActivateRef() : null) :
					(this.filter != null && this.filter.doesApplyActivate() && !this.filter.isIndexMode() ?
						this.getFilterActivateRef() : null); //Getting right-most activation function. Setting function of index-mode filter like max-pooling filter to be null because of the filtered result of index-mode filter is indexing matrix.
				Matrix input0 = actualErrInput != null ? actualErrInput : queryInput(); //X^k-1 = input.
				Matrix output0 = actualErrOutput != null ? actualErrOutput : queryOutput(); //Xk = output.
				assert (input0 != null && output0 != null);
				errors[i] = this.nextLayer.getWeight().dValue(input0, output0, errors[i], thisActivateRef);
			}
			
			//Applying dropout mask into errors.
			mask = mask != null ? mask : (this instanceof DropoutLayer ? ((DropoutLayer)this).dropoutMask : null);
			errors[i] = mask != null ? mask.multiplyWise(errors[i]) : errors[i];
			//Adding residual backward errors.
			if (getEndLayer() != null) {
				LayerInput layerInput = outputErrors[i].layerOInput(getEndLayer());
				Matrix endError = layerInput != null && layerInput.backwardError != null ? layerInput.backwardError.error() : null;
				if (endError != null) errors[i] = errors[i].add(endError);
			}
			//Adjusting errors.
			errors[i] = adjustError(errors[i], outputErrors[i]);

			//Calculating weight gradient.
			if (this.weight != null) {
				if (this.weight.backwardErrorMode()) {
					Matrix prevOutput0 = errPrevOutput != null ? errPrevOutput : getPrevOutput();
					prevOutput0 = prevOutput0 != null ? prevOutput0 : this.prevLayer.queryOutput();
					assert (prevOutput0 != null);
					dWKernels[i] = this.weight.dKernel(prevOutput0, errors[i]);
				}
				else {
					//Calculating value errors at this layer.
					Matrix prevInput0 = errPrevOutput != null ? errPrevOutput : (errPrevPrevInput != null ? errPrevPrevInput : getPrevOutput()); //If actualPrevOutput is null then actualPrevInput is always null.
					prevInput0 = prevInput0 != null ? prevInput0 : this.prevLayer.queryOutput(); //If the method getPrevOutput() returns null then the method getPrevInput() always returns null.
					Matrix prevOutput0 = actualErrInput != null ? actualErrInput : queryInput();
					assert (prevInput0 != null && prevOutput0 != null);
					//Variables prevInput0 and prevOutput0 are not important for network weight and transformer because they do not take derivatives on such variables when they have particular internal structures.
					if (this.weight instanceof NetworkWeight) {
						errors[i] = ((NetworkWeight)this.weight).dValue(prevInput0, prevOutput0, errors[i], this.getWeightActivateRef(), learning, learningRate);
						dWKernels[i] = null;
					}
					else {
						errors[i] = this.weight.dValue(prevInput0, prevOutput0, errors[i], this.getWeightActivateRef());
						dWKernels[i] = this.weight.dKernel(prevOutput0, errors[i]);
					}
				}
			} //Calculating weight gradient.

			//Calculating filter gradient.
			if (this.filter != null) {
				if (this.weight != null && this.weight.backwardErrorMode()) {
					//Calculating value errors at this layer.
					Function thisActivateRef = this.filter.doesApplyActivate() && !this.filter.isIndexMode() ? this.getFilterActivateRef() : null; //Setting function of index-mode filter like max-pooling filter to be null because of the filtered result of index-mode filter is indexing matrix.
					Matrix prevInput0 = errPrevInput != null ? errPrevInput : getPrevInput();
					Matrix prevOutput0 = getPrevOutput();
					assert (prevInput0 != null && prevOutput0 != null);
					errors[i] = this.weight.dValue(prevInput0, prevOutput0, errors[i], thisActivateRef);
				}
				dFBiases[i] = MatrixUtil.valueSum(errors[i]); //Filter errors.
				dFKernels[i] = dFilterKernel(errors[i], outputErrors[i]);
				outputErrors[i].errorSet(dFilterValue(errors[i], learning, learningRate, outputErrors[i])); //Please pay attention to this code line to back-warding value errors.
			}
			else {
				outputErrors[i].errorSet(errors[i]); //Please pay attention to this code line to back-warding value errors.
			} //Calculating filter gradient.
			
		} //Browsing errors.
		
		//Update weight bias, first weight, and second weight.
		if (this.bias != null) {
			Matrix dBiasMean0 = MatrixUtil.mean(errors);
			if (learning) {
				Matrix bias0 = this.bias.add(dBiasMean0.multiply0(learningRate));
				this.setBias(bias0);
			}
			else
				this.dWBiasAccum = this.dWBiasAccum != null ? this.dWBiasAccum.add(dBiasMean0) : dBiasMean0;
		}
		if (this.weight != null && dWKernels[0] != null) {
			Kernel dWKernelMean0 = Kernel.mean(dWKernels);
			if (learning) {
//				this.weight = this.weight.accumKernel(dWKernelMean0, learningRate, decay(learningRate, dWKernels.length));
				if (this.weight != this.weight.accumKernel(dWKernelMean0, learningRate, decay(learningRate, dWKernels.length))) throw new IllegalArgumentException();
			}
			else 
				this.dWKernelAccum = this.dWKernelAccum != null ? this.dWKernelAccum.add(dWKernelMean0) : dWKernelMean0;
		}
		
		//Update filter and filter bias.
		if (this.filter != null && isLearnFilter()) {
			NeuronValue dFilterBiasMean0 = NeuronValue.valueMean(dFBiases);
			if (learning) {
				NeuronValue filterBias0 = this.filterBias.add(dFilterBiasMean0.multiply(learningRate));
				setFilterBias(filterBias0);
			}
			else
				this.dFBiasAccum = this.dFBiasAccum != null ? this.dFBiasAccum.add(dFilterBiasMean0) : dFilterBiasMean0;
			
			if (dFKernels[0] != null) {
				Kernel dFilterKernelMean0 = Kernel.mean(dFKernels);
				if (learning) {
//					this.filter = accumFilterKernel(dFilterKernelMean0, learningRate, dFKernels.length);
					if (this.filter != accumFilterKernel(dFilterKernelMean0, learningRate, dFKernels.length)) throw new IllegalArgumentException();
				}
				else
					this.dFKernelAccum = this.dFKernelAccum != null ? this.dFKernelAccum.add(dFilterKernelMean0) : dFilterKernelMean0;
			}
		}
		
		this.tempAccum += outputErrors.length;
		
		//Returning output errors if there is no previous layers or this layer is focused layer.
		if (this.prevLayer == null || this == focus) return outputErrors;
		
		errors = null;
		dWKernels = null;
		dFBiases = null;
		dFKernels = null;
		//Browsing backward layers.
		return this.prevLayer.backward(outputErrors, focus, learning, learningRate);
	}


	/**
	 * Adjusting error.
	 * @param error error will be adjusted.
	 * @param ERROR referred error which can be null.
	 * @return adjusted error.
	 */
	Matrix adjustError(Matrix error, Error ERROR) {return error;}
	
	/**
	 * Back-warding layer as learning matrix neural network.
	 * @param outputErrors core last errors which are core last biases.
	 * @param learningRate learning rate.
	 * @return training error.
	 */
	Error[] backwardWithoutLearning(Error[] outputErrors, double learningRate) {
		resetBackwardInfo();
		if (outputErrors == null || outputErrors.length == 0) return null;
		List<Error> outputErrorList = Util.newList(0);
		for (int i = 0; i < outputErrors.length; i++) {
			Error[] errors = backward(new Error[] {outputErrors[i]}, this, false, learningRate);
			assert (errors != null && errors.length == 1 && errors[0] != null);
			if (errors != null) outputErrorList.add(errors[0]);
		}
		return outputErrorList.size() > 0 ? outputErrorList.toArray(new Error[] {}) : null;
	}

	
	/**
	 * Updating parameters from backward information.
	 * @param recordCount count of records in sample.
	 * @param learningRate learning rate.
	 */
	protected void updateParametersFromBackwardInfo(int recordCount, double learningRate) {
		assert (tempAccum == recordCount && recordCount > 0);
		
		//Update weight bias, first weight, and second weight.
		if (this.bias != null && this.dWBiasAccum != null) {
			Matrix dBiasMean0 = this.dWBiasAccum.divide0(recordCount);
			Matrix bias0 = this.getBias().add(dBiasMean0.multiply0(learningRate));
			this.setBias(bias0);
			this.dWBiasAccum = null;
		}
		if (this.weight != null && this.dWKernelAccum != null) {
			Kernel dWMean0 = this.dWKernelAccum.divide(recordCount);
//			this.weight = this.weight.accumKernel(dWMean0, learningRate, decay(learningRate, recordCount));
			if (this.weight != this.weight.accumKernel(dWMean0, learningRate, decay(learningRate, recordCount))) throw new IllegalArgumentException();
			this.dWKernelAccum = null;
		}
		if (this.weight != null && this.weight instanceof NetworkWeight) {
			((NetworkWeight)this.weight).updateParametersFromBackwardInfo(recordCount, learningRate);
		}

		//Update filter and filter bias.
		if (this.filter != null && this.isLearnFilter()) {
			if (this.dFBiasAccum != null) {
				NeuronValue dFilterBiasMean0 = this.dFBiasAccum.divide(recordCount);
				NeuronValue filterBias0 = this.filterBias.add(dFilterBiasMean0.multiply(learningRate));
				this.setFilterBias(filterBias0);
				this.dFBiasAccum = null;
			}
			
			if (this.dFKernelAccum != null) {
				Kernel dFilterKernelMean0 = this.dFKernelAccum.divide(recordCount);
//				this.filter = accumFilterKernel(dFilterKernelMean0, learningRate, recordCount);
				if (this.filter != accumFilterKernel(dFilterKernelMean0, learningRate, recordCount)) throw new IllegalArgumentException();
				this.dFKernelAccum = null;
			}
		}
		if (this.filter != null && this.isLearnFilter() && this.filter instanceof NetworkFilter) {
			((NetworkFilter)this.filter).updateParametersFromBackwardInfo(recordCount, learningRate);
		}
		
		//Resetting all accumulative quantities only in this class.
		this.dWKernelAccum = null;
		this.dWBiasAccum = null;
		this.dFKernelAccum = null;
		this.dFBiasAccum = null;
		
		this.tempAccum = 0;
	}
	
	
//	/*
//	 * Please pay attention that filter is before weight in the same layer.
//	 * This method is the core of matrix neural network.
//	 */
//	/**
//	 * Back-warding layer as learning matrix neural network.
//	 * This method is the core of matrix neural network.
//	 * @param outputErrors core last errors which are core last biases.
//	 * @param focus focused layer to stop back-warding.
//	 * @param learning learning flag. If it is false, parameters are not updated (learned).
//	 * @param learningRate learning rate.
//	 * @return backward error.
//	 */
//	public Error[] backward(Error[] outputErrors, MatrixLayer focus, boolean learning, double learningRate) {
//		if (outputErrors == null || outputErrors.length == 0) return null;
//		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? Network.LEARN_RATE_DEFAULT : learningRate;
//		if (focus == null) learning = true;
//		
//		Matrix[] errors = new Matrix[outputErrors.length];
//		Kernel[] dWKernels = new Kernel[outputErrors.length];
//		NeuronValue[] dFBiases = new NeuronValue[outputErrors.length];
//		Kernel[] dFKernels = new Kernel[outputErrors.length];
//
//		//Browsing errors. Please pay attention that filter is before weight in the same layer.
//		for (int i = 0; i < outputErrors.length; i++) {
//			//Calculating value errors from next layer.
//			if (this.nextLayer == null) {
//				errors[i] = outputErrors[i].error(); //Getting value errors from environment.
//			}
//			else if (this.nextLayer.getFilter() == null && this.nextLayer.getWeight().backwardErrorMode()) {
//				Matrix input = queryInput(), output = queryOutput(); //X^k-1 = input, Xk = output.
//				Function thisActivateRef = input == getInput() ? this.getWeightActivateRef() :
//					(input == getPrevInput() && this.filter.doesApplyActivate() ? this.getFilterActivateRef() : null ); //Getting right-most activation function.
//				
//				//Setting function of index-mode filter like max-pooling filter to be null because of the filtered result of index-mode filter is indexing matrix.
//				if (this.weight == null && this.filter != null && this.filter.isIndexMode()) thisActivateRef = null;
//				
//				errors[i] = this.nextLayer.getWeight().dValue(input, output, outputErrors[i].error(), thisActivateRef);
//			}
//			else {
//				errors[i] = outputErrors[i].error(); //Getting value errors from next layer.
//			}
//			
//			//Calculating weight gradient.
//			if (this.weight != null) {
//				if (this.weight.backwardErrorMode()) {
//					Matrix prevOutput = getPrevOutput();
//					prevOutput = prevOutput != null ? prevOutput : this.prevLayer.queryOutput(); //Xk-1
//					dWKernels[i] = this.weight.dKernel(prevOutput, errors[i]);
//				}
//				else {
//					//Calculating value errors at this layer.
//					Matrix prevInput = getPrevOutput();
//					prevInput = prevInput != null ? prevInput : this.prevLayer.queryOutput(); //Xk-1
//					Matrix prevOutput = getInput();
//					if (this.weight instanceof NetworkWeight) {
//						errors[i] = ((NetworkWeight)this.weight).dValue(prevInput, prevOutput, errors[i], this.getWeightActivateRef(), learning, learningRate);
//						dWKernels[i] = null;
//					}
//					else {
//						errors[i] = this.weight.dValue(prevInput, prevOutput, errors[i], this.getWeightActivateRef());
//						dWKernels[i] = this.weight.dKernel(prevOutput, errors[i]);
//					}
//				}
//			} //Calculating weight gradient.
//
//			//Calculating filter gradient.
//			if (this.filter != null) {
//				if (this.weight != null && this.weight.backwardErrorMode()) {
//					//Calculating value errors at this layer.
//					Matrix prevInput = getPrevInput(), prevOutput = getPrevOutput(); //X^k-1 = input, Xk = output.
//					Function thisActivateRef = this.filter.doesApplyActivate() && !this.filter.isIndexMode() ? this.getFilterActivateRef() : null; //Setting function of index-mode filter like max-pooling filter to be null because of the filtered result of index-mode filter is indexing matrix.
//					errors[i] = this.weight.dValue(prevInput, prevOutput, errors[i], thisActivateRef);
//				}
//				dFBiases[i] = MatrixUtil.valueSum(errors[i]); //Filter errors.
//				dFKernels[i] = dFilterKernel(errors[i], outputErrors[i]);
//				outputErrors[i].errorSet(dFilterValue(errors[i], learning, learningRate, outputErrors[i])); //Please pay attention to this code line to back-warding value errors.
//			}
//			else {
//				outputErrors[i].errorSet(errors[i]); //Please pay attention to this code line to back-warding value errors.
//			} //Calculating filter gradient.
//		} //Browsing errors.
//		
//		//Update weight bias, first weight, and second weight.
//		if (this.bias != null) {
//			Matrix dBiasMean = MatrixUtil.mean(errors);
//			if (learning) {
//				Matrix bias = this.bias.add(dBiasMean.multiply0(learningRate));
//				this.setBias(bias);
//			}
//			else
//				this.dWBiasAccum = this.dWBiasAccum != null ? this.dWBiasAccum.add(dBiasMean) : dBiasMean;
//		}
//		if (this.weight != null && dWKernels[0] != null) {
//			Kernel dWKernelMean = Kernel.mean(dWKernels);
//			if (learning) {
////				this.weight = this.weight.accumKernel(dWKernelMean, learningRate);
//				if (this.weight != this.weight.accumKernel(dWKernelMean, learningRate)) throw new IllegalArgumentException();
//			}
//			else 
//				this.dWKernelAccum = this.dWKernelAccum != null ? this.dWKernelAccum.add(dWKernelMean) : dWKernelMean;
//		}
//		
//		//Update filter and filter bias.
//		if (this.filter != null && isLearnFilter()) {
//			NeuronValue dFilterBiasMean = NeuronValue.valueMean(dFBiases);
//			if (learning) {
//				NeuronValue filterBias = this.filterBias.add(dFilterBiasMean.multiply(learningRate));
//				setFilterBias(filterBias);
//			}
//			else
//				this.dFBiasAccum = this.dFBiasAccum != null ? this.dFBiasAccum.add(dFilterBiasMean) : dFilterBiasMean;
//			
//			if (dFKernels[0] != null) {
//				Kernel dFilterKernelMean = Kernel.mean(dFKernels);
//				if (learning) {
////					this.filter = accumFilterKernel(dFilterKernelMean, learningRate);
//					if (this.filter != accumFilterKernel(dFilterKernelMean, learningRate)) throw new IllegalArgumentException();
//				}
//				else
//					this.dFKernelAccum = this.dFKernelAccum != null ? this.dFKernelAccum.add(dFilterKernelMean) : dFilterKernelMean;
//			}
//		}
//		
//		this.tempAccum += outputErrors.length;
//		
//		//Returning output errors if there is no previous layers or this layer is focused layer.
//		if (this.prevLayer == null || this == focus) return outputErrors;
//		
//		errors = null;
//		dWKernels = null;
//		dFBiases = null;
//		dFKernels = null;
//		//Browsing backward layers.
//		return this.prevLayer.backward(outputErrors, focus, learning, learningRate);
//	}


}
