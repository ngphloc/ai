/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.awt.Dimension;

import net.ea.ann.conv.ConvLayer2DAbstract;
import net.ea.ann.conv.ConvLayerSingle2D;
import net.ea.ann.conv.ConvNeuron;
import net.ea.ann.conv.filter.Filter2D;
import net.ea.ann.conv.filter.ProductFilter2D;
import net.ea.ann.core.Id;
import net.ea.ann.core.Network;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.NeuronValueRaster;
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
	 * Very small number for initialization.
	 */
	protected final static double EPSILON = Network.LEARN_TERMINATED_THRESHOLD_DEFAULT;
	
	
	/**
	 * The first weight matrix.
	 */
	protected Matrix weight1 = null;
	
	
	/**
	 * The second weight matrix.
	 */
	protected Matrix weight2 = null;

	
	/**
	 * Bias.
	 */
	protected Matrix bias = null;

	
	/**
	 * Previous input value.
	 */
	protected ConvLayerSingle2D prevInput = null;
	
	
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
	protected Filter2D filter = null;
	
	
	/**
	 * Convolutional filter bias.
	 */
	protected NeuronValue filterBias = null;
	
	
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
		this.prevInput = null;
		this.input = this.output = null;
		this.weight1 = this.weight2 = this.bias = null;
		this.filter = null;
		this.filterBias = null;
		this.setPrevLayer(null);
		this.setNextLayer(null);
	}
	
	
	/**
	 * Initializing layer with size, previous layer size, and filter.
	 * @param size this size.
	 * @param prevSize previous layer size. It can be null.
	 * @param filter filter. It can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension size, Dimension prevSize, Filter2D filter) {
		this.prevInput = null;
		this.input = this.output = null;
		this.weight1 = this.weight2 = this.bias = null;
		this.filter = null;
		this.filterBias = null;
		this.setPrevLayer(null);
		this.setNextLayer(null);

		//Initialize filter and weights.
		if (size == null || size.height <= 0 || size.width <= 0)
			return false;
		else if (prevSize != null) {
			if (prevSize.height <= 0 || prevSize.width <= 0)
				return false;
			else if (filter != null) {
				this.filter = filter;
//				this.weight1 = newMatrix(size.height, size.height);
			}
			else if (prevSize.height == size.height && prevSize.width == size.width) {
				this.weight1 = newMatrix(size.height, size.height);
				this.weight2 = null;
			}
			else if (prevSize.height != size.height && prevSize.width == size.width) {
				this.weight1 = newMatrix(size.height, prevSize.height);
				this.weight2 = null;
			}
			else if (prevSize.height == size.height && prevSize.width != size.width) {
				this.weight1 = null;
				this.weight2 = newMatrix(prevSize.width, size.width);
			}
			else {
				this.weight1 = newMatrix(size.height, prevSize.height);
				this.weight2 = newMatrix(prevSize.width, size.width);
			}
		}
		else if (filter != null) {
			this.filter = filter;
		}
		
		//Initialize ones related to filter.
		if (this.filter != null) {
			if (isVectorized()) {
				int height = getVecRows();
				int width = size.height / height;
				if (width == 0) return false;
				size = new Dimension(width, height);
				this.prevInput = newConvLayer(width, height);
			}
			else
				this.prevInput = newConvLayer(size.width, size.height);
			this.filterBias = newNeuronValue();
		}
		
		//Initialize ones related to weights.
		if (this.weight1 != null) Matrix.fill(this.weight1, EPSILON);
		if (this.weight2 != null) Matrix.fill(this.weight2, EPSILON);
		if (this.weight1 != null || this.weight2 != null) this.bias = newMatrix(size.height, size.width);
		if (this.bias != null) {
			this.input = newMatrix(size.height, size.width);
			this.output = newMatrix(size.height, size.width);
		}
		else if (this.filter == null) {
			this.output = this.input = newMatrix(size.height, size.width); //Only for input layer where both filter and weights are null and so, its input and output must be initialized.
		}
		
		return true;
	}
	
	
	/**
	 * Initializing layer with size, previous layer, and filter.
	 * @param size this size.
	 * @param prevSize previous size.
	 * @param prevLayer previous layer. It can be null.
	 * @param filter filter. It can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension size, Dimension prevSize, MatrixLayerAbstract prevLayer, Filter2D filter) {
		if (prevLayer == null) return initialize(size, (Dimension)null, filter);
		if (prevSize == null) {
			Matrix prevInput = prevLayer.queryOutput();
			if (prevInput == null) return false;
			prevSize = new Dimension(prevInput.columns(), prevInput.rows());
		}
		if (!initialize(size, prevSize, filter)) return false;
		
		//Connecting two layers.
		this.setPrevLayer(prevLayer);
		
		return true;
	}
	
	
	@Override
	protected ConvLayerSingle2D getPrevInputConvLayer() {
		return prevInput;
	}


	@Override
	protected Matrix getPrevInput() {
		return convLayerToMatrix(this.prevInput);
	}


	@Override
	protected void setPrevInput(Matrix prevInput) {
		if (this.prevInput == null || prevInput == null) return;
		prevInput = isVectorized() ? prevInput.vecInverse(vecRows) : prevInput;
		int rows = prevInput.rows();
		int columns = prevInput.columns();
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				ConvNeuron neuron = this.prevInput.get(j, i);
				neuron.clear();
				neuron.setValue(prevInput.get(i, j));
			}
		}
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
	protected Matrix getWeight1() {
		return weight1;
	}

	
	@Override
	protected void setWeight1(Matrix weight1) {
		this.weight1 = weight1;
	}


	@Override
	protected Matrix getWeight2() {
		return weight2;
	}


	@Override
	protected void setWeight2(Matrix weight2) {
		this.weight2 = weight2;
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
	protected boolean containsWeights() {
		return weight1 != null || weight2 != null;
	}


	@Override
	protected boolean removeWeights() {
		if (!containsWeights() || getFilter() != null || this.prevLayer == null) return false;
		Matrix prevOutput = this.prevLayer.queryOutput();
		if (prevOutput == null) return false;
		Matrix output = this.queryOutput();
		if (output == null) return false;

		Dimension prevSize = new Dimension(prevOutput.columns(), prevOutput.rows());
		Dimension size = new Dimension(output.columns(), output.rows());
		if (size.height <= 0 || size.width <= 0) return false;
		if (prevSize.height <= 0 || prevSize.width <= 0) return false;
		int strideHeight = prevSize.height/size.height;
		int strideWidth = prevSize.width/size.width;
		if (strideHeight == 0 || strideWidth == 0) return false;
		Dimension filterStride = new Dimension(strideWidth, strideHeight);
		
		this.filter = getNetwork() != null ? getNetwork().defaultFilter(filterStride) :
			ProductFilter2D.create(new Size(filterStride), this, 1.0/(double)(filterStride.height*filterStride.width));
		if (isVectorized()) {
			int height = getVecRows();
			int width = size.height / height;
			if (width == 0) return false;
			size = new Dimension(width, height);
			this.prevInput = newConvLayer(width, height);
		}
		else
			this.prevInput = newConvLayer(size.width, size.height);
		this.filterBias = newNeuronValue();
		
		return true;
	}
	
	
	@Override
	protected Filter2D getFilter() {
		return filter;
	}


	@Override
	protected void setFilter(Filter2D filter) {
		this.filter = filter;
	}


	@Override
	protected boolean removeFilter() {
		if (this.filter == null || this.prevLayer == null) return false;
		Matrix prevOutput = this.prevLayer.queryOutput();
		if (prevOutput == null) return false;
		Matrix output = this.queryOutput();
		if (output == null) return false;
		
		Dimension prevSize = new Dimension(prevOutput.columns(), prevOutput.rows());
		Dimension size = new Dimension(output.columns(), output.rows());
		if (size.height <= 0 || size.width <= 0) return false;
		if (prevSize.height <= 0 || prevSize.width <= 0) return false;
		
		if (prevSize.height == size.height && prevSize.width == size.width) {
			this.weight1 = newMatrix(size.height, size.height);
			this.weight2 = null;
		}
		else if (prevSize.height != size.height && prevSize.width == size.width) {
			this.weight1 = newMatrix(size.height, prevSize.height);
			this.weight2 = null;
		}
		else if (prevSize.height == size.height && prevSize.width != size.width) {
			this.weight1 = null;
			this.weight2 = newMatrix(prevSize.width, size.width);
		}
		else {
			this.weight1 = newMatrix(size.height, prevSize.height);
			this.weight2 = newMatrix(prevSize.width, size.width);
		}
		if (this.weight1 == null && this.weight2 == null) return false;
		
		if (this.weight1 != null) Matrix.fill(this.weight1, EPSILON);
		if (this.weight2 != null) Matrix.fill(this.weight2, EPSILON);
		this.bias = newMatrix(size.height, size.width);
		this.input = newMatrix(size.height, size.width);
		this.output = newMatrix(size.height, size.width);
		
		this.filter = null;
		this.filterBias = null;
		this.prevInput = null;
		
		return true;
	}
	
	
	/**
	 * Evaluating by filtering.
	 * @return filtered matrix.
	 */
	private Matrix evaluateByFilter() {
		if (this.prevLayer == null) return null;
		if (this.filter == null) return null;
		Matrix prevLayerOutput = this.prevLayer.queryOutput();
		if (prevLayerOutput == null) return null;
		
		ConvLayerSingle2D prevConvLayer = this.prevLayer.matrixToConvLayer(prevLayerOutput);
		ConvLayer2DAbstract.forward(prevConvLayer, this.prevInput, this.filter);
		return getPrevInput();
	}
	
	
	@Override
	public Matrix evaluate() {
		if (this.prevLayer == null) return null;
		Matrix prevInput = this.filter != null ? evaluateByFilter() : null;
		if (weight1 == null && weight2 == null) return prevInput;
		
		this.input = prevInput != null ? prevInput : this.prevLayer.queryOutput();
		if (weight1 != null) input = weight1.multiply(input);
		if (weight2 != null) input = input.multiply(weight2);
		input = input.add(bias);
		
		output = activateRef != null ? input.evaluate0(activateRef) : input;
		return output;
	}


	@Override
	public Matrix forward(Matrix input) {
		if (this.prevLayer == null) {
			if (input != null) Matrix.copy(input, this.input);
			if (this.input != this.output) this.output = this.input;
			if (this.nextLayer == null) return input;
		}
		
		MatrixLayer nextLayer = null;
		Matrix resulted = null;
		while ((nextLayer = this.getNextLayer()) != null) {
			resulted = nextLayer.evaluate();
		}
		return resulted;
	}


	@Override
	public Matrix[] backward(Matrix[] outputErrors, MatrixLayer focus, boolean learning, double learningRate) {
		if (outputErrors == null || outputErrors.length == 0) return null;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? Network.LEARN_RATE_DEFAULT : learningRate;
		if (focus == null) learning = true;
		
		Matrix[] errors = new Matrix[outputErrors.length];
		Matrix[] dW1s = new Matrix[outputErrors.length];
		Matrix[] dW2s = new Matrix[outputErrors.length];
		NeuronValue[] dFilterErrors = new NeuronValue[outputErrors.length];
		NeuronValue[][][] dFilterKernels = new NeuronValue[outputErrors.length][][];
		MatrixLayerAbstract prevLayer = getPrevLayer();
		MatrixLayerAbstract nextLayer = getNextLayer();

		//Browsing errors.
		for (int j = 0; j < outputErrors.length; j++) {
			if (nextLayer == null) {
				//Getting errors from environment.
				errors[j] = outputErrors[j];
				
				//Training adapter here.
			}
			else {
				if (nextLayer.getFilter() == null) {
					Matrix input = this.getInput(); //X'k-1
					Matrix output = this.queryOutput(); //Xk
					Matrix derivative = input != null ? input.derivativeWise(this.getActivateRef()) : null;
					
					//Updating errors based on weights.
					Matrix nextW1T = nextLayer.getWeight1();
					Matrix nextW2 = nextLayer.getWeight2();
					nextW1T = (nextW1T != null) ? nextW1T.transpose() : output.createIdentity(output.rows());
					nextW2 = (nextW2 != null) ? nextW2 : output.createIdentity(output.columns());
					
					Matrix[] errorArray = new Matrix[nextW2.rows()];
					Matrix vecNextError = outputErrors[j].vec(); //Please pay attention to this code line.
					for (int row = 0; row < errorArray.length; row++) {
						//errorArray[row] = Matrix.kroneckerProductMutilply(nextW2, nextW1T, row, vecNextError);
						errorArray[row] = nextW2.kroneckerProductRowOf(nextW1T, row).multiply(vecNextError); //Faster.
					}
					errors[j] = Matrix.concatV(errorArray);
					errors[j] = derivative != null ? derivative.multiplyWise(errors[j]) : errors[j];
				}
				else {
					errors[j] = outputErrors[j]; //Please pay attention to this code line.
				}
				
			} //Calculating errors[j]
				
			//Updating outputErrors[j] by filter.
			if (this.getFilter() != null) {
				ConvLayerSingle2D prevLayer2D = prevLayer.matrixToConvLayer(prevLayer.queryOutput());
				ConvLayerSingle2D errorj = this.matrixToConvLayer(errors[j]);
				NeuronValueRaster dValues = prevLayer2D.dValue(errorj, this.getFilter());
				//Please pay attention to this code line to assign current errors to output errors.
				outputErrors[j] = prevLayer.arrayToMatrix(dValues.getValues(), prevLayer2D.getHeight(), prevLayer2D.getWidth());
				
				if (this.isLearnFilter()) {
					dFilterErrors[j] = dValues.getCountValues() > 0 ? Matrix.valueSum(outputErrors[j]).divide(dValues.getCountValues()) :
						outputErrors[j].get(0, 0).zero(); //Filter errors.
					dFilterKernels[j] = prevLayer2D.dKernel(errorj, this.getFilter()); //Filter kernel errors.
				}
			}

			//Update weight errors[j].
			if (this.containsWeights()) {
				Matrix W1 = this.getWeight1();
				Matrix W2 = this.getWeight2();
				Matrix prevInput = this.getPrevInput();
				prevInput = prevInput != null ? prevInput : prevLayer.queryOutput(); //Xk-1
				
				Matrix vecError = errors[j].vec();
				if (W1 != null) {
					Matrix XW2 = W2 != null ? prevInput.multiply(W2) : prevInput;
					Matrix I = W1.createIdentity(W1.rows());
					Matrix[] W1s = new Matrix[XW2.rows()];
					for (int row = 0; row < W1s.length; row++) {
						//W1s[row] = Matrix.kroneckerProductMutilply(XW2, I, row, vecError); //Lower but consuming less memory.
						W1s[row] = XW2.kroneckerProductRowOf(I, row).multiply(vecError); //Faster.
					}
					dW1s[j] = Matrix.concatV(W1s);
				}
				
				if (W2 != null) {
					Matrix W1XT = W1 != null ? W1.multiply(prevInput) : prevInput;
					W1XT = W1XT.transpose();
					Matrix I = W2.createIdentity(W2.columns());
					Matrix[] W2s = new Matrix[I.rows()];
					for (int row = 0; row < W2s.length; row++) {
						//W2s[row] = Matrix.kroneckerProductMutilply(I, W1XT, row, vecError); //Lower but consuming less memory.
						W2s[row] = I.kroneckerProductRowOf(W1XT, row).multiply(vecError); //Faster.
					}
					dW2s[j] = Matrix.concatV(W2s);
				}
			} //Updating error of W1 and W2
			
		} //End browsing errors.
		
		
		//Update weight bias, first weight, and second weight.
		if (this.getBias() != null && learning) {
			Matrix biasMean = Matrix.mean(errors);
			Matrix bias = this.getBias().add(biasMean.multiply0(learningRate));
			this.setBias(bias);
		}
		if (this.getWeight1() != null && learning) {
			Matrix w1Mean = Matrix.mean(dW1s);
			Matrix w1 = this.getWeight1().add(w1Mean.multiply0(learningRate));
			this.setWeight1(w1);
		}
		if (this.getWeight2() != null && learning) {
			Matrix w2Mean = Matrix.mean(dW2s);
			Matrix w2 = this.getWeight2().add(w2Mean.multiply0(learningRate));
			this.setWeight2(w2);
		}
		
		//Update filter and filter bias.
		if (this.getFilter() != null && this.isLearnFilter() && learning) {
			NeuronValue filterErrorsMean = NeuronValue.valueMean(dFilterErrors);
			NeuronValue filterBias = this.getFilterBias().add(filterErrorsMean.multiply(learningRate));
			this.setFilterBias(filterBias); //Update filter bias.
			
			if (this.getFilter() instanceof ProductFilter2D) {
				ProductFilter2D filter = (ProductFilter2D)this.getFilter();
				NeuronValue[][] filterKernelsMean = ProductFilter2D.kernelMean(dFilterKernels);
				filterKernelsMean = NeuronValue.multiply(filterKernelsMean, learningRate);
				filter = filter.shallowClone();
				filter.accumKernel(filterKernelsMean);
				this.setFilter(filter); //Update filter.
			}
		}
		
		//Please pay attention to this code line to assign current errors to output errors.
		if (this.getFilter() == null) outputErrors = errors;

		//Returning output errors if there is no previous layers.
		if (outputErrors == null || prevLayer == null) return outputErrors;
		//Stop at focused layer.
		if (this == focus) return outputErrors;
		
		//Browsing backward layers.
		errors = null;
		dW1s = null;
		dW2s = null;
		dFilterErrors = null;
		dFilterKernels = null;
		return prevLayer.backward(outputErrors, focus, learning, learningRate);
	}


	/**
	 * Learning matrix neural network.
	 * @param outputErrors core last errors which are core last biases.
	 * @param learningRate learning rate.
	 * @return training error.
	 */
	public Matrix[] learn(Matrix[] outputErrors, double learningRate) {
		 return backward(outputErrors, this, true, learningRate);
	}

	
}
