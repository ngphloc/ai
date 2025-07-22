/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.awt.Dimension;
import java.util.List;

import net.ea.ann.conv.filter.Filter2D;
import net.ea.ann.conv.filter.ProductFilter2D;
import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.raster.Image;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Size;

/**
 * This class implements partially matrix neural network in default.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class MatrixNetworkAbstract extends NetworkAbstract implements MatrixNetwork {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Name of large scale field.
	 */
	final static String LARGE_SCALE_FIELD = "mane_large_scale";
	
	
	/**
	 * Default value of large scale field.
	 */
	final static boolean LARGE_SCALE_DEFAULT = false;

	
	/**
	 * Name of vectorization field.
	 */
	final static String VECTORIZED_FIELD = "mane_vectorized";
	
	
	/**
	 * Default value of vectorization field.
	 */
	final static boolean VECTORIZED_DEFAULT = false;

	
	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;
	
	
	/**
	 * Activation function reference.
	 */
	protected Function activateRef = null;

	
	/**
	 * Convolutional activation function reference.
	 */
	protected Function convActivateRef = null;

	
	/**
	 * Likelihood gradient for training matrix neural network.
	 */
	protected LikelihoodGradient likelihoodGradient = null;
	
	
	/**
	 * Array of layers.
	 */
	protected MatrixLayerAbstract[] layers = null;
	
	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public MatrixNetworkAbstract(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(idRef);
		this.config.put(LEARN_MAX_ITERATION_FIELD, 1);
		this.config.put(Raster.NORM_FIELD, Raster.NORM_DEFAULT);
		this.config.put(Image.ALPHA_FIELD, Image.ALPHA_DEFAULT);
		this.config.put(LARGE_SCALE_FIELD, LARGE_SCALE_DEFAULT);
		this.config.put(VECTORIZED_FIELD, VECTORIZED_DEFAULT);
		this.config.put(MatrixLayerAbstract.LEARN_FILTER_FIELD, MatrixLayerAbstract.LEARN_FILTER_DEFAULT);

		if (neuronChannel < 1)
			this.neuronChannel = neuronChannel = 1;
		else
			this.neuronChannel = neuronChannel;
		this.activateRef = activateRef == null ? (activateRef = Raster.toActivationRef(this.neuronChannel, true)) : activateRef;
		this.convActivateRef = convActivateRef == null ? (convActivateRef = Raster.toConvActivationRef(this.neuronChannel, true)) : convActivateRef;
	}
	

	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public MatrixNetworkAbstract(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public MatrixNetworkAbstract(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public MatrixNetworkAbstract(int neuronChannel) {
		this(neuronChannel, null, null, null);
	}

	
	/**
	 * Creating matrix layer.
	 * @return matrix layer.
	 */
	protected abstract MatrixLayerAbstract newLayer();
	
	
	/**
	 * Default filter.
	 * @param filterStride1
	 * @return default filter.
	 */
	Filter2D defaultFilter(Dimension filterStride1) {
		if (filterStride1 == null)
			return null;
		else if (filterStride1.width <= 0 || filterStride1.height <= 0)
			return null;
		else
			return ProductFilter2D.create(new Size(filterStride1), newLayer(), 1.0/(double)(filterStride1.height*filterStride1.width));
	}
	
	
	/**
	 * Validating network.
	 * @return true if network is valid.
	 */
	protected boolean validate() {
		return layers != null && layers.length > 1;
	}
	
	
	/**
	 * Getting the number of layers.
	 * @return the number of layers.
	 */
	public int size() {return layers.length;}
	
	
	/**
	 * Getting layer at specified index.
	 * @param index specified index.
	 * @return layer at specified index.
	 */
	public MatrixLayerAbstract get(int index) {return layers[index];}
	
	/**
	 * Getting input layer.
	 * @return input layer.
	 */
	public MatrixLayerAbstract getInputLayer() {return layers[0];}
	
	
	/**
	 * Getting output layer.
	 * @return output layer.
	 */
	public MatrixLayerAbstract getOutputLayer() {return layers[layers.length-1];}

	
	/**
	 * Resetting matrix neural network.
	 */
	public void reset() {
		this.layers = null;
	}
	
	
	/**
	 * Evaluating matrix neural network.
	 * @param inputRaster input raster for evaluating.
	 * @return matrix as output.
	 */
	public Matrix evaluate(Raster inputRaster) {
		try {
			MatrixLayerAbstract inputLayer = getInputLayer();
			Matrix input = inputLayer.getInput();
			Matrix matrixInput = inputLayer.toMatrix(inputRaster, input.rows(), input.columns());
			return matrixInput != null ? evaluate(matrixInput) : null;
		} catch (Throwable e) {Util.trace(e);}
		return null;
	}


	/**
	 * Learning matrix neural network.
	 * @param inouts sample as collection of input and output whose each element is an 2-component array of input (the first) and output (the second).
	 * @return learned error.
	 */
	public Matrix[] learnByRaster(Iterable<Raster[]> inouts) {
		try {
			MatrixLayerAbstract inputLayer = getInputLayer();
			Matrix input = inputLayer.getInput();
			MatrixLayerAbstract outputLayer = getOutputLayer();
			Matrix output = outputLayer.queryOutput();
			List<Matrix[]> sample = Util.newList(0);
			for (Raster[] inout : inouts) {
				Matrix matrixInput = inputLayer.toMatrix(inout[0], input.rows(), input.columns());
				Matrix matrixOutput = outputLayer.toMatrix(inout[1], output.rows(), output.columns());
				if (matrixInput != null && matrixOutput != null)
					sample.add(new Matrix[] {matrixInput, matrixOutput});
			}
			return sample.size() > 0 ? learn(sample) : null;
		} catch (Throwable e) {Util.trace(e);}
		return null;
		
	}
	
	
	/**
	 * Calculating the last bias which is often the negative of output error, often multiplied with gradient.
	 * @param output computed or predicted output.
	 * @param realOutput real output from environment.
	 * @param outputLayer output layer.
	 * @return the last bias.
	 */
	protected Matrix calcOutputError(Matrix output, Matrix realOutput, MatrixLayerAbstract outputLayer) {
		LikelihoodGradient grad = this.likelihoodGradient;
		if (grad == null) grad = LikelihoodGradient::error;
		Matrix error = grad.gradient(output, realOutput);
		
		if (outputLayer == null) return error;
		Matrix input = outputLayer.getInput();
		Matrix derivative = input != null ? input.derivativeWise(outputLayer.getActivateRef()) : null;
		return derivative != null ? derivative.multiplyWise(error) : error;
	}

	
	/**
	 * Getting likelihood gradient.
	 * @return likelihood gradient.
	 */
	LikelihoodGradient getLikelihoodGradient() {
		return likelihoodGradient;
	}
	
	
	/**
	 * Setting likelihood gradient.
	 * @param likelihoodGradient likelihood gradient.
	 * @return this network.
	 */
	MatrixNetworkAbstract setLikelihoodGradient(LikelihoodGradient likelihoodGradient) {
		this.likelihoodGradient = likelihoodGradient;
		return this;
	}
	
	
	/**
	 * Checking whether something normalized in rang [0, 1].
	 * @return whether something normalized in rang [0, 1].
	 */
	boolean isNorm() {
		if (config.containsKey(Raster.NORM_FIELD))
			return config.getAsBoolean(Raster.NORM_FIELD);
		else
			return Raster.NORM_DEFAULT;
	}


	/**
	 * Getting default alpha.
	 * @return default alpha.
	 */
	int getDefaultAlpha() {
		if (config.containsKey(Image.ALPHA_FIELD))
			return config.getAsInt(Image.ALPHA_FIELD);
		else
			return Image.ALPHA_DEFAULT;
	}

	
	/**
	 * Checking whether the network is large scale.
	 * @return whether the network is large scale.
	 */
	boolean isLargeScale() {
		if (config.containsKey(LARGE_SCALE_FIELD))
			return config.getAsBoolean(LARGE_SCALE_FIELD);
		else
			return LARGE_SCALE_DEFAULT;
	}
	

	/**
	 * Checking whether the network data is vectorized.
	 * @return whether the network data is vectorized.
	 */
	public boolean isVectorized() {
		if (config.containsKey(VECTORIZED_FIELD))
			return config.getAsBoolean(VECTORIZED_FIELD);
		else
			return VECTORIZED_DEFAULT;
	}

	
	/**
	 * Setting vectorization mode.
	 * @param vectorized vectorization mode.
	 * @return this network.
	 */
	public MatrixNetworkAbstract setVectorized(boolean vectorized) {
		config.put(VECTORIZED_FIELD, vectorized);
		return this;
	}
	
		
	/**
	 * Checking whether filter is learned.
	 * @return whether filter is learned.
	 */
	public boolean isLearnFilter() {
		if (config.containsKey(MatrixLayerAbstract.LEARN_FILTER_FIELD))
			return config.getAsBoolean(MatrixLayerAbstract.LEARN_FILTER_FIELD);
		else
			return MatrixLayerAbstract.LEARN_FILTER_DEFAULT;
	}

	
	/**
	 * Setting whether filter is learned.
	 * @param learnFilter whether filter is learned.
	 */
	public MatrixNetworkAbstract setLearnFilter(boolean learnFilter) {
		config.put(MatrixLayerAbstract.LEARN_FILTER_FIELD, learnFilter);
		return this;
	}


}
