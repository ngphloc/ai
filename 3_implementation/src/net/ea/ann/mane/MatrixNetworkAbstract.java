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
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueCreator;
import net.ea.ann.mane.filter.Filter;
import net.ea.ann.raster.Image;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAbstract;
import net.ea.ann.raster.Size;

/**
 * This class implements partially matrix neural network in default.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class MatrixNetworkAbstract extends NetworkAbstract implements MatrixNetwork, MatrixLayerExt, NeuronValueCreator {


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
	public final static String VECTORIZED_FIELD = "mane_vectorized";
	
	
	/**
	 * Default value of vectorization field.
	 */
	public final static boolean VECTORIZED_DEFAULT = false;

	
	/**
	 * Field for middle size.
	 */
	public static final String MIDSIZE_FIELD = "classifier_midsize";
	
	
	/**
	 * Default value for middle size.
	 */
	public static final int MIDSIZE_DEFAULT = 0;

	
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
		this.config.put(LEARN_MAX_ITERATION_FIELD, LEARN_MAX_ITERATION_DEFAULT);
		this.config.put(RasterAbstract.RASTER_CHANNEL_FIELD, RasterAbstract.RASTER_CHANNEL_DEFAULT);
		this.config.put(Raster.NORM_FIELD, Raster.NORM_DEFAULT);
		this.config.put(Image.ALPHA_FIELD, Image.ALPHA_DEFAULT);
		this.config.put(LARGE_SCALE_FIELD, LARGE_SCALE_DEFAULT);
		this.config.put(VECTORIZED_FIELD, VECTORIZED_DEFAULT);
		this.config.put(MatrixLayerAbstract.LEARN_FILTER_FIELD, MatrixLayerAbstract.LEARN_FILTER_DEFAULT);
		this.config.put(MIDSIZE_FIELD, MIDSIZE_DEFAULT);

		this.neuronChannel = neuronChannel = (neuronChannel < 1 ? 1 : neuronChannel);
		this.activateRef = activateRef == null ? (activateRef = Raster.toActivationRef(this.neuronChannel, paramIsNorm())) : activateRef;
		this.convActivateRef = convActivateRef == null ? (convActivateRef = Raster.toConvActivationRef(this.neuronChannel, paramIsNorm())) : convActivateRef;
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
	
	
	@Override
	public NeuronValue newNeuronValue() {
		return newLayer().newNeuronValue();
	}


	/**
	 * Creating filter.
	 * @param filterSize filter size.
	 * @param layerSpec layer specification.
	 * @return filter.
	 */
	Filter newFilter(Size filterSize, MatrixLayerAbstract.LayerSpec layerSpec) {
		Filter filter = newLayer().newFilter(filterSize, layerSpec);
		if (filter != null && paramGetMiddleSize() > 0) filter.setMoveStride(true);
		return filter;
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
	

	@Override
	public MatrixLayerAbstract getOutputLayer() {return layers[layers.length-1];}

	
//	@Override
//	public Function getOutputActivateRef() {return layers[layers.length-1].activateRef;}


	@Override
	public void enterInputs(Record record) {
		MatrixLayerAbstract inputLayer = getInputLayer();
		Matrix input = record.input();
		if (input != null) MatrixUtil.copy(input, inputLayer.getInput());
	}


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
			Matrix matrixInput = inputLayer.toMatrix(inputRaster, new Size(input.columns(), input.rows()));
			return matrixInput != null ? evaluate(matrixInput) : null;
		} catch (Throwable e) {Util.trace(e);}
		return null;
	}


	/**
	 * Learning matrix neural network.
	 * @param inouts sample.
	 * @return learned error.
	 */
	public Error[] learnByRaster(Iterable<Raster[]> inouts) {
		try {
			MatrixLayerAbstract inputLayer = getInputLayer();
			Matrix input = inputLayer.getInput();
			MatrixLayerAbstract outputLayer = getOutputLayer();
			Matrix output = outputLayer.queryOutput();
			List<Record> sample = Util.newList(0);
			for (Raster[] inout : inouts) {
				Matrix matrixInput = inputLayer.toMatrix(inout[0], new Size(input.columns(), input.rows()));
				Matrix matrixOutput = outputLayer.toMatrix(inout[1], new Size(output.columns(), output.rows()));
				if (matrixInput != null && matrixOutput != null)
					sample.add(new Record(matrixInput, matrixOutput));
			}
			return sample.size() > 0 ? learn(sample) : null;
		} catch (Throwable e) {Util.trace(e);}
		return null;
	}
	
	
	/**
	 * Back-warding layer as learning matrix neural network.
	 * @param outputErrors core last errors which are core last biases.
	 * @param learningRate learning rate.
	 * @return training error.
	 */
	public Error[] backwardWithoutLearning(Error[] outputErrors, double learningRate) {
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
	public void updateParametersFromBackwardInfo(int recordCount, double learningRate) {
		if (layers == null) return;
		for (int i = layers.length-1; i >= 0; i--) {
			if (!(layers[i] instanceof MatrixLayerImpl)) continue;
			MatrixLayerImpl layer = (MatrixLayerImpl)layers[i];
			layer.updateParametersFromBackwardInfo(recordCount, learningRate);
		}
	}
	

	/**
	 * Resetting backward information.
	 */
	public void resetBackwardInfo() {
		if (layers == null) return;
		for (int i = layers.length-1; i >= 0; i--) {
			if (!(layers[i] instanceof MatrixLayerImpl)) continue;
			MatrixLayerImpl layer = (MatrixLayerImpl)layers[i];
			layer.resetBackwardInfo();
		}
	}
	

	/**
	 * Calculating the last bias which is often the negative of output error, often multiplied with gradient.
	 * Derived class can override this method but it is better to apply the method {@link #setLikelihoodGradient(LikelihoodGradient)} into changing how to calculate the bias (error). 
	 * @param output computed or predicted output.
	 * @param realOutput real output from environment. It is can be null.
	 * @param outputLayer output layer. It is can be null.
	 * @param params additional parameters.
	 * @return the last bias.
	 */
	protected Matrix calcOutputError(Matrix output, Matrix realOutput, MatrixLayerAbstract outputLayer, Object...params) {
		LikelihoodGradient grad = this.likelihoodGradient;
		if (grad == null) grad = LikelihoodGradient::error;
		Matrix error = grad.gradient(output, realOutput, params);
		
		if (outputLayer == null) return error;
		Matrix input = outputLayer.getInput();
		Matrix derivative = input != null ? input.derivativeWise(outputLayer.getActivateRef()) : null;
		return derivative != null ? derivative.multiplyWise(error) : error;
	}

	
	/**
	 * Defining additional parameters for output errors.
	 * @return additional parameters for output errors.
	 */
	protected Object[] defineOutputErrorParams() {return null;}
	

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
	 * Getting activation function.
	 * @return activation function.
	 */
	public Function getActivateRef() {return activateRef;}
	
	
	/**
	 * Getting convolutional activation function.
	 * @return convolutional activation function.
	 */
	public Function getConvActivateRef() {return convActivateRef;}
	
	
	/**
	 * Checking normalization mode.
	 * @return normalization mode in rang [0, 1].
	 */
	public boolean paramIsNorm() {
		if (config.containsKey(Raster.NORM_FIELD))
			return config.getAsBoolean(Raster.NORM_FIELD);
		else
			return Raster.NORM_DEFAULT;
	}


	/**
	 * Setting normalization mode.
	 * @param isNorm normalization mode in rang [0, 1]..
	 * @return this network.
	 */
	public MatrixNetworkAbstract paramSetNorm(boolean isNorm) {
		if (paramIsNorm() == isNorm) return this;
		this.config.put(Raster.NORM_FIELD, isNorm);
		this.activateRef = Raster.toActivationRef(this.neuronChannel, isNorm);
		this.convActivateRef = Raster.toConvActivationRef(this.neuronChannel, isNorm);
		return this;
	}

	
	/**
	 * Getting default alpha.
	 * @return default alpha.
	 */
	int paramGetDefaultAlpha() {
		if (config.containsKey(Image.ALPHA_FIELD))
			return config.getAsInt(Image.ALPHA_FIELD);
		else
			return Image.ALPHA_DEFAULT;
	}

	
	/**
	 * Checking whether the network is large scale.
	 * @return whether the network is large scale.
	 */
	boolean paramIsLargeScale() {
		if (config.containsKey(LARGE_SCALE_FIELD))
			return config.getAsBoolean(LARGE_SCALE_FIELD);
		else
			return LARGE_SCALE_DEFAULT;
	}
	

	/**
	 * Getting raster channel.
	 * @return raster channel.
	 */
	int paramGetRasterChannel() {
		if (config.containsKey(RasterAbstract.RASTER_CHANNEL_FIELD))
			return config.getAsInt(RasterAbstract.RASTER_CHANNEL_FIELD);
		else
			return RasterAbstract.RASTER_CHANNEL_DEFAULT;
	}
	
	
	/**
	 * Setting raster channel.
	 * @param rasterChannel raster channel.
	 * @return this network.
	 */
	MatrixNetworkAbstract paramSetRasterChannel(int rasterChannel) {
		rasterChannel = rasterChannel < 1 ? RasterAbstract.RASTER_CHANNEL_DEFAULT : rasterChannel;
		config.put(RasterAbstract.RASTER_CHANNEL_FIELD, rasterChannel);
		return this;
	}

	
	/**
	 * Checking vectorization mode.
	 * @return vectorization mode.
	 */
	public boolean paramIsVectorized() {
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
	public MatrixNetworkAbstract paramSetVectorized(boolean vectorized) {
		config.put(VECTORIZED_FIELD, vectorized);
		return this;
	}
	
		
	/**
	 * Checking whether filter is learned.
	 * @return whether filter is learned.
	 */
	public boolean paramIsLearnFilter() {
		if (config.containsKey(MatrixLayerAbstract.LEARN_FILTER_FIELD))
			return config.getAsBoolean(MatrixLayerAbstract.LEARN_FILTER_FIELD);
		else
			return MatrixLayerAbstract.LEARN_FILTER_DEFAULT;
	}

	
	/**
	 * Setting whether filter is learned.
	 * @param learnFilter whether filter is learned.
	 */
	public MatrixNetworkAbstract paramSetLearnFilter(boolean learnFilter) {
		config.put(MatrixLayerAbstract.LEARN_FILTER_FIELD, learnFilter);
		return this;
	}


	/**
	 * Checking middle size.
	 * @return middle size.
	 */
	public int paramGetMiddleSize() {
		if (config.containsKey(MIDSIZE_FIELD))
			return config.getAsInt(MIDSIZE_FIELD);
		else
			return MIDSIZE_DEFAULT;
	}
	
	
	/**
	 * Setting middle size.
	 * @param minSize middle size.
	 * @return this network.
	 */
	public MatrixNetworkAbstract paramSetMiddleSize(int minSize) {
		config.put(MIDSIZE_FIELD, minSize);
		return this;
	}

	
}
