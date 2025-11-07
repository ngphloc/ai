/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.awt.Dimension;

import net.ea.ann.conv.Content;
import net.ea.ann.conv.ConvLayer2DImpl;
import net.ea.ann.conv.ConvLayerSingle2D;
import net.ea.ann.conv.ConvNeuron;
import net.ea.ann.conv.filter.Filter2D;
import net.ea.ann.core.Id;
import net.ea.ann.core.LayerAbstract;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueCreator;
import net.ea.ann.raster.Image;
import net.ea.ann.raster.Raster;

/**
 * This abstract class implements partially layer in matrix neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class MatrixLayerAbstract extends LayerAbstract implements MatrixLayer, NeuronValueCreator {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Name of learning filter field.
	 */
	final static String LEARN_FILTER_FIELD = "mane_learn_filter";
	
	
	/**
	 * Default value of learning filter field.
	 */
	final static boolean LEARN_FILTER_DEFAULT = true;

	
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
	 * Previous layer.
	 */
	protected MatrixLayerAbstract prevLayer = null;
	
	
	/**
	 * Next layer.
	 */
	protected MatrixLayerAbstract nextLayer = null;

	
	/**
	 * Number of rows in case of vectorization. By default it is zero, which means that there is no vectorization by default.
	 */
	protected int vecRows = 0;
	

	/**
	 * Learning filter.
	 */
	protected boolean learnFilter = LEARN_FILTER_DEFAULT;
	
	
	/**
	 * Reference to matrix neural network.
	 */
	protected MatrixNetworkAbstract network = null;
	
	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public MatrixLayerAbstract(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(idRef);
		
		this.neuronChannel = neuronChannel = (neuronChannel < 1 ? 1 : neuronChannel);
		this.activateRef = activateRef == null ? (activateRef = Raster.toActivationRef(this.neuronChannel, true)) : activateRef;
		this.convActivateRef = convActivateRef == null ? (convActivateRef = Raster.toConvActivationRef(this.neuronChannel, true)) : convActivateRef;
	}
	

	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public MatrixLayerAbstract(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public MatrixLayerAbstract(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public MatrixLayerAbstract(int neuronChannel) {
		this(neuronChannel, null, null, null);
	}

	
	@Override
	public NeuronValue newNeuronValue() {
		NeuronValue value = newMatrix(1, 1).get(0, 0);
		if (value instanceof Matrix)
			return value;
		else if (value instanceof Content)
			return value;
		else
			return value.zero();
	}

	
	/**
	 * Creating matrix.
	 * @param rows rows.
	 * @param columns columns.
	 * @return new matrix.
	 */
	protected Matrix newMatrix(int rows, int columns) {
		NeuronValue value = NeuronValueCreator.newNeuronValue(neuronChannel);
		return Matrix.create(rows, columns, value);
	}


	/**
	 * Creating convolutional layer.
	 * @param width specified width.
	 * @param height specified height.
	 * @return convolutional layer.
	 */
	protected ConvLayerSingle2D newConvLayer(int width, int height) {
		return ConvLayer2DImpl.create(neuronChannel, convActivateRef, width, height, null, idRef);
	}
	

	@Override
	public MatrixLayerAbstract getPrevLayer() {
		return this.prevLayer;
	}
	
	
	/**
	 * Setting previous layer.
	 * @param prevLayer previous layer.
	 * @return true if setting previous layer is successful.
	 */
	protected boolean setPrevLayer(MatrixLayerAbstract prevLayer) {
		if (prevLayer == this.prevLayer) return false;
		
		MatrixLayerAbstract oldPrevLayer = this.prevLayer;
		if (oldPrevLayer != null) oldPrevLayer.nextLayer = null;
		
		this.prevLayer = prevLayer;
		if (prevLayer == null) return true;
		
		MatrixLayerAbstract nextLayerOfPrevLayer = prevLayer.nextLayer;
		if (nextLayerOfPrevLayer != null) nextLayerOfPrevLayer.prevLayer = null;

		prevLayer.nextLayer = this;
		
		return true;
	}


	@Override
	public MatrixLayerAbstract getNextLayer() {
		return this.nextLayer;
	}

	
	/**
	 * Setting next layer.
	 * @param nextLayer next layer.
	 * @return true if setting next layer is successful.
	 */
	protected boolean setNextLayer(MatrixLayerAbstract nextLayer) {
		if (nextLayer == this.nextLayer) return false;

		MatrixLayerAbstract oldNextLayer = this.nextLayer;
		if (oldNextLayer != null) oldNextLayer.prevLayer = null;

		this.nextLayer = nextLayer;
		if (nextLayer == null) return true;
		
		MatrixLayerAbstract prevLayerOfNextLayer = nextLayer.prevLayer;
		if (prevLayerOfNextLayer != null) prevLayerOfNextLayer.nextLayer = null;
		
		nextLayer.prevLayer = this;
		
		return true;
	}


	/**
	 * Getting reference to activation function.
	 * @return reference to activation function.
	 */
	public Function getActivateRef() {
		return activateRef;
	}

	
	/**
	 * Setting reference to activation function.
	 * @param activateRef reference to activation function.
	 * @return previous function reference.
	 */
	protected Function setActivateRef(Function activateRef) {
		return this.activateRef = activateRef;
	}

	
	/**
	 * Getting reference to convolutional activation function.
	 * @return reference to convolutional activation function.
	 */
	public Function getConvActivateRef() {
		return convActivateRef;
	}
	
	
	/**
	 * Setting reference to convolutional activation function.
	 * @param activateRef reference to convolutional activation function.
	 * @return previous function reference.
	 */
	protected Function setConvActivateRef(Function convActivateRef) {
		return this.convActivateRef = convActivateRef;
	}

	
	/**
	 * Getting previous input value, which is for filtering by default.
	 * @return previous input value.
	 */
	protected abstract Matrix getPrevInput();

	
	/**
	 * Getting previous input as convolutional layer.
	 * @return previous input as convolutional layer.
	 */
	protected abstract ConvLayerSingle2D getPrevInputConvLayer();
	
	
	/**
	 * Setting previous input value, which is often for filtering by default.
	 * @param prevInput previous input value.
	 */
	protected abstract void setPrevInput(Matrix prevInput);

	
	/**
	 * Querying output by most, which can be previous input.
	 * @return output by most, which can be previous input.
	 */
	public Matrix queryInput() {
		Matrix input = getInput();
		return input != null ? input : getPrevInput();
	}

	
	/**
	 * Querying actual output by most, which can be previous input.
	 * @return actual output by most, which can be previous input.
	 */
	public Matrix queryActualInput() {
		if (containsWeights())
			return queryInput();
		else if (getFilter() != null) {
			Matrix prevInput = getPrevInput();
			return prevInput != null ? prevInput : queryInput();
		}
		else
			return queryInput();
	}
	
	
	/**
	 * Setting input value.
	 * @param input input value.
	 */
	protected abstract void setInput(Matrix input);

	
	/**
	 * Querying output by most, which can be previous input.
	 * @return output by most, which can be previous input.
	 */
	protected Matrix queryOutput() {
		Matrix output = getOutput();
		return output != null ? output : getPrevInput();
	}
	
	
	/**
	 * Setting output value.
	 * @param output output value.
	 */
	protected abstract void setOutput(Matrix output);

	
	/**
	 * Getting bias.
	 * @return bias.
	 */
	protected abstract Matrix getBias();
	
	
	/**
	 * Setting bias.
	 * @param bias specified bias.
	 */
	protected abstract void setBias(Matrix bias);


	/**
	 * Getting the first weight matrix.
	 * @return the first weight matrix.
	 */
	protected abstract Matrix getWeight1();
	
	
	/**
	 * Setting the first weight matrix.
	 * @param weight1 the first weight matrix.
	 */
	protected abstract void setWeight1(Matrix weight1);


	/**
	 * Getting the second weight matrix.
	 * @return the second weight matrix.
	 */
	protected abstract Matrix getWeight2();
	
	
	/**
	 * Setting the second weight matrix.
	 * @param weight2 the second weight matrix.
	 */
	protected abstract void setWeight2(Matrix weight2);


	/**
	 * Removing weights.
	 */
	protected abstract boolean removeWeights();

	
	/**
	 * Checking whether to contain weights.
	 * @return whether to contain weights.
	 */
	protected abstract boolean containsWeights();
	
	
	/**
	 * Getting convolutional filter.
	 * @return convolutional filter.
	 */
	protected abstract Filter2D getFilter();
	
	
	/**
	 * Setting filter.
	 * @param filter filter.
	 */
	protected abstract void setFilter(Filter2D filter);


	/**
	 * Getting convolutional filter bias.
	 * @return convolutional filter bias.
	 */
	protected abstract NeuronValue getFilterBias();
	
	
	/**
	 * Setting filter bias.
	 * @param filterBias specified filter bias.
	 */
	protected abstract void setFilterBias(NeuronValue filterBias);

	
	/**
	 * Removing filter.
	 * @return true if removal is successful.
	 */
	protected abstract boolean removeFilter();

	
	/**
	 * Getting size of this layer.
	 */
	public Dimension getSize() {
		Matrix output = queryOutput();
		return output != null ? new Dimension(output.columns(), output.rows()) : null; 
	}
	

	/**
	 * Getting size of this layer with regard to vectorization.
	 * @return size of this layer with regard to vectorization.
	 */
	public Dimension getSizeByVecRows() {
		Dimension size = getSize();
		return vecRows > 0 ? new Dimension(size.height/vecRows, vecRows) : size;
	}
	
	
	/**
	 * Getting matrix neural network.
	 * @return matrix neural network.
	 */
	MatrixNetworkAbstract getNetwork() {return network;}
	
	
	/**
	 * Setting matrix neural network.
	 * @param network matrix neural network.
	 */
	void setNetwork(MatrixNetworkAbstract network) {this.network = network;}
	
	
	/**
	 * Getting vectorization rows.
	 * @return Number of rows in case of vectorization. By default it is zero, which means that there is no vectorization by default.
	 */
	public int getVecRows() {
		return vecRows;
	}
	
	
	/**
	 * Checking whether to apply vectorization.
	 * @return whether to apply vectorization.
	 */
	public boolean isVectorized() {
		return vecRows > 0;
	}
	
	
	/**
	 * Setting vectorization rows.
	 * @param vecRows Number of rows in case of vectorization. By default it is zero, which means that there is no vectorization by default.
	 */
	void setVecRows(int vecRows) {
		vecRows = vecRows <= 0 ? 0 : vecRows;
		this.vecRows = vecRows;
	}

	
	/**
	 * Create raster from matrix.
	 * @param matrix matrix.
	 * @return raster.
	 */
	Raster toRaster(Matrix matrix) {
		matrix = isVectorized() ? matrix.vecInverse(vecRows) : matrix;
		return Matrix.toRaster(matrix, neuronChannel, paramIsNorm(), paramGetDefaultAlpha());
	}

	
	/**
	 * Extracting raster into matrix.
	 * @param raster raster.
	 * @param rows rows.
	 * @param columns columns.
	 * @return matrix.
	 */
	Matrix toMatrix(Raster raster, int rows, int columns) {
		if (!isVectorized()) return toMatrix0(raster, rows, columns);
		int vecRows = getVecRows();
		int vecColumns = rows / vecRows;
		if (vecRows <= 0 || vecColumns <= 0)
			return null;
		else
			return toMatrix0(raster, vecRows, vecColumns).vec();
	}
	
	
	/**
	 * Extracting raster into matrix.
	 * @param raster raster.
	 * @return matrix.
	 */
	public Matrix toMatrix(Raster raster) {
		Matrix input = getInput();
		return input != null ? toMatrix(raster, input.rows(), input.columns()) : null;
	}
	
	
	/**
	 * Extracting raster into matrix.
	 * @param raster raster.
	 * @param rows rows.
	 * @param columns columns.
	 * @return matrix.
	 */
	private Matrix toMatrix0(Raster raster, int rows, int columns) {
		Matrix ref = queryOutput().create(1, 1);
		return Matrix.toMatrix(rows, columns, raster, neuronChannel, paramIsNorm(), ref);
	}

	
	/**
	 * Converting convolutional layer to matrix.
	 * @param layer convolutional layer.
	 * @return matrix.
	 */
	Matrix convLayerToMatrix(ConvLayerSingle2D layer) {
		if (layer == null) return null;
		int rows = layer.getHeight();
		int columns = layer.getWidth();
		Matrix matrix = newMatrix(rows, columns);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				matrix.set(i, j, layer.get(j, i).getValue());
			}
		}
		return isVectorized() ? matrix.vec() : matrix;
	}
	
	
	/**
	 * Converting array to matrix.
	 * @param array array.
	 * @param rows rows.
	 * @param columns columns.
	 * @return matrix.
	 */
	Matrix arrayToMatrix(NeuronValue[] array, int rows, int columns) {
		Matrix matrix = newMatrix(rows, columns);
		for (int i = 0; i < rows; i++) {
			int rowLength = i*columns;
			for (int j = 0; j < columns; j++) {
				int index = rowLength + j;
				matrix.set(i, j, array[index]);
			}
		}
		return isVectorized() ? matrix.vec() : matrix;
	}
	
	
	/**
	 * Converting matrix to convolutional layer.
	 * @param matrix matrix.
	 * @return convolutional layer.
	 */
	ConvLayerSingle2D matrixToConvLayer(Matrix matrix) {
		if (matrix == null) return null;
		matrix = isVectorized() ? matrix.vecInverse(vecRows) : matrix;
		int rows = matrix.rows();
		int columns = matrix.columns();
		ConvLayerSingle2D layer = newConvLayer(columns, rows);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				ConvNeuron neuron = layer.get(j, i);
				neuron.setValue(matrix.get(i, j));
			}
		}
		return layer;
	}
	
	
	/**
	 * Checking whether filter is learned.
	 * @return whether filter is learned.
	 */
	boolean isLearnFilter() {
		return learnFilter;
	}
	
	
	/**
	 * Setting whether filter is learned.
	 * @param learnFilter flag to indicate whether filter is learned.
	 */
	void setLearnFilter(boolean learnFilter) {
		this.learnFilter = learnFilter;
	}
	
	
	/**
	 * Checking whether something normalized in rang [0, 1].
	 * @return whether something normalized in rang [0, 1].
	 */
	boolean paramIsNorm() {
		return network != null ? network.paramIsNorm() : Raster.NORM_DEFAULT;
	}


	/**
	 * Getting default alpha.
	 * @return default alpha.
	 */
	int paramGetDefaultAlpha() {
		return network != null ? network.paramGetDefaultAlpha() : Image.ALPHA_DEFAULT;
	}


}
