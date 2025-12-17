/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.io.Serializable;

import net.ea.ann.conv.Content;
import net.ea.ann.core.Id;
import net.ea.ann.core.LayerAbstract;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueCreator;
import net.ea.ann.mane.filter.Filter;
import net.ea.ann.mane.filter.FilterSpec;
import net.ea.ann.mane.filter.MaxPoolFilter;
import net.ea.ann.mane.filter.ProductFilter;
import net.ea.ann.raster.Image;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAbstract;
import net.ea.ann.raster.Size;

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
	 * This class contains size and filter.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static class LayerSpec implements Cloneable, Serializable {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;
		
		/**
		 * Previous size.
		 */
		public Size prevSize = null;
		
		/**
		 * Size.
		 */
		public Size size = null;
		
		/**
		 * Weight specification.
		 */
		public WeightSpec weightSpec = null;
		
		/**
		 * Filter.
		 */
		public FilterSpec filterSpec = null;
		
		/**
		 * Flag to allow vectorization.
		 */
		public int vecRows = 0;
		
		/**
		 * Constructor with filter specification.
		 * @param filterSpec filter specification.
		 */
		public LayerSpec(LayerSpec layerSpec) {
			this(layerSpec.size, layerSpec.filterSpec);
			this.vecRows = layerSpec.vecRows;
			this.weightSpec = layerSpec.weightSpec;
			this.prevSize = layerSpec.prevSize;
		}
	
		/**
		 * Constructor with size and weight specification.
		 * @param size size.
		 * @param weightSpec weight specification.
		 */
		public LayerSpec(Size size, WeightSpec weightSpec) {
			this.size = size;
			this.weightSpec = weightSpec;
		}

		/**
		 * Constructor with size and filter specification.
		 * @param size size.
		 * @param filterSpec filter specification.
		 */
		public LayerSpec(Size size, FilterSpec filterSpec) {
			this.size = size;
			this.filterSpec = filterSpec;
		}
	
		/**
		 * Constructor with size.
		 * @param size size.
		 */
		public LayerSpec(Size size) {
			this.size = size;
		}
	
		/**
		 * Default constructor.
		 */
		public LayerSpec() {
			
		}
		
		/**
		 * Getting whether to be vectorization.
		 * @return whether to be vectorization.
		 */
		public boolean isVectorized() {return vecRows > 0;}
		
		/**
		 * Getting size of this layer with regard to vectorization.
		 * @return size of this layer with regard to vectorization.
		 */
		public Size sizeByVecRows() {
			return vecRows > 0 ? new Size(size.height/vecRows, vecRows, size.depth, size.time) : size;
		}
	}


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
	private boolean learnFilter = LEARN_FILTER_DEFAULT;
	
	
	/**
	 * Reference to matrix neural network.
	 */
	private MatrixNetworkAbstract network = null;
	
	
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
		NeuronValue value = newMatrix(new Size(1, 1, 1, 1)).get(0, 0);
		if (value instanceof Matrix)
			return value;
		else if (value instanceof Content)
			return value;
		else
			return value.zero();
	}

	
	/**
	 * Creating matrix.
	 * @param size size.
	 * @return new matrix.
	 */
	protected Matrix newMatrix(Size size) {
		Matrix output = queryOutput();
		if (output != null) return output.create(size);
		NeuronValue value = NeuronValueCreator.newNeuronValue(neuronChannel);
		return MatrixUtil.create(size, value);
	}


	/**
	 * Constructor with the first weight size and the second weight size.
	 * @param sizeW1 the first weight size.
	 * @param sizeW2 the second weight size.
	 * @param layerSpec layer specification, which can be null.
	 */
	protected Weight newWeight(Size sizeW1, Size sizeW2, LayerSpec layerSpec) {
		return WeightImpl.create(sizeW1, sizeW2, newNeuronValue());
	}

	
	/**
	 * Creating filter.
	 * @param filterSize filter size.
	 * @param layerSpec layer specification, which can be null.
	 * @return filter.
	 */
	protected Filter newFilter(Size filterSize, LayerSpec layerSpec) {
		return newFilter(filterSize, layerSpec.filterSpec, newNeuronValue());
	}
	
	
	/**
	 * Creating default filter.
	 * @param filterSize filter size.
	 * @param hint matrix hint.
	 * @return default filter.
	 */
	private static Filter newFilter(Size filterSize, FilterSpec filterSpec, NeuronValue hint) {
		if (filterSize == null || filterSize.width <= 0 || filterSize.height <= 0) return null;
		double factor = 1.0 / (filterSize.width*filterSize.height);
		Filter filter = null;
		switch (filterSpec.type) {
			case kernel:
				filter = ProductFilter.create(factor, filterSize, hint);
				filter.setMoveStride(false);
				break;
			case pool:
				Size adjustedSize = new Size(filterSize.width, filterSize.height, filterSize.time, 1);
				filter = MaxPoolFilter.create(adjustedSize);
				filter.setMoveStride(true);
				break;
			default:
				break;
		}
		return filter;
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
	 * Setting previous input value, which is for filtering by default.
	 * @param prevInput previous input value.
	 */
	protected abstract void setPrevInput(Matrix prevInput);

	
	/**
	 * Getting previous output value, which is for filtering by default.
	 * @return previous output value.
	 */
	protected abstract Matrix getPrevOutput();

	
	/**
	 * Setting previous output value, which is for filtering by default.
	 * @param prevOutput previous output value.
	 */
	protected abstract void setPrevOutput(Matrix prevOutput);

	
	/**
	 * Querying output by most, which can be previous input.
	 * @return output by most, which can be previous input.
	 */
	public Matrix queryInput() {
		Matrix input = getInput();
		return input != null ? input : getPrevInput();
	}

	
	/**
	 * Querying actual input by most (right-most), which can be previous input.
	 * @return actual input by most (right-most), which can be previous input.
	 */
	public Matrix queryActualInput() {
		if (getWeight() != null)
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
		output = output != null ? output : getPrevOutput();
		return output != null ? output : queryInput();
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
	 * Getting weight.
	 * @return the weight.
	 */
	protected abstract Weight getWeight();
	
	
	/**
	 * Setting the weight.
	 * @param weight the weight.
	 */
	protected abstract void setWeight(Weight weight);


	/**
	 * Removing weights.
	 * @param layerSpec layer specification which can be null.
	 */
	protected abstract boolean removeWeights(MatrixLayerAbstract.LayerSpec layerSpec);

	
	/**
	 * Getting convolutional filter.
	 * @return convolutional filter.
	 */
	protected abstract Filter getFilter();
	
	
	/**
	 * Setting filter.
	 * @param filter filter.
	 */
	protected abstract void setFilter(Filter filter);


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
	public Size getSize() {
		Matrix output = queryOutput();
		int depth = output instanceof MatrixStack ? ((MatrixStack)output).depth() : 1;
		return output != null ? new Size(output.columns(), output.rows(), depth) : null; 
	}
	

	/**
	 * Getting size of this layer with regard to vectorization.
	 * @return size of this layer with regard to vectorization.
	 */
	public Size getSizeByVecRows() {
		Size size = getSize();
		return vecRows > 0 ? new Size(size.height/vecRows, vecRows, size.depth) : size;
	}
	
	
	/**
	 * Getting matrix neural network.
	 * @return matrix neural network.
	 */
	public MatrixNetworkAbstract getNetwork() {return network;}
	
	
	/**
	 * Setting matrix neural network.
	 * @param network matrix neural network.
	 */
	public void setNetwork(MatrixNetworkAbstract network) {this.network = network;}
	
	
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
		return MatrixUtil.toRaster(matrix, neuronChannel, paramIsNorm(), paramGetDefaultAlpha());
	}

	
	/**
	 * Extracting raster into matrix.
	 * @param raster raster.
	 * @param size size.
	 * @return matrix.
	 */
	Matrix toMatrix(Raster raster, Size size) {
		if (!isVectorized()) return toMatrix0(raster, size);
		int vecRows = getVecRows();
		int vecColumns = size.height / vecRows;
		if (vecRows <= 0 || vecColumns <= 0)
			return null;
		else
			return toMatrix0(raster, new Size(vecColumns, vecRows)).vec();
	}
	
	
	/**
	 * Extracting raster into matrix.
	 * @param raster raster.
	 * @return matrix.
	 */
	public Matrix toMatrix(Raster raster) {
		Matrix input = getInput();
		return input != null ? toMatrix(raster, new Size(input.columns(), input.rows())) : null;
	}
	
	
	/**
	 * Extracting raster into matrix.
	 * @param raster raster.
	 * @param size size.
	 * @return matrix.
	 */
	private Matrix toMatrix0(Raster raster, Size size) {
		Matrix ref = queryOutput().create(size);
		int rasterChannel = getNetwork() != null ? getNetwork().paramGetRasterChannel() : RasterAbstract.RASTER_CHANNEL_DEFAULT;
		return MatrixUtil.toMatrix(size, raster, neuronChannel, rasterChannel, paramIsNorm(), ref);
	}

	
	/**
	 * Converting convolutional layer to matrix.
	 * @param convLayer convolutional layer.
	 * @return matrix.
	 */
	Matrix convLayerToMatrix(Matrix convLayer) {
		return convLayer != null && isVectorized() ? convLayer.vec() : convLayer;
	}
	
	
	/**
	 * Converting matrix to convolutional layer.
	 * @param matrix matrix.
	 * @return convolutional layer.
	 */
	Matrix matrixToConvLayer(Matrix matrix) {
		return matrix != null && isVectorized() ? matrix.vecInverse(vecRows) : matrix;
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
