/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import java.awt.Dimension;
import java.awt.Point;
import java.awt.Rectangle;

import net.ea.ann.conv.filter.BiasFilter;
import net.ea.ann.conv.filter.DeconvConvFilter;
import net.ea.ann.conv.filter.DeconvConvFilter1D;
import net.ea.ann.conv.filter.DeconvFilter;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.conv.filter.Filter1D;
import net.ea.ann.conv.filter.ProductFilter1D;
import net.ea.ann.core.Id;
import net.ea.ann.core.LayerAbstract;
import net.ea.ann.core.Network;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.NeuronRaster;
import net.ea.ann.raster.NeuronValueRaster;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.Size;

/**
 * This class is an abstract implementation of convolutional layer in 2D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class ConvLayer1DAbstract extends LayerAbstract implements ConvLayerSingle1D {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Neuron channel or depth.
	 */
	protected int neuronChannel = 1;
	
	
	/**
	 * Activation function reference.
	 */
	protected Function activateRef = null;

	
	/**
	 * Raster width.
	 */
	protected int width = 1;
	
	
	/**
	 * Internal filter.
	 */
	protected Filter filter = null;
	
	
	/**
	 * Internal bias associated with filter.
	 */
	protected NeuronValue bias = null;
	
	
	/**
	 * Internal array of neurons.
	 */
	protected ConvNeuron[] neurons = null;
	

	/**
	 * Previous layer.
	 */
	protected ConvLayer prevLayer = null;
	
	
	/**
	 * Next layer.
	 */
	protected ConvLayer nextLayer = null;

	
	/**
	 * Flag to indicate whether to pad zero when filtering.
	 */
	protected boolean isPadZeroFilter = false;
	
	
	/**
	 * Constructor with neuron channel, activation function, width, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	protected ConvLayer1DAbstract(int neuronChannel, Function activateRef, int width, Filter filter, Id idRef) {
		this(neuronChannel, activateRef, filter, idRef);
		
		this.width = width;
		this.neurons = new ConvNeuron[width];
		NeuronValue zero = this.newNeuronValue().zero();
		for (int index = 0; index < width; index++) {
			ConvNeuron neuron = this.newNeuron();
			neuron.setValue(zero);
			this.neurons[index] = neuron;
		}
	}

	
	/**
	 * Constructor with neuron channel, activation function, width, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param filter kernel filter.
	 */
	protected ConvLayer1DAbstract(int neuronChannel, Function activateRef, int width, Filter filter) {
		this(neuronChannel, activateRef, width, filter, null);
	}

	
	/**
	 * Constructor with neuron channel, activation function, and width.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 */
	protected ConvLayer1DAbstract(int neuronChannel, Function activateRef, int width) {
		this(neuronChannel, activateRef, width, null, null);
	}

	
	/**
	 * Default constructor with neuron channel, activation function, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	ConvLayer1DAbstract(int neuronChannel, Function activateRef, Filter filter, Id idRef) {
		super(idRef);
		this.neuronChannel = neuronChannel = (neuronChannel < 1 ? 1 : neuronChannel);
		this.activateRef = activateRef == null ? (activateRef = Raster.toConvActivationRef(this.neuronChannel, true)) : activateRef;

		this.neurons = new ConvNeuron[1];
		NeuronValue zero = this.newNeuronValue().zero();
		ConvNeuron neuron = this.newNeuron();
		neuron.setValue(zero);
		this.neurons[0] = neuron;
		
		this.filter = filter;
		this.bias = zero;
	}


	@Override
	public ConvNeuron newNeuron() {
		return new ConvNeuronImpl(this);
	}


	@Override
	public int getNeuronChannel() {
		return neuronChannel;
	}


	@Override
	public int getWidth() {
		return width;
	}


	@Override
	public int getHeight() {
		return 1;
	}

	
	@Override
	public int getDepth() {
		return 1;
	}


	@Override
	public int getTime() {
		return 1;
	}


	@Override
	public Filter getFilter() {
		return filter;
	}


	@Override
	public Filter1D getFilter1D() {
		if (filter == null)
			return null;
		else if (filter instanceof Filter1D)
			return (Filter1D)filter;
		else
			return null;
	}

	
	@Override
	public Filter setFilter(Filter filter) {
		Filter prevFilter = this.filter;
		this.filter = filter;
		return prevFilter;
	}


	@Override
	public NeuronValue getBias() {
		return bias;
	}


	@Override
	public boolean setBias(NeuronValue bias) {
		if (bias != null) {
			this.bias = bias;
			return true;
		}
		else
			return false;
	}
	
	
	@Override
	public ConvNeuron get(int index) {
		return neurons[index];
	}


	@Override
	public NeuronValue set(int index, NeuronValue value) {
		ConvNeuron neuron = neurons[index];
		if (neuron == null)
			return null;
		else {
			NeuronValue prevValue = neuron.getValue();
			neuron.setValue(value);
			return prevValue;
		}
	}


	@Override
	public boolean isPadZeroFilter() {
		return isPadZeroFilter;
	}


	@Override
	public void setPadZeroFilter(boolean isPadZeroFilter) {
		this.isPadZeroFilter = isPadZeroFilter;
	}
	
	
	@Override
	public int length() {
		return neurons != null ? neurons.length : 0;
	}


	@Override
	public ConvNeuron[] getNeurons() {
		return neurons;
	}


	@Override
	public NeuronValue[] getData() {
		if (neurons == null || neurons.length == 0) return null;
		
		NeuronValue[] data = new NeuronValue[neurons.length];
		for (int i = 0; i < neurons.length; i++) {
			NeuronValue value = neurons[i].getValue();
			data[i] = value;
		}
		
		return data;
	}

	
	/**
	 * Getting data from specified region.
	 * @param region specified region.
	 * @return data extracted from specified region.
	 */
	protected NeuronValue[] getData(Rectangle region) {
		if (region == null) return getData();
		int width = getWidth();
		
		region.x = region.x < 0 ? 0 : region.x;
		region.width = region.x + region.width <= width ? region.width : width - region.x;
		if (region.width <= 0) return null;
		
		int regionIndex = 0;
		NeuronValue[] data = new NeuronValue[region.width];
		int xwidth = region.x + region.width;
		for (int x = region.x; x < xwidth; x++) {
			data[regionIndex] = neurons[x].getValue();
			regionIndex++;
		}
		
		return data;
	}

	
	@Override
	public NeuronValue[] setData(NeuronValue[] data) {
		if (data == null || neurons.length == 0) return null;
		
		data = NeuronValue.adjustArray(data, neurons.length, this);
		for (int i = 0; i < neurons.length; i++) {
			neurons[i].setValue(data[i]);
		}
		return data;
	}
	
	
	/**
	 * Setting data at specified region.
	 * @param data data as array of neuron value.
	 * @param region specified region.
	 * @return data to be set.
	 */
	protected NeuronValue[] setData(NeuronValue[] data, Rectangle region) {
		if (region == null) setData(data);
		if (data == null || neurons.length == 0) return null;
		int width = getWidth();
		
		region.x = region.x < 0 ? 0 : region.x;
		region.width = region.x + region.width <= width ? region.width : width - region.x;
		if (region.width <= 0) return null;
		
		int regionIndex = 0;
		data = NeuronValue.adjustArray(data, region.width, this);
		int xwidth = region.x + region.width;
		for (int x = region.x; x < xwidth; x++) {
			neurons[x].setValue(data[regionIndex]);
			regionIndex++;
		}

		return data;
	}

	
	@Override
	public ConvLayer getPrevLayer() {
		return prevLayer;
	}


	@Override
	public ConvLayer getNextLayer() {
		return nextLayer;
	}


	@Override
	public boolean setNextLayer(ConvLayer nextLayer) {
		if (nextLayer == this.nextLayer) return false;

		ConvLayer oldNextLayer = this.nextLayer;
		ConvLayer oldNextNextLayer = null;
		if (oldNextLayer != null) oldNextNextLayer = oldNextLayer.getNextLayer();

		this.nextLayer = nextLayer;
		if (nextLayer == null) return true;

		((ConvLayer1DAbstract)nextLayer).prevLayer = this;
		
		if (oldNextNextLayer == null) return true;
		((ConvLayer1DAbstract)oldNextNextLayer).prevLayer = nextLayer;
		((ConvLayer1DAbstract)nextLayer).nextLayer = oldNextNextLayer;

		return true;
	}


	@Override
	public Function getActivateRef() {
		return this.activateRef;
	}

	
	@Override
	public Function setActivateRef(Function activateRef) {
		return this.activateRef = activateRef;
	}

	
	@Override
	public Raster createRaster(NeuronValue[] values,
			boolean isNorm, int defaultAlpha) {
		return RasterAssoc.createRaster(this, values, isNorm, defaultAlpha);
	}

	
	/**
	 * Getting the region in next layer, corresponding to the current neuron at specified coordination at current layer.
	 * @param x X coordination.
	 * @return the region in next layer, corresponding to the current neuron at specified coordination at current layer.
	 */
	protected Rectangle getNextRegion(int x) {
		if ((nextLayer == null) || !(nextLayer instanceof ConvLayerSingle)) return null;
		Filter filter = getFilter();
		if (filter == null) return null;
		
		int filterStrideWidth = filter.getStrideWidth();
		int nextWidth = ((ConvLayerSingle)nextLayer).getWidth();
		
		Rectangle nextRegion = new Rectangle(new Point(0, 0), new Dimension(0, 0));
		if (filter instanceof DeconvFilter) {
			nextRegion.x = x * filterStrideWidth;
			nextRegion.width = filterStrideWidth;
		}
		else {
			nextRegion.x = x / filterStrideWidth;
			nextRegion.width = 1;
		}
		
		nextRegion.width = nextRegion.x + nextRegion.width <= nextWidth ? nextRegion.width : nextWidth - nextRegion.x;
		if (nextRegion.width <= 0)
			return null;
		else
			return nextRegion;
	}

	
	/**
	 * Getting the next region of this region.
	 * @param thisRegion this region.
	 * @return the next region of this region.
	 */
	protected Rectangle getNextRegion(Rectangle thisRegion) {
		thisRegion = thisRegion != null ? thisRegion : new Rectangle(getWidth(), 0);
		if ((nextLayer == null) || !(nextLayer instanceof ConvLayerSingle)) return null;
		Filter filter = getFilter();
		if (filter == null) return null;
		
		int filterStrideWidth = filter.getStrideWidth();
		int nextWidth = ((ConvLayerSingle)nextLayer).getWidth();
		
		Rectangle nextRegion = new Rectangle(new Point(0, 0), new Dimension(0, 0));
		if (filter instanceof DeconvFilter) {
			nextRegion.x = thisRegion.x * filterStrideWidth;
			nextRegion.width = thisRegion.width * filterStrideWidth;
		}
		else {
			nextRegion.x = thisRegion.x / filterStrideWidth;
			nextRegion.width = thisRegion.width / filterStrideWidth;
			nextRegion.width = nextRegion.width < 1 ? 1 : nextRegion.width;
		}
		
		nextRegion.width = nextRegion.x + nextRegion.width <= nextWidth ? nextRegion.width : nextWidth - nextRegion.x;
		if (nextRegion.width <= 0)
			return null;
		else
			return nextRegion;
	}

	
	/**
	 * Getting the region in next layer, corresponding to the current neuron at specified coordination at current layer.
	 * @param x X coordination.
	 * @return the region in next layer, corresponding to the current neuron at specified coordination at current layer.
	 */
	protected Rectangle getPrevRegion(int x) {
		if ((this.prevLayer == null) || !(this.prevLayer instanceof ConvLayerSingle)) return null;
		ConvLayerSingle prevLayer = (ConvLayerSingle)this.prevLayer;
		Filter filter = prevLayer.getFilter();
		if (filter == null) return null;
		
		int filterStrideWidth = filter.getStrideWidth();
		int prevWidth = prevLayer.getWidth();
		int prevBlockWidth = filter.isMoveStride() ? prevWidth / filterStrideWidth : prevWidth;
		
		Rectangle prevRegion = new Rectangle(new Point(0, 0), new Dimension(0, 0));
		if (filter instanceof DeconvFilter) {
			prevRegion.x = x / filterStrideWidth;
			prevRegion.x = prevRegion.x < prevWidth ? prevRegion.x : prevWidth-1;

			prevRegion.width = 1;
		}
		else {
			int xBlock = x < prevBlockWidth ? x : prevBlockWidth-1;
			prevRegion.x = xBlock*filterStrideWidth;
			
			prevRegion.width = filterStrideWidth;
		}
		
		prevRegion.width = prevRegion.x + prevRegion.width <= prevWidth ? prevRegion.width : prevWidth - prevRegion.x;
		if (prevRegion.width <= 0)
			return null;
		else
			return prevRegion;
	}

	
	/**
	 * Getting the previous region of this region.
	 * @param thisRegion this region.
	 * @return the previous region of this region.
	 */
	protected Rectangle getPrevRegion(Rectangle thisRegion) {
		thisRegion = thisRegion != null ? thisRegion : new Rectangle(getWidth(), 0);
		if ((this.prevLayer == null) || !(this.prevLayer instanceof ConvLayerSingle)) return null;
		ConvLayerSingle prevLayer = (ConvLayerSingle)this.prevLayer;
		Filter filter = prevLayer.getFilter();
		if (filter == null) return null;
		
		int filterStrideWidth = filter.getStrideWidth();
		int prevWidth = prevLayer.getWidth();
		int prevBlockWidth = filter.isMoveStride() ? prevWidth / filterStrideWidth : prevWidth;
		
		Rectangle prevRegion = new Rectangle(new Point(0, 0), new Dimension(0, 0));
		if (filter instanceof DeconvFilter) {
			prevRegion.x = thisRegion.x / filterStrideWidth;
			prevRegion.x = prevRegion.x < prevWidth ? prevRegion.x : prevWidth-1;

			prevRegion.width = thisRegion.width / filterStrideWidth;
			prevRegion.width = prevRegion.width < 1 ? 1 : prevRegion.width;
		}
		else {
			int xBlock = thisRegion.x < prevBlockWidth ? thisRegion.x : prevBlockWidth-1;
			prevRegion.x = xBlock*filterStrideWidth;
			
			prevRegion.width = thisRegion.width*filterStrideWidth;
		}
		
		prevRegion.width = prevRegion.x + prevRegion.width <= prevWidth ? prevRegion.width : prevWidth - prevRegion.x;
		if (prevRegion.width <= 0)
			return null;
		else
			return prevRegion;
	}

	
	@Override
	public ConvLayer forward() {
		ConvLayer nextLayer = getNextLayer();
		if ((nextLayer == null) || !(nextLayer instanceof ConvLayerSingle1D)) return null;
		NeuronRaster result = forward(this, (ConvLayerSingle1D)nextLayer, getFilter(), null, null, true);
		return result != null ? nextLayer : null;
	}


	@Override
	public ConvLayerSingle forward(ConvLayerSingle nextLayer, Filter filter) {
		NeuronRaster result = forward(this, (ConvLayerSingle1D)nextLayer, filter, null, null, true);
		return result != null ? nextLayer : null;
	}


	@Override
	public NeuronValue[][] dKernel(ConvLayerSingle nextError, Filter filter) {
		return dKernel(this, (ConvLayerSingle1D)nextError, filter, null, null);
	}


	@Override
	public NeuronValueRaster dValue(ConvLayerSingle nextError, Filter filter) {
		return dValue(this, (ConvLayerSingle1D)nextError, filter, null, null);
	}


	/**
	 * Forwarding evaluation from current layer to next layer.
	 * @param thisLayer current layer.
	 * @param nextLayer next layer.
	 * @param f filter of current layer.
	 * @param thisFilterRegion filtering region of current layer. It can be null.
	 * @param nextFilterRegion filtering region of next layer. It can be null.
	 * @param nextAffected flag to indicate whether the next layer is affected.
	 * @return arrays of neurons filtered.
	 */
	static NeuronRaster forward(ConvLayerSingle1D thisLayer, ConvLayerSingle1D nextLayer, Filter f, Rectangle thisFilterRegion, Rectangle nextFilterRegion, boolean nextAffected) {
		if (thisLayer == null) return null;
		Filter1D filter = (f != null && f instanceof Filter1D) ? (Filter1D)f : thisLayer.getFilter1D();
		if (filter == null) return null;

		NeuronValue nextZero = nextLayer != null ? nextLayer.newNeuronValue().zero() : thisLayer.newNeuronValue().zero();
		ConvNeuron[] nextNeurons = nextLayer.getNeurons();
		if (nextAffected)
			nextNeurons = nextLayer.getNeurons();
		else {
			nextNeurons = new ConvNeuron[nextLayer.length()];
			for (int i = 0; i < nextNeurons.length; i++) {
				nextNeurons[i].setValue(nextZero);
				nextNeurons[i].setInput(nextZero);
			}
		}
		if (nextNeurons == null || nextNeurons.length == 0) return null;
		
		if (filter instanceof DeconvConvFilter) {
			for (ConvNeuron nextNeuron : nextNeurons) {
				nextNeuron.setValue(null);
				nextNeuron.setInput(null);
			}
		}
		
		if (thisFilterRegion != null && nextFilterRegion != null) nextFilterRegion = null;
		
		int filterStrideWidth = filter.getStrideWidth();
		int thisWidth = thisLayer.getWidth();
		int thisBlockWidth = filter.isMoveStride() ? thisWidth / filterStrideWidth : thisWidth;
		int nextWidth = nextLayer.getWidth();
		Function activateRef = nextLayer.getActivateRef();
		activateRef = activateRef == null ? thisLayer.getActivateRef() : activateRef;
		
		for (int nextX = 0; nextX < nextWidth; nextX++) {
			int thisX = 0;
			if (filter instanceof DeconvFilter) {
				thisX = nextX / filterStrideWidth;
				if (!nextLayer.isPadZeroFilter()) thisX = thisX < thisWidth ? thisX : thisWidth-1;
			}
			else {
				int xBlock = nextLayer.isPadZeroFilter() ? nextX : (nextX < thisBlockWidth ? nextX : thisBlockWidth-1);
				thisX = xBlock*filterStrideWidth;
			}
			
			//Setting zero to outline pixels.
			if (thisX >= thisWidth) {
				nextNeurons[nextX].setValue(nextZero);
				continue;
			}
			
			//Checking region.
			if (thisFilterRegion != null && (thisX < thisFilterRegion.x || thisFilterRegion.x + thisFilterRegion.width <= thisX))
				continue;
			if (nextFilterRegion != null && (nextX < nextFilterRegion.x || nextFilterRegion.x + nextFilterRegion.width <= nextX))
				continue;
			
			//Filtering
			NeuronValue filteredValue = null;
			if (filter instanceof DeconvConvFilter)
				filteredValue = ((DeconvConvFilter1D)filter).apply(thisX, thisLayer, nextX, nextLayer);
			else
				filteredValue = filter.apply(thisX, thisLayer);
			
			if (filteredValue != null) {
				filteredValue = filteredValue.add(thisLayer.getBias());
				nextNeurons[nextX].setInput(filteredValue);
				if (activateRef != null) filteredValue = activateRef.evaluate(filteredValue);
				nextNeurons[nextX].setValue(filteredValue);
			}
			else
				nextNeurons[nextX].setValue(nextZero);
		}
		
		if (filter instanceof DeconvConvFilter) {
			for (ConvNeuron nextNeuron : nextNeurons) {
				if (nextNeuron.getValue() == null) nextNeuron.setValue(nextZero);
				if (nextNeuron.getInput() == null) nextNeuron.setInput(nextZero);
			}
		}

		if ((!(thisLayer instanceof ConvLayer1DAbstract)) || (thisFilterRegion == null && nextFilterRegion == null))
			return new NeuronRaster(nextLayer.getNeuronChannel(), nextNeurons, new Size(nextWidth, 1, 1, 1));
		
		Rectangle nextRegion = null;
		if (thisFilterRegion != null)
			nextRegion = ((ConvLayer1DAbstract)thisLayer).getNextRegion(thisFilterRegion);
		else
			nextRegion = nextFilterRegion;
		if (nextRegion == null) return new NeuronRaster(nextLayer.getNeuronChannel(), nextNeurons, new Size(nextWidth, 1, 1, 1));
		
		ConvNeuron[] regionNeurons = new ConvNeuron[nextRegion.width];
		int regionIndex = 0;
		for (int nextX = nextRegion.x; nextX < nextRegion.x + nextRegion.width; nextX++) {
			regionNeurons[regionIndex] = nextNeurons[nextX];
			regionIndex++;
		}
		return new NeuronRaster(nextLayer.getNeuronChannel(), regionNeurons, new Size(nextRegion.width, 1, 1, 1));
	}

		
	/**
	 * Calculating derivative of this filter given next layer as bias layer at specified coordinator.
	 * @param thisLayerSize current layer.
	 * @param nextLayer next layer.
	 * @param f filter of current layer.
	 * @param thisFilterRegion filtering region of current layer. It can be null.
	 * @param nextFilterRegion filtering region of next layer. It can be null.
	 * @return differentials of kernel.
	 */
	static NeuronValue[][] dKernel(ConvLayerSingle1D thisLayer, ConvLayerSingle1D nextLayer, Filter f, Rectangle thisFilterRegion, Rectangle nextFilterRegion) {
		throw new RuntimeException("Method ConvLayer1DAbstract::dKernel(ConvLayerSingle1D, ConvLayerSingle1D, Filter, Rectangle, Rectangle) not implmented yet");
	}
	
	
	/**
	 * Calculating derivative of this layer given next layer as bias layer at specified coordinator.
	 * @param thisLayer current layer.
	 * @param nextLayer next layer.
	 * @param f filter of current layer.
	 * @param thisFilterRegion filtering region of current layer. It can be null.
	 * @param nextFilterRegion filtering region of next layer. It can be null.
	 * @return arrays of values filtered.
	 */
	static NeuronValueRaster dValue(ConvLayerSingle1D thisLayer, ConvLayerSingle1D nextLayer, Filter f, Rectangle thisFilterRegion, Rectangle nextFilterRegion) {
		throw new RuntimeException("Method ConvLayer1DAbstract::dValue(ConvLayerSingle1D, ConvLayerSingle1D, Filter, Rectangle, Rectangle) not implmented yet");
	}
	
	
	/**
	 * Forwarding evaluation from current layer to next layer.
	 * @param input input data. It can be null.
	 * @param thisLayer current layer.
	 * @param nextLayer next layer.
	 * @param f filter of current layer.
	 * @param thisFilterRegion filtering region of current layer. It can be null.
	 * @param nextFilterRegion filtering region of next layer. It can be null.
	 * @param nextAffected flag to indicate whether the next layer is affected
	 * @return arrays of neurons filtered.
	 */
	public static NeuronRaster forward(NeuronValue[] input, ConvLayerSingle1D thisLayer, ConvLayerSingle1D nextLayer, Filter f, Rectangle thisFilterRegion, Rectangle nextFilterRegion, boolean nextAffected) {
		if ((input != null) && (thisLayer instanceof ConvLayer1DAbstract)) ((ConvLayer1DAbstract)thisLayer).setData(input);
		return forward(thisLayer, nextLayer, f, thisFilterRegion, nextFilterRegion, nextAffected);
	}

	
	/**
	 * Learning filter from this layer and next layer.
	 * @return filter learned from this layer and next layer.
	 */
	public BiasFilter learnFilter() {
		return learnFilter(null, true, 1, 1);
	}
	
	
	@Override
	public BiasFilter learnFilter(BiasFilter initialFilter, boolean learningBias, double learningRate, int maxIteration) {
		if (nextLayer == null) return null;
		ConvLayerSingle1D smallLayer = null, largeLayer = null;
		if (this.length() >= ((ConvLayerSingle)nextLayer).length()) {
			smallLayer = (ConvLayerSingle1D)nextLayer;
			largeLayer = this;
		}
		else {
			smallLayer = this;
			largeLayer = (ConvLayerSingle1D)nextLayer;
		}
		
		maxIteration = maxIteration > 0 ? maxIteration : Network.LEARN_MAX_ITERATION_MAX;
		int iteration = 0;
		BiasFilter filter = initialFilter;
		while (iteration < maxIteration) {
			double lr = NetworkAbstract.calcLearningRate(learningRate, iteration, NetworkAbstract.LEARN_RATE_FIXED_DEFAULT);
			filter = learnFilter(smallLayer, largeLayer, filter, learningBias, lr);
			iteration++;
		}
		
		return filter;
	}

	
	/**
	 * Learning product filter from large layer to small layer.
	 * @param smallLayer specified small layer.
	 * @param largeLayer specified large layer.
	 * @param initialFilter initial filter. This is initial filter of larger layer.
	 * @param learningBias flag to indicate whether to lean bias.
	 * @param learningRate learning rate.
	 * @return filter learned from large layer to small layer. It is filter stored in large layer.
	 */
	static BiasFilter learnFilter(ConvLayerSingle1D smallLayer, ConvLayerSingle1D largeLayer, BiasFilter initialFilter, boolean learningBias, double learningRate)  {
		if (smallLayer == null || largeLayer == null) return null;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? 1 : learningRate;

		ConvLayerSingle1D thisLayer = largeLayer, nextLayer = smallLayer;
		int n = Math.min(largeLayer.getWidth()/smallLayer.getWidth(), largeLayer.getHeight()/smallLayer.getHeight());
		boolean isMoveStride = true;
		if (n <= 1) {
			n = 3;
			isMoveStride = false;
		}
		
		NeuronValue zero = thisLayer.newNeuronValue().zero();
		ProductFilter1D filter = null;
		ProductFilter1D initialProductFilter = ((initialFilter != null) && (initialFilter.filter != null) && (initialFilter.filter instanceof ProductFilter1D)) ?
			(ProductFilter1D)(initialFilter.filter) : null;
		if (initialProductFilter == null || initialProductFilter.width() != n) {
			NeuronValue[] kernel = new NeuronValue[n];
			for (int j = 0; j < n; j++) kernel[j] = zero;
			filter = ProductFilter1D.create(kernel, zero.unit());
		}
		else {
			NeuronValue[] kernel = new NeuronValue[n];
			for (int j = 0; j < n; j++) kernel[j] = initialProductFilter.getKernel()[j];
			filter = ProductFilter1D.create(kernel, initialProductFilter.getWeight());
		}
		
		filter.setMoveStride(isMoveStride);
		NeuronValue[] kernel = filter.getKernel();
		
		NeuronValue bias = null;
		if ((initialFilter != null) && (initialFilter.bias != null))
			bias = initialFilter.bias;
		else
			bias = zero; 
		
		int filterStrideWidth = filter.getStrideWidth();
		int thisWidth = thisLayer.getWidth();
		int thisBlockWidth = filter.isMoveStride() ? thisWidth / filterStrideWidth : thisWidth;
		int nextWidth = nextLayer.getWidth();
	
		for (int nextX = 0; nextX < nextWidth-1; nextX++) { //Please pay attention to minus one (-1) which prevents overlapping estimation.
			int thisX = 0;
			int xBlock = nextX < thisBlockWidth ? nextX : thisBlockWidth-1;
			thisX = xBlock*filterStrideWidth;
			
			NeuronValue filteredNextValue = filter.apply(thisX, thisLayer);
			if (filteredNextValue == null) continue;
			Function nextActivateRef = nextLayer.getActivateRef(); 
			if (nextActivateRef != null) filteredNextValue = nextActivateRef.evaluate(filteredNextValue); 
			
			NeuronValue realNextValue = nextLayer.get(nextX).getValue();
			NeuronValue error = realNextValue.subtract(filteredNextValue);
			if (nextActivateRef != null)
				error = error.multiplyDerivative(nextActivateRef.derivative(filteredNextValue));
			
			//Learning kernel.
			for (int j = 0; j < n; j++) {
				NeuronValue thisValue = thisLayer.get(thisX+j).getValue();
				NeuronValue delta = error.multiply(thisValue).multiply(learningRate);
				kernel[j] = kernel[j].add(delta);
			}
			
			//Learning bias.
			if (learningBias) bias = bias.add(error.multiply(learningRate));
		}
	
		return new BiasFilter(filter, learningBias?bias:null);
	}
	
	
}
