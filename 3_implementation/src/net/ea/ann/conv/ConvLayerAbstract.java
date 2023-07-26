/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import net.ea.ann.conv.filter.DeconvFilter;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.core.Id;
import net.ea.ann.core.LayerAbstract;
import net.ea.ann.core.NeuronValue;

/**
 * This class is an abstract implementation of convolutional layer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class ConvLayerAbstract extends LayerAbstract implements ConvLayer {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Neuron channel or depth.
	 */
	protected int neuronChannel = 1;
	
	
	/**
	 * Raster width.
	 */
	protected int width = 0;
	
	
	/**
	 * Raster height.
	 */
	protected int height = 0;
	
	
	/**
	 * Internal filter.
	 */
	protected Filter filter = null;
	
	
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
	 * Constructor with neuron channel, width, height, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param width layer width.
	 * @param height layer height.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	protected ConvLayerAbstract(int neuronChannel, int width, int height, Filter filter, Id idRef) {
		super(idRef);

		this.neuronChannel = neuronChannel;
		this.width = width;
		this.height = height;
		this.filter = filter;
		
		this.neurons = new ConvNeuron[width*height];
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int index = y*width + x;
				ConvNeuron neuron = this.newNeuron();
				neuron.setValue(this.newNeuronValue());
				
				this.neurons[index] = neuron;
			}
		}
	}


	/**
	 * Constructor with neuron channel, width, height, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param width layer width.
	 * @param height layer height.
	 * @param filter kernel filter.
	 */
	protected ConvLayerAbstract(int neuronChannel, int width, int height, Filter filter) {
		this(neuronChannel, width, height, filter, null);
	}


	@Override
	public ConvNeuron newNeuron() {
		return new ConvNeuronImpl(this);
	}


	@Override
	public int getWidth() {
		return width;
	}


	@Override
	public int getHeight() {
		return height;
	}


	@Override
	public Filter getFilter() {
		return filter;
	}


	@Override
	public ConvNeuron get(int x, int y) {
		return neurons[y*width + x];
	}


	@Override
	public NeuronValue set(int x, int y, NeuronValue value) {
		ConvNeuron neuron = neurons[y*width + x];
		if (neuron == null)
			return null;
		else {
			NeuronValue prevValue = neuron.getValue();
			neuron.setValue(value);
			return prevValue;
		}
	}


	@Override
	public int size() {
		return neurons.length;
	}


	@Override
	public ConvNeuron[] getNeurons() {
		return neurons;
	}


	@Override
	public NeuronValue[] getData() {
		if (neurons == null || neurons.length <= 0) return null;
		
		NeuronValue[] data = new NeuronValue[neurons.length];
		for (int i = 0; i < neurons.length; i++) {
			NeuronValue value = neurons[i].getValue();
			data[i] = value;
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

		((ConvLayerAbstract)nextLayer).prevLayer = this;
		
		if (oldNextNextLayer == null) return true;
		((ConvLayerAbstract)oldNextNextLayer).prevLayer = nextLayer;
		((ConvLayerAbstract)nextLayer).nextLayer = oldNextNextLayer;

		return true;
	}


	@Override
	public ConvNeuron[] forward() {
		ConvLayer nextLayer = getNextLayer();
		if (nextLayer == null) return null;
		ConvNeuron[] nextNeurons = nextLayer.getNeurons();
		if (nextNeurons == null || nextNeurons.length == 0) return null;
		
		Filter filter = getFilter();
		if (filter == null) return null;
		
		int filterWidth = filter.slideWidth();
		int filterHeight = filter.slideHeight();
		int thisWidth = this.getWidth();
		int thisHeight = this.getHeight();
		int nextWidth = nextLayer.getWidth();
		int nextHeight = nextLayer.getHeight();
		
		int blockWidth = filter.isBlockSlide() ? thisWidth / filterWidth : 1;
		int blockHeight = filter.isBlockSlide() ? thisHeight / filterHeight : 1;
		if (filter instanceof DeconvFilter) {
			blockWidth = filterWidth;
			blockHeight = filterHeight;
		}
		
		for (int nextY = 0; nextY < nextHeight; nextY++) {
			int Y = 0;
			if (filter instanceof DeconvFilter) {
				Y = (int) ((double)nextY/blockHeight + 0.5);
				Y = Y < thisHeight ? Y : thisHeight-1;
			}
			else {
				int yBlock = nextY < blockHeight ? nextY : blockHeight-1;
				Y = yBlock*filterHeight;
			}
			
			for (int nextX = 0; nextX < nextWidth; nextX++) {
				int X = 0;
				if (filter instanceof DeconvFilter) {
					X = (int) ((double)nextX/blockWidth + 0.5);
					X = X < thisWidth ? X : thisWidth-1;
				}
				else {
					int xBlock = nextX < blockWidth ? nextX : blockWidth-1;
					X = xBlock*filterWidth;
				}
				
				NeuronValue value = filter.apply(X, Y, this);
				int index = nextY*nextWidth + nextX;
				if (value != null)
					nextNeurons[index].setValue(value);
				else
					nextNeurons[index].setValue(newNeuronValue().zero());
			}
		}
		
		return nextNeurons;
	}

	
}
