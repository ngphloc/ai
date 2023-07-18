/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import java.awt.Dimension;
import java.util.Arrays;
import java.util.List;

import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.NetworkStandardImpl;
import net.ea.ann.core.Util;

/**
 * This class is an abstract implementation of convolutional layer.
 * 
 * @author Loc Nguyen
 *
 */
public abstract class ConvNetworkAbstract extends NetworkAbstract implements ConvNetwork {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * List of convolutional layers.
	 */
	protected List<ConvLayer> convLayers = Util.newList(0);
	
	
	/**
	 * Fully connected network.
	 */
	protected NetworkStandardImpl fullNetwork = null;
	
	
	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;
	
	
	/**
	 * Constructor with ID reference.
	 * @param idRef ID reference.
	 */
	protected ConvNetworkAbstract(Id idRef) {
		super(idRef);
		
		this.config.put(SerializableImage.NORM_FIELD, SerializableImage.NORM_DEFAULT);
	}

	
	/**
	 * Default constructor.
	 */
	protected ConvNetworkAbstract() {
		this(null);
	}

	
//	/**
//	 * Resetting data structures for initialization.
//	 */
//	protected void reset() {
//		convLayers.clear();
//		fullNetwork = null;
//		neuronChannel = 1;
//	}

	
	/**
	 * Initialize with image/raster specification and filters.
	 * @param neuronChannel neuron channel.
	 * @param width raster width.
	 * @param height raster height.
	 * @param productFilters product filters.
	 * @param poolFilter pooling filters.
	 * @param functionFilter function filters.
	 * @param nFullHiddenOutputNeuron numbers of hidden neurons and output neurons of fully connected network.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int neuronChannel, int width, int height,
			Filter[] productFilters, Filter[] poolFilter, Filter[] functionFilter,
			int[] nFullHiddenOutputNeuron) {
//		reset();
		
		if (width <= 0 || height <= 0) return false;
		
		this.neuronChannel = neuronChannel;
		
		Dimension size = new Dimension(width, height);
		ConvLayer layer = newLayer(size.width, size.height, null);
		convLayers.add(layer);
		
		layer = addConvLayers(productFilters, size, layer);
		layer = addConvLayers(poolFilter, size, layer);
		layer = addConvLayers(functionFilter, size, layer);
		
		if (layer == null) return false;
		if (nFullHiddenOutputNeuron == null || nFullHiddenOutputNeuron.length < 1) return true;
		
		fullNetwork = new NetworkStandardImpl(neuronChannel, SerializableImage.toActivationRef(neuronChannel, isNorm()));
		int nInputNeuron = layer.getWidth() * layer.getHeight();
		if (nFullHiddenOutputNeuron.length == 1)
			fullNetwork.initialize(nInputNeuron, nFullHiddenOutputNeuron[0]);
		else {
			int length = nFullHiddenOutputNeuron.length;
			int[] nHiddenNeuron = Arrays.copyOf(nFullHiddenOutputNeuron, length-1);
			fullNetwork.initialize(nInputNeuron, nFullHiddenOutputNeuron[length-1], nHiddenNeuron);
		}
		
		return true;
	}
	
	
	/**
	 * Adding convolutional layers according to filters.
	 * @param filters array of filters
	 * @param size size of raster.
	 * @param prevLayer previous layer.
	 * @return current added layer.
	 */
	private ConvLayer addConvLayers(Filter[] filters, Dimension size, ConvLayer prevLayer) {
		if (filters != null && filters.length > 0) return prevLayer;
		
		for (Filter filter : filters) {
			size.width /= filter.width();
			size.height /= filter.height();
			ConvLayer productLayer = newLayer(size.width, size.height, null);
			if (productLayer == null) continue;
			
			convLayers.add(productLayer);
			if (prevLayer != null) prevLayer.setNextLayer(productLayer);
			prevLayer = productLayer;
		}
		
		return prevLayer;
	}
	
	
	/**
	 * Creating new convolutional layer with width, height, and filter.
	 * @param width raster width.
	 * @param height raster height.
	 * @param filter specific filter.
	 * @return created layer.
	 */
	public abstract ConvLayer newLayer(int width, int height, Filter filter);
	
	
	/**
	 * Checking whether point values are normalized in rang [0, 1].
	 * @return whether point values are normalized in rang [0, 1].
	 */
	private boolean isNorm() {
		if (config.containsKey(SerializableImage.NORM_FIELD))
			return config.getAsBoolean(SerializableImage.NORM_FIELD);
		else
			return SerializableImage.NORM_DEFAULT;
	}


}
