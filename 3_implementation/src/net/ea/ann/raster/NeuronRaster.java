/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.io.Serializable;

import net.ea.ann.core.Neuron;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class represent a neuron raster as a pair of neuron array and size.
 * 
 * @author Loc Nguyen
 * @version 1.0
 */
public class NeuronRaster implements Serializable, Cloneable {
	
	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;
	
	
	/**
	 * Neuron array as data.
	 */
	protected Neuron[] neurons = null;
	
	
	/**
	 * Size of neuron raster.
	 */
	protected Size size = null;
	
	
	/**
	 * Constructor with data and size.
	 * @param neuronChannel neuron channel.
	 * @param neurons neuron array as data.
	 * @param size raster size
	 */
	public NeuronRaster(int neuronChannel, Neuron[] neurons, Size size) {
		this.neuronChannel = neuronChannel;
		this.neurons = neurons;
		this.size = size;
	}

	
	/**
	 * Getting size.
	 * @return raster size.
	 */
	public Size getSize() {
		return size;
	}
	
	
	/**
	 * Getting neurons.
	 * @return neurons.
	 */
	public Neuron[] getNeurons() {
		return neurons;
	}

	
	/**
	 * Getting data as array of neuron value.
	 * @return data as array of neuron value.
	 */
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
	 * Converting this plain raster to realistic raster.
	 * @param isNorm flag to indicate whether value is normalized in range [0, 1].
	 * @param defaultAlpha default alpha value.
	 * @return realistic raster.
	 */
	public Raster toRaster(boolean isNorm, int defaultAlpha) {
		return RasterAssoc.createRaster(getData(), neuronChannel, size, isNorm, defaultAlpha);
	}
	
	
	/**
	 * Converting this plain raster to realistic raster.
	 * @param isNorm flag to indicate whether value is normalized in range [0, 1].
	 * @return realistic raster.
	 */
	public Raster toRaster(boolean isNorm) {
		return toRaster(isNorm, Image.ALPHA_DEFAULT);
	}

	
}
