/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.io.Serializable;

import net.ea.ann.core.value.NeuronValue;

/**
 * This class represent a neuron value raster as a pair of neuron value array and size.
 * 
 * @author Loc Nguyen
 * @version 1.0
 */
public class NeuronValueRaster implements Serializable, Cloneable {
	
	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;
	
	
	/**
	 * Neuron value array as data.
	 */
	protected NeuronValue[] values = null;
	
	
	/**
	 * Size of neuron raster.
	 */
	protected Size size = null;
	
	
	/**
	 * Number of actual values.
	 */
	protected int countValues = 0;
	
	
	/**
	 * Constructor with data and size.
	 * @param neuronChannel neuron channel.
	 * @param values neuron value array as data.
	 * @param size raster size.
	 * @param countValues number of actual values.
	 */
	public NeuronValueRaster(int neuronChannel, NeuronValue[] values, Size size, int countValues) {
		this.neuronChannel = neuronChannel;
		this.values = values;
		this.size = size;
		this.countValues = countValues;
	}

	
	/**
	 * Getting size.
	 * @return raster size.
	 */
	public Size getSize() {
		return size;
	}
	
	
	/**
	 * Getting neuron values.
	 * @return neuron values.
	 */
	public NeuronValue[] getValues() {
		return values;
	}

	
	/**
	 * Getting actual number of values.
	 * @return actual number of values.
	 */
	public int getCountValues() {
		return countValues;
	}
	
	
	/**
	 * Converting this plain raster to realistic raster.
	 * @param isNorm flag to indicate whether value is normalized in range [0, 1].
	 * @param defaultAlpha default alpha value.
	 * @return realistic raster.
	 */
	public Raster toRaster(boolean isNorm, int defaultAlpha) {
		return RasterAssoc.createRaster(getValues(), neuronChannel, size, isNorm, defaultAlpha);
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
