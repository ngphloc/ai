/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.beans;

import java.util.List;

import net.ea.ann.core.Id;
import net.ea.ann.core.function.Function;
import net.ea.ann.raster.Raster;

/**
 * This class implements Proxy-NCA (Proxy-Neighborhood Component Analysis) algorithm for deep metric learning, supporting both classification and clustering.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ProxyNCAClassiter extends ProxyNCA {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public ProxyNCAClassiter(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}

	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public ProxyNCAClassiter(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public ProxyNCAClassiter(int neuronChannel, Function activateRef) {this(neuronChannel, activateRef, null, null);}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public ProxyNCAClassiter(int neuronChannel) {this(neuronChannel, null, null, null);}

	
	/**
	 * Classifying or clustering rasters. This method currently supports only classification.
	 * @param sample raster sample.
	 * @return classified or clustered rasters.
	 */
	public List<Raster> classiterRaster(Iterable<Raster> sample) {
		if (!isLabeled()) throw new RuntimeException("Not implemented yet");
		
		throw new RuntimeException("Not implemented yet");
	}
	
	
}
