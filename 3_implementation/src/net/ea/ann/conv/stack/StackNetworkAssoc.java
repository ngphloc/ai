/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.stack;

import java.awt.Dimension;
import java.io.Serializable;

import net.ea.ann.conv.Content;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.conv.filter.FilterAssoc;
import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkStandardImpl;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Size;

/**
 * This class is associator of convolutional stack network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class StackNetworkAssoc implements Serializable, Cloneable {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal network.
	 */
	protected StackNetworkAbstract network = null;
	
	
	/**
	 * Constructor with specified stack network.
	 * @param network specified stack network.
	 */
	public StackNetworkAssoc(StackNetworkAbstract network) {
		this.network = network;
	}

	
	/**
	 * Getting feature of specified raster.
	 * @param raster specified raster.
	 * @return feature of specified raster.
	 */
	public Content getFeature(Raster raster) {
		if (network == null || raster == null) return null;
		StackNetworkAbstract clonedConv = network;//(StackNetworkAbstract)Util.cloneBySerialize(network);
		try {
			clonedConv.evaluateRaster(raster);
			return clonedConv.getFeatureFitChannel();
		} catch (Throwable e) {Util.trace(e);}
		
		return null;
	}


	/**
	 * Getting full network.
	 * @return full network.
	 */
	public NetworkStandardImpl getFullNetwork() {
		return network.fullNetwork;
	}
	
	
	/**
	 * Converting feature into unified content data, which call {@link StackNetworkAbstract#convertFeatureToUnifiedContentData(NeuronValue[])}.
	 * @param feature specified feature represented by array of neuron values.
	 * @return array of neuron values as data of unified content.
	 */
	public NeuronValue[] convertFeatureToUnifiedContentData(NeuronValue[] feature) {
		return network.convertFeatureToUnifiedContentData(feature);
	}
	
	
	/**
	 * Checking whether network has learning.
	 * @return whether network has learning.
	 */
	public boolean hasLearning() {
		return !network.onlyForward || network.fullNetwork != null || network.reversedFullNetwork != null;
	}
	
	
	/**
	 * Creating stack network for extracting 2D features with neuron channel, activation function, content activation, and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param widthHeight specified width and height.
	 * @param activateRef activation function, which is often activation function related to weights like sigmod function.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 * @param idRef ID reference.
	 * @return stack network.
	 */
	public static StackNetworkImpl createFeatureExtractor2D(int neuronChannel, Dimension widthHeight, Function activateRef, Function contentActivateRef, Id idRef) {
		StackNetworkImpl extractor = StackNetworkImpl.create(neuronChannel, activateRef, contentActivateRef, idRef);
		if (extractor == null) return null;
		
		Size size = Size.unit();
		size.width = widthHeight.width;
		size.height = widthHeight.height;
		Filter[][] filterArrays = FilterAssoc.createFeatureExtractor2D(extractor.newStack(Size.unit()));
		return extractor.initialize(size, filterArrays) ? extractor : null;
	}


	/**
	 * Creating stack network for extracting 2D features with neuron channel, content activation function, and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param widthHeight specified width and height.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 * @param idRef ID reference.
	 * @return stack network.
	 */
	public static StackNetworkImpl createFeatureExtractor2D(int neuronChannel, Dimension widthHeight, Function contentActivateRef, Id idRef) {
		return createFeatureExtractor2D(neuronChannel, widthHeight, null, contentActivateRef, idRef);
	}


	/**
	 * Creating stack network for extracting 2D features with neuron channel, and content activation function.
	 * @param neuronChannel neuron channel.
	 * @param widthHeight specified width and height.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 * @return stack network.
	 */
	public static StackNetworkImpl createFeatureExtractor2D(int neuronChannel, Dimension widthHeight, Function contentActivateRef) {
		return createFeatureExtractor2D(neuronChannel, widthHeight, null, contentActivateRef, null);
	}
	
	
	/**
	 * Creating stack network for extracting 2D features with neuron channel, and content activation function.
	 * @param neuronChannel neuron channel.
	 * @param widthHeight specified width and height.
	 * @param isNorm normalization flag.
	 * @return stack network.
	 */
	public static StackNetworkImpl createFeatureExtractor2D(int neuronChannel, Dimension widthHeight, boolean isNorm) {
		Function contentActivateRef = Raster.toConvActivationRef(neuronChannel, isNorm);
		return createFeatureExtractor2D(neuronChannel, widthHeight, null, contentActivateRef, null);
	}

	
}
