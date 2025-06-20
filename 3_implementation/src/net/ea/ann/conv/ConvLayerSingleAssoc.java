/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import java.io.Serializable;

import net.ea.ann.conv.filter.Filter;

/**
 * This class is an associator of single convolutional layer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ConvLayerSingleAssoc implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal convolutional layer.
	 */
	protected ConvLayerSingle convLayer = null;
	
	
	/**
	 * Constructor with convolutional layer.
	 * @param convLayer convolutional layer.
	 */
	public ConvLayerSingleAssoc(ConvLayerSingle convLayer) {
		this.convLayer = convLayer;
	}

	
	/**
	 * Getting filter of current layer.
	 * @return filter of current layer.
	 */
	public Filter getMyFilter() {
		ConvLayer prevLayer = convLayer.getPrevLayer();
		return (prevLayer != null && prevLayer instanceof ConvLayerSingle) ? ((ConvLayerSingle)prevLayer).getFilter() : null;
	}

	
}
