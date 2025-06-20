/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.stack;

import java.io.Serializable;

import net.ea.ann.conv.filter.Filter;
import net.ea.ann.core.value.Weight;

/**
 * This class represents a triple of layer, weight, and filter.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class WeightedElementLayer implements Serializable, Cloneable {
	
	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Internal layer.
	 */
	public ElementLayer layer = null;
	
	
	/**
	 * Weight.
	 */
	public Weight weight = null;
	
	
	/**
	 * Filter which can be null. It is optional field.
	 */
	public Filter filter = null;
	
	
	/**
	 * Constructor with layer and weight.
	 * @param layer specified layer.
	 * @param weight specified weight.
	 */
	public WeightedElementLayer(ElementLayer layer, Weight weight) {
		this.layer = layer;
		this.weight = weight;
	}
	
	
	/**
	 * Constructor with layer, weight, and filter.
	 * @param layer specified layer.
	 * @param weight specified weight.
	 * @param filter specified filter.
	 */
	public WeightedElementLayer(ElementLayer layer, Weight weight, Filter filter) {
		this(layer, weight);
		this.filter = filter;
	}


}


