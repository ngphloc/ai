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
 * This class is an associator of stack element layer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ElementLayerAssoc implements Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal element layer.
	 */
	protected ElementLayerAbstract layer = null;
	
	
	/**
	 * Constructor with specified layer.
	 * @param layer specified layer.
	 */
	public ElementLayerAssoc(ElementLayerAbstract layer) {
		this.layer = layer;
	}

	
	/**
	 * Setting next layers with array of weights.
	 * @param weights array of weights.
	 */
	public void setNextLayers(Weight...weights) {
		if (weights == null || weights.length == 0) return;
		if (layer.stack == null || layer.stack.getNextStack() == null) return;

		Stack nextStack = layer.stack.getNextStack();
		int n = Math.min(weights.length, nextStack.size());
		for (int i = 0; i < n; i++) layer.setNextLayer(nextStack.get(i), weights[i], null);
	}
	
	
	/**
	 * Setting next layers with arrays of weights and filters.
	 * @param weights array of weights.
	 * @param filters array of filters.
	 */
	public void setNextLayers(Weight[] weights, Filter[] filters) {
		if (filters == null || filters.length == 0) {
			setNextLayers(weights);
			return;
		}
		if (layer.stack == null || layer.stack.getNextStack() == null) return;
		
		Stack nextStack = layer.stack.getNextStack();
		int n = Math.min(filters.length, nextStack.size());
		Weight weight0 = layer.stack.newWeight();
		for (int i = 0; i < n; i++) {
			Weight weight = (weights != null && i < weights.length) ?  weights[i] : weight0;
			layer.setNextLayer(nextStack.get(i), weight, filters[i]);
		}
		
	}
	
	
	/**
	 * Setting next layers with array of filters.
	 * @param filters array of filters.
	 */
	public void setNextLayers(Filter...filters) {
		setNextLayers(null, filters);
	}
	
	
}
