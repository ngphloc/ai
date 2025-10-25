/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.io.Serializable;
import java.util.Random;

import net.ea.ann.conv.filter.ProductFilter2D;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.NeuronValue;

/**
 * This utility class provides utility methods for matrix network layer.
 *  
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MatrixLayerAssoc implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Internal matrix network layer.
	 */
	protected MatrixLayerImpl layer = null;
	

	/**
	 * Constructor with layer.
	 * @param layer layer.
	 */
	public MatrixLayerAssoc(MatrixLayerImpl layer) {
		this.layer = layer;
	}

	
	/**
	 * Initializing parameters by specified value.
	 * @param v value.
	 */
	public void initParams(double v) {
		if (layer.weight1 != null) Matrix.fill(layer.weight1, v);
		if (layer.weight2 != null) Matrix.fill(layer.weight2, v);
		if (layer.bias != null) Matrix.fill(layer.bias, v);
		
		if (layer.filter != null && layer.filter instanceof ProductFilter2D) {
			ProductFilter2D filter = (ProductFilter2D)layer.filter;
			NeuronValue.fill(filter.getKernel(), v);
			filter.setWeight(filter.getWeight().unit());
		}
		if (layer.filterBias != null) layer.filterBias = layer.filterBias.valueOf(v);
	}


	/**
	 * Initializing parameters.
	 * @param rnd randomizer.
	 */
	public void initParams(Random rnd) {
		if (layer.weight1 != null) Matrix.fill(layer.weight1, rnd);
		if (layer.weight2 != null) Matrix.fill(layer.weight2, rnd);
		if (layer.bias != null) Matrix.fill(layer.bias, rnd);
		
		if (layer.filter != null && layer.filter instanceof ProductFilter2D) {
			ProductFilter2D filter = (ProductFilter2D)layer.filter;
			NeuronValue.fill(filter.getKernel(), rnd);
			filter.setWeight(filter.getWeight().unit());
		}
		if (layer.filterBias != null) layer.filterBias = layer.filterBias.valueOf(NeuronValue.r(rnd));
	}

	
}
