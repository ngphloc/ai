/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen;

import java.io.Serializable;

import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Size;

/**
 * This interface provides methods related to feature and X data, for instance, X is image and feature is aspects of such image.
 * However, in context of generative model, X is data (original as well as generated) and feature is feature of some other digital content like image.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface FeatureToX extends Cloneable, Serializable {

	
	/**
	 * Converting feature to X (original as well as generated).
	 * @param feature feature array.
	 * @return X array (original as well as generated).
	 */
	NeuronValue[] convertFeatureToX(NeuronValue[] feature);
	
	
	/**
	 * Converting X (original as well as generated) to feature.
	 * @param dataX X array (original as well as generated).
	 * @return feature array.
	 */
	NeuronValue[] convertXToFeature(NeuronValue[] dataX);

	
	/**
	 * Getting feature size.
	 * @return feature size.
	 */
	Size getFeatureSize();

	
}
