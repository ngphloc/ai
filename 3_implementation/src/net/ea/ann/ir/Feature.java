/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir;

import java.io.Serializable;

import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represents feature.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Feature extends Cloneable, Serializable {

	
	/**
	 * Calculating similarity of this feature and other feature.
	 * @param other other feature.
	 * @return similarity of this feature and other feature.
	 */
	NeuronValue sim(Feature other);
	
	
	/**
	 * Calculating distance of this feature and other feature.
	 * @param other other feature.
	 * @return distance of this feature and other feature.
	 */
	NeuronValue distance(Feature other);
	
	
}
