/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value.vector;

import net.ea.ann.core.value.WeightValue;

/**
 * This interface represents a weight value vector.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface WeightValueVector extends WeightValue {

	
	/**
	 * Getting length of this vector.
	 * @return length of this vector.
	 */
	int length();


}
