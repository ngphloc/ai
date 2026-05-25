/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir;

import java.io.Serializable;
import java.util.List;

/**
 * This interface represents Approximate Nearest Neighbor (ANN) algorithm.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface AppNN extends Cloneable, Serializable {

	
	/**
	 * Obtaining candidates.
	 * @return candidates.
	 */
	List<Feature> obtainCandidates();


}
