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
 * This interface represents searching component (matching component).
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Searcher extends Serializable, Cloneable {

	
	/**
	 * This interface represents scored feature.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	interface ScoredFeature extends Feature {
		
		/**
		 * Getting score.
		 * @return score.
		 */
		double getScore();
		
	}
	
	
	/**
	 * Searching for given query feature.
	 * @param query query feature.
	 * @param maxFound the maximum number of found features.
	 * @param params additional parameters.
	 * @return list of found features.
	 */
	List<ScoredFeature> search(Feature query, int maxFound, Object...params);
	
	
	/**
	 * Searching for given query feature.
	 * @param query query feature.
	 * @param params additional parameters.
	 * @return list of found features.
	 */
	default List<ScoredFeature> search(Feature query, Object...params) {
		return search(query, 0, params);
	}
	
	
}
