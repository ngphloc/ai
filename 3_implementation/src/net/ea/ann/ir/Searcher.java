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
 *
 * @param <T> feature type.
 */
public interface Searcher<T extends Feature> extends Serializable, Cloneable {

	
	/**
	 * Searching for given query feature.
	 * @param query query feature.
	 * @param maxFound the maximum number of found features.
	 * @return list of found features.
	 */
	List<ScoredFeature<T>> search(T query, int maxFound);
	
	
	/**
	 * Searching for given query feature.
	 * @param query query feature.
	 * @param params additional parameters.
	 * @return list of found features.
	 */
	default List<ScoredFeature<T>> search(T query) {
		return search(query, 0);
	}
	
	
}
