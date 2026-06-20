/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir;

import java.util.Collection;
import java.util.List;

import net.ea.ann.core.Util;
import net.ea.ann.ir.Feature.MatrixFeature;
import net.ea.ann.ir.ScoredFeature.ScoredFeatureImpl;

/**
 * This class implements partially searching component (matching component).
 * @author Loc Nguyen
 * @version 1.0
 *
 * @param <T> feature type.
 */
public abstract class SearcherAbstract<T extends Feature> implements Searcher<T> {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public SearcherAbstract() {

	}
	
	
	/**
	 * Creating scored feature.
	 * @param feature feature.
	 * @param score score.
	 * @return scored feature.
	 */
	protected ScoredFeature<T> newScoredFeature(T feature, double score) {
		return new ScoredFeatureImpl<T>(feature, score);
	}
	
	
	/**
	 * Getting library.
	 * @return library.
	 */
	protected abstract Library<T> getLibrary();
	
	
	@Override
	public List<ScoredFeature<T>> search(T query, int maxFound) {
		return search(getLibrary(), query, maxFound);
	}


	/**
	 * Searching for given query feature.
	 * @param library library.
	 * @param query query feature.
	 * @param maxFound the maximum number of found features.
	 * @return list of found features.
	 */
	public List<ScoredFeature<T>> search(Library<T> library, T query, int maxFound) {
		return search(library.obtainCandidates(query), query, maxFound);
	}

	
	/**
	 * Searching for given query feature.
	 * @param searchSpace searching space.
	 * @param query query feature.
	 * @param maxFound the maximum number of found features.
	 * @return list of found features.
	 */
	protected List<ScoredFeature<T>> search(Collection<T> searchSpace, T query, int maxFound) {
		List<ScoredFeature<T>> result = Util.newList(0);
		for (T feature : searchSpace) {
			if (maxFound > 0 && result.size() >= maxFound) break;

			double score = feature.sim(query).mean();
			boolean found = false;
			for (int i = result.size() - 1; i >= 0; i++) {
				if (result.get(i).score() >= score) {
					result.add(i + 1, newScoredFeature(feature, score));
					found = true;
					break;
				}
			}
			if (!found) result.add(0, newScoredFeature(feature, score));
		}
		return result;
	}
	
	
	/**
	 * This class implements partially searching component (matching component) with respect matrix feature.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static abstract class MatrixFeatureSearcherAbstract extends SearcherAbstract<MatrixFeature> {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Default constructor.
		 */
		public MatrixFeatureSearcherAbstract() {
			super();
		}
		
	}

	
}
