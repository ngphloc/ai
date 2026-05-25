/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir;

import java.util.List;

import net.ea.ann.core.Util;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class implements partially searching component (matching component)
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class SearcherAbstract implements Searcher {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * This class represents feature associated with score.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public class ScoredFeatureWrapper implements ScoredFeature {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Internal feature.
		 */
		protected Feature feature = null;
		
		/**
		 * Score.
		 */
		protected double score = 0;

		/**
		 * Constructor with feature and score.
		 * @param feature feature.
		 * @param score score.
		 */
		public ScoredFeatureWrapper(Feature feature, double score) {
			this.feature = feature;
			this.score = score;
		}

		@Override
		public NeuronValue sim(Feature other) {
			return feature.sim(other);
		}

		@Override
		public NeuronValue distance(Feature other) {
			return feature.distance(other);
		}

		@Override
		public double getScore() {
			return score;
		}
		
	}

	
	/**
	 * Default constructor.
	 */
	public SearcherAbstract() {

	}
	
	
	/**
	 * Obtaining searching space.
	 * @return searching space.
	 */
	protected abstract List<Feature> obtainSearchSpace();


	@Override
	public List<ScoredFeature> search(Feature query, int maxFound, Object... params) {
		List<ScoredFeature> result = Util.newList(0);
		List<Feature> space = obtainSearchSpace();
		for (Feature feature : space) {
			if (maxFound > 0 && result.size() >= maxFound) break;

			double score = feature.sim(query).mean();
			boolean found = false;
			for (int i = result.size() - 1; i >= 0; i++) {
				if (result.get(i).getScore() >= score) {
					result.add(i + 1, new ScoredFeatureWrapper(feature, score));
					found = true;
					break;
				}
			}
			if (!found) result.add(0, new ScoredFeatureWrapper(feature, score));
		}
		return result;
	}
	

}
