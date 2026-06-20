/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir;

import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represents scored feature.
 * @author Loc Nguyen
 * @version 1.0
 *
 * @param <T> feature type.
 */
public interface ScoredFeature<T extends Feature> extends Feature {

	
	/**
	 * Getting score.
	 * @return score.
	 */
	double score();

	
	/**
	 * Getting feature.
	 * @return feature.
	 */
	T feature();

	
	/**
	 * This class represents feature associated with score.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 * @param <T> feature type.
	 */
	class ScoredFeatureImpl<T extends Feature> implements ScoredFeature<T> {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Internal feature.
		 */
		protected T feature = null;
		
		/**
		 * Score.
		 */
		protected double score = 0;

		/**
		 * Constructor with feature and score.
		 * @param feature feature.
		 * @param score score.
		 */
		public ScoredFeatureImpl(T feature, double score) {
			this.feature = feature;
			this.score = score;
		}

		@Override
		public NeuronValue sim(Feature other) {
			if (other instanceof ScoredFeature<?>) {
				ScoredFeature<?> otherSF = (ScoredFeature<?>)other;
				return this.feature.sim(otherSF.feature());
			}
			else
				return this.feature.sim(other);
		}

		@Override
		public NeuronValue distance(Feature other) {
			if (other instanceof ScoredFeature<?>) {
				ScoredFeature<?> otherSF = (ScoredFeature<?>)other;
				return this.feature.distance(otherSF.feature());
			}
			else
				return this.feature.distance(other);
		}

		@Override
		public Record getRecord() {return feature != null ? feature.getRecord() : null;}

		@Override
		public double score() {return score;}

		@Override
		public T feature() {return this.feature;}
		
	}

	
}
