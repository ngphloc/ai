/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir;

import net.ea.ann.ir.Feature.MatrixFeature;

/**
 * This class is an abstract class of Approximate Nearest Neighbor (ANN) algorithm.
 * @author Loc Nguyen
 * @version 1.0
 *
 * @param <T> feature type.
 */
public abstract class AppNNAbstract<T extends Feature> implements AppNN<T> {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public AppNNAbstract() {

	}

	
	/**
	 * This class is an abstract class of Approximate Nearest Neighbor (ANN) algorithm with respect to matrix feature.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static abstract class MatrixFeatureAppNNAbstract extends AppNNAbstract<MatrixFeature> {

		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Default constructor.
		 */
		public MatrixFeatureAppNNAbstract() {
			super();
		}
		
	}
	

}
