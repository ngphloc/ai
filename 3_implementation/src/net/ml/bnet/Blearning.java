/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ml.bnet;

import net.hudup.core.data.Profile;

/**
 * This interface represents a learning algorithm for Bayesian network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Blearning {

	
	/**
	 * The main method learns or create Bayesian network from training data.
	 * @param input specified training data.
	 * @param params additional parameters.
	 * @return Bayesian network from specified training data.
	 */
	Bnet learn(Iterable<Profile> input, Object...params);
	
	
}
