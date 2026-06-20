/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter;

import net.hudup.core.alg.Alg;

/**
 * This interface represents delegate algorithm.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Delegator extends Alg {

	
	/**
	 * Creating delegated model.
	 * @return delegated model.
	 */
	Object createDelegate();
	
	
}
