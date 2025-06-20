/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import java.io.Serializable;

/**
 * This interface represents layer in neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Layer extends Serializable, Cloneable {

	
	/**
	 * Getting identifier reference.
	 * @return identifier reference.
	 */
	Id getIdRef();

	
	/**
	 * Getting identifier.
	 * @return identifier.
	 */
	int id();

	
}