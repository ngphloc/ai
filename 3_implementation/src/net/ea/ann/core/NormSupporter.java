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
 * This interface represents a utility class to provide methods related normalization.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface NormSupporter extends Cloneable, Serializable {

	
	/**
	 * Checking whether something normalized in rang [0, 1].
	 * @return whether something normalized in rang [0, 1].
	 */
	boolean isNorm();
	
	
}
