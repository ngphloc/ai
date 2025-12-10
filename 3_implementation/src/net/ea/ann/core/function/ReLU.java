/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.function;

import net.ea.ann.core.NormSupporter;

/**
 * This interface represents rectified linear unit (ReLU) function.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface ReLU extends FunctionInvertible, NormSupporter {

	
	/**
	 * Flag to concern maximum bound.
	 */
	final static boolean CONCERN_MAX = false; //true;
	
	
}
