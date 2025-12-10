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
 * Logistic function.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Logistic extends FunctionInvertible, NormSupporter {


	/**
	 * Default slope.
	 */
	final static double DEFAULT_SLOPE = 1; //0.5;
	
	
	/**
	 * Flag to concern middle value.
	 */
	final static boolean CONCERN_MID = false; //true;
	
}
