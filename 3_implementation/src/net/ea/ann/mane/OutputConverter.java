/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import net.ea.ann.core.value.Matrix;

/**
 * This functional interface converts an output into another meaning output.
 *  
 * @author Loc Nguyen
 * @version 1.0
 *
 */
@FunctionalInterface
public interface OutputConverter {

	
	/**
	 * Converting specified output into another meaning output.
	 * @param output specified output.
	 * @return another meaning output.
	 */
	Matrix convert(Matrix output);
	
	
}
