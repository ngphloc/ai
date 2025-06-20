/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import net.ea.ann.core.value.NeuronValueComposite;
import net.ea.ann.core.value.WeightValue;

/**
 * This interface represents content value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface ContentValue extends Content, NeuronValueComposite {

	
	/**
	 * This interface represents content weight.
	 * 
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	interface ContentWeight extends Content, WeightValue {

		
	}


}
