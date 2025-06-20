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
 * This class represents an object that can be converted from and to text.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface TextParsable extends Serializable, Cloneable {

	
	/**
	 * Converting this object to text.
	 * @return text form.
	 */
	String toText();
	
	
}
