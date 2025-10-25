/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.transformer;

import java.io.Serializable;

/**
 * This utility class provides utility methods for default transformer.
 *  
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class TransformerAssoc implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Internal transformer.
	 */
	protected TransformerImpl transformer = null;

	
	/**
	 * Constructor with transformer.
	 * @param transformer transformer.
	 */
	public TransformerAssoc(TransformerImpl transformer) {
		this.transformer = transformer;
	}

	
}
