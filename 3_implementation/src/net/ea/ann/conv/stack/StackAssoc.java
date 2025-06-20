/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.stack;

import java.io.Serializable;

/**
 * This class is an associator of convolutional stack.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class StackAssoc implements Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal stack.
	 */
	protected StackAbstract stack = null;
	
	
	/**
	 * Constructor with specified stack.
	 * @param stack specified stack.
	 */
	public StackAssoc(StackAbstract stack) {
		this.stack = stack;
	}

	
}
