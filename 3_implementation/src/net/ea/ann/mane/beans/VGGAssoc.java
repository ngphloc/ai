/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.beans;

import java.io.Serializable;

/**
 * This associative class provides utility methods for VGG model.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class VGGAssoc implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * VGG model.
	 */
	protected VGG vgg = null;
	
	
	/**
	 * Constructor with VGG model.
	 * @param vgg VGG model.
	 */
	public VGGAssoc(VGG vgg) {
		this.vgg = vgg;
	}
	

}
