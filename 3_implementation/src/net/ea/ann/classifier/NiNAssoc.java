/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.classifier;

import java.io.Serializable;

/**
 * This class provides utility methods for NiN.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class NiNAssoc implements Cloneable, Serializable {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * NiN classifier.
	 */
	protected NiN nin = null;
	
	
	/**
	 * Constructor with NiN.
	 * @param nin NiN model.
	 */
	public NiNAssoc(NiN nin) {
		this.nin = nin;
	}


}
