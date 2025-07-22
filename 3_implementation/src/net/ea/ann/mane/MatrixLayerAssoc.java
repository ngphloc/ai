/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.io.Serializable;

/**
 * This utility class provides utility methods for matrix network layer.
 *  
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MatrixLayerAssoc implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Internal matrix network layer.
	 */
	protected MatrixLayerImpl layer = null;
	

	/**
	 * Constructor with layer.
	 * @param layer layer.
	 */
	public MatrixLayerAssoc(MatrixLayerImpl layer) {
		this.layer = layer;
	}

	
}
