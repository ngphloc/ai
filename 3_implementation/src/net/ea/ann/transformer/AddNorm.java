/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.transformer;

import net.ea.ann.core.Id;

/**
 * This class implements add & norm component.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class AddNorm extends MatrixNetworkImpl {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with ID reference.
	 * @param idRef ID reference.
	 */
	public AddNorm(Id idRef) {
		super(idRef);
	}
	

	/**
	 * Default constructor.
	 */
	public AddNorm() {
		this(new Id());
	}

	
}
