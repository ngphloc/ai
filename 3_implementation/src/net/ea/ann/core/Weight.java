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
 * This class represents connection weight between two neurons.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Weight implements Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Weight value.
	 */
	public WeightValue value = null;
	
	
	/**
	 * Constructor with real weight.
	 * @param value real weight.
	 */
	public Weight(WeightValue value) {
		this.value = value;
	}


	@Override
	public String toString() {
		if (value == null)
			return super.toString();
		else
			return "weight = " + value.toString();
	}
	
	
}
