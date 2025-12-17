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
 * This class represents filter specification.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class WeightSpec implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * This enum specifies weight type.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static enum Type {
		
		/**
		 * Normal weight.
		 */
		normal,
		
		/**
		 * Transformed-based weight.
		 */
		transformer,
		
	}

	
	/**
	 * Weight type.
	 */
	public Type type = Type.normal;

	
	/**
	 * Default constructor.
	 */
	public WeightSpec(Type type) {
		this.type = type;
	}

	
}
