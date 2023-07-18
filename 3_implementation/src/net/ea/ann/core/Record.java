/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import java.io.Serializable;
import java.util.Map;

/**
 * This class is sample record for learning neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Record implements Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Backbone input.
	 */
	public NeuronValue[] input = null;
	
	
	/**
	 * Backbone output. It can be null.
	 */
	public NeuronValue[] output = null;
	
	
	/**
	 * Rib input. It can be null.
	 */
	public Map<Integer, NeuronValue[]> ribInput = Util.newMap(0);
	
	
	/**
	 * Rib output. It can be null.
	 */
	public Map<Integer, NeuronValue[]> ribOutput = Util.newMap(0);

	
	/**
	 * Undefined input.
	 */
	public Object undefinedInput = null;

	
	/**
	 * Undefined output.
	 */
	public Object undefinedOutput = null;

	
	/**
	 * Default constructor.
	 */
	public Record() {

	}

	
}
