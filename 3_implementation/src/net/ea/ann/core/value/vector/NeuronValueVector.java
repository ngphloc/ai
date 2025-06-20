/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value.vector;

import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueComposite;

/**
 * This interface represents a neuron value vector.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface NeuronValueVector extends NeuronValueComposite {

	
	/**
	 * Getting element at specified index.
	 * @param index specified index.
	 * @return element at specified index.
	 */
	NeuronValue get(int index);

	
	/**
	 * Setting element at specified index.
	 * @param index specified index.
	 * @param element element.
	 * @return replaced element.
	 */
	NeuronValue set(int index, NeuronValue element);
	
	
	/**
	 * This enum represents operator.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	static enum Operator {
		
		/**
		 * Addition operator.
		 */
		add,
		
		/**
		 * Multiplication operator.
		 */
		multiply,
		
	}

	
	/**
	 * Taking operator between neuron value and neuron value.
	 * @param v1 neuron value 1.
	 * @param v2 neuron value 2.
	 * @return neuron value.
	 */
	private static NeuronValue operator0(NeuronValue v1, NeuronValue v2, Operator op) {
		NeuronValue result = null;
		switch (op) {
		case add:
			result = v1.add(v2);
			break;
		case multiply:
			result = v1.multiply(v2);
			break;
		}
		return result;
	}

	
	/**
	 * Taking operator between neuron value and neuron value with regard to vector.
	 * @param v1 neuron value 1.
	 * @param v2 neuron value 2.
	 * @return neuron value.
	 */
	private static NeuronValue operator(NeuronValue v1, NeuronValue v2, Operator op) {
		if ((v1 instanceof NeuronValueVector) && (v2 instanceof NeuronValueVector))
			return operator0(v1, v2, op);
		if (!(v1 instanceof NeuronValueVector) && !(v2 instanceof NeuronValueVector))
			return operator0(v1, v2, op);
		
		NeuronValueVector vector1 = null, vector2 = null;
		if (v1 instanceof NeuronValueVector) {
			vector1 = (NeuronValueVector)v1;
			vector2 = new NeuronValueVectorImpl(vector1.length(), v2);
		}
		else {
			vector1 = (NeuronValueVector)v2;
			vector2 = new NeuronValueVectorImpl(vector1.length(), v1);
		}
		
		NeuronValue result = null;
		switch (op) {
		case add:
			result = vector1.add(vector2);
			break;
		case multiply:
			result = vector1.multiply(vector2);
			break;
		default:
			break;
		}
		return result;
	}

	
	/**
	 * Adding neuron value and neuron value.
	 * @param v1 neuron value 1.
	 * @param v2 neuron value 2.
	 * @return neuron value.
	 */
	public static NeuronValue add(NeuronValue v1, NeuronValue v2) {
		return operator(v1, v2, Operator.add);
	}
	
	
	/**
	 * Adding neuron value and neuron zero.
	 * @param value neuron value.
	 * @param zero neuron zero.
	 * @return neuron value.
	 */
	public static NeuronValue addZero(NeuronValue value, NeuronValue zero) {
		if ((value instanceof NeuronValueVector) && (zero instanceof NeuronValueVector))
			return value;
		if (!(value instanceof NeuronValueVector) && !(zero instanceof NeuronValueVector))
			return value;
		else
			return add(value, zero);
	}


}
