package net.ea.ann.mane;

import java.io.Serializable;
import java.util.List;

import net.ea.ann.core.Util;
import net.ea.ann.core.value.Matrix;

/**
 * This class represents record.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Record implements Cloneable, Serializable {
	
	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * This class represents a pair of input and output.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static class Inout implements Cloneable, Serializable {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;
	
		/**
		 * Input.
		 */
		public Matrix input = null;
		
		/**
		 * Output.
		 */
		public Matrix output = null;
		
		/**
		 * Constructor with input and output.
		 * @param input input.
		 * @param output output.
		 */
		public Inout (Matrix input, Matrix output) {
			this.input = input;
			this.output = output;
		}
		
	}
	
	
	/**
	 * List of inputs and outputs.
	 */
	public List<Inout> inouts = Util.newList(0);
	
	
	/**
	 * Extra inputs.
	 */
	public Object[] extraInputs = null;
	
	
	/**
	 * Default constructor.
	 */
	public Record() {
		
	}
	
	
	/**
	 * Constructor of array of inputs and outputs.
	 * @param inouts array of inputs and outputs.
	 */
	public Record(Inout...inouts) {
		if (inouts == null || inouts.length == 0) return;
		for (int i = 0; i < inouts.length; i++) {
			if (inouts[i] != null) this.inouts.add(inouts[i]);
		}
	}
	
	
	/**
	 * Constructor with two pairs of inputs and outputs.
	 * @param input1 input 1.
	 * @param output1 output 1.
	 * @param input2 input 2.
	 * @param output2 output 2.
	 */
	public Record(Matrix input1, Matrix output1, Matrix input2, Matrix output2) {
		this(new Inout(input1, output1), new Inout(input2, output2));
	}

	
	/**
	 * Constructor with input and output.
	 * @param input input.
	 * @param output output.
	 */
	public Record(Matrix input, Matrix output) {
		this(new Inout(input, output));
	}
	
	
	/**
	 * Constructor with input.
	 * @param input input.
	 */
	public Record(Matrix input) {
		this(new Inout(input, null));
	}

	
	/**
	 * Getting size of record.
	 * @return size of record.
	 */
	public int size() {return inouts.size();}
	
	
	/**
	 * Getting input and output at specified index.
	 * @param index specified index.
	 * @return input and output at specified index.
	 */
	public Inout get(int index) {return inouts.get(index);}
	
	
	/**
	 * Getting input at specified index.
	 * @param index specified index.
	 * @return input at specified index.
	 */
	public Matrix input(int index) {return get(index).input;}
	
	
	/**
	 * Getting output at specified index.
	 * @param index specified index.
	 * @return output at specified index.
	 */
	public Matrix output(int index) {return get(index).output;}

	
	/**
	 * Getting first input.
	 * @return first input.
	 */
	public Matrix input() {return size() > 0 ? input(0) : null;}
	
	
	/**
	 * Getting first output.
	 * @return first output.
	 */
	public Matrix output() {return size() > 0 ? output(0) : null;}
	
	
	/**
	 * Getting second input.
	 * @return second input.
	 */
	public Matrix input2() {return size() > 1 ? input(1) : null;}
	
	
	/**
	 * Getting second output.
	 * @return second output.
	 */
	public Matrix output2() {return size() > 1 ? output(1) : null;}

	
	/**
	 * Getting extra input at specified index.
	 * @param index specified index.
	 * @return extra input at specified index.
	 */
	public Object extraInput(int index) {
		return extraInputs != null && extraInputs.length > 0 && index >= 0 && index < extraInputs.length ? extraInputs[index] : null; 
	}

	
	/**
	 * Getting first extra input.
	 * @return first extra input.
	 */
	public Object extraInput() {
		return extraInputs != null && extraInputs.length > 0 ? extraInputs[0] : null; 
	}
	
	
}


