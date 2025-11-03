/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
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
	public List<Object> extraInputs = Util.newList(0);
	
	
	/**
	 * Additional parameters.
	 */
	public List<Object> params = Util.newList(0);
	
	
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
	 * Adding input and output.
	 * @param inout pair of input and output.
	 * @return true if adding is successful.
	 */
	public boolean add(Inout inout) {
		return this.inouts.add(inout);
	}
	
	
	/**
	 * Adding input and output.
	 * @param input input.
	 * @param output output.
	 */
	public boolean add(Matrix input, Matrix output) {
		return add(new Inout(input, output));
	}
	
	
	/**
	 * Adding input.
	 * @param input input.
	 */
	public boolean add(Matrix input) {
		return add(new Inout(input, null));
	}

	
	/**
	 * Removing input and output at specified index.
	 * @param index specified index.
	 * @return previous input and output.
	 */
	public Inout remove(int index) {
		return this.inouts.remove(index);
	}
	
	
	/**
	 * Clearing inputs and outputs.
	 */
	public void clear() {
		this.inouts.clear();
	}
	
	
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
	 * Extracting inputs.
	 * @return inputs.
	 */
	public Matrix[] inputs() {
		if (inouts.size() == 0) return null;
		Matrix[] inputs = new Matrix[inouts.size()];
		for (int i = 0; i < inputs.length; i++) inputs[i] = inouts.get(i).input;
		return inputs;
	}
	
	
	/**
	 * Extracting outputs.
	 * @return outputs.
	 */
	public Matrix[] outputs() {
		if (inouts.size() == 0) return null;
		Matrix[] outputs = new Matrix[inouts.size()];
		for (int i = 0; i < outputs.length; i++) outputs[i] = inouts.get(i).output;
		return outputs;
	}

	
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
	 * Getting size of extra inputs.
	 * @return size of extra inputs.
	 */
	public int getExtraInputSize() {return extraInputs.size();}
	
	
	/**
	 * Getting extra input at specified index.
	 * @param index specified index.
	 * @return extra input at specified index.
	 */
	public Object getExtraInput(int index) {
		return extraInputs != null && extraInputs.size() > 0 && index >= 0 && index < extraInputs.size() ? extraInputs.get(index) : null; 
	}

	
	/**
	 * Getting first extra input.
	 * @return first extra input.
	 */
	public Object extraInput() {
		return extraInputs != null && extraInputs.size() > 0 ? extraInputs.get(0) : null; 
	}
	
	
	/**
	 * Getting second extra input.
	 * @return second extra input.
	 */
	public Object extraInput2() {
		return extraInputs != null && extraInputs.size() > 1 ? extraInputs.get(1) : null; 
	}

	
	/**
	 * Adding extra input.
	 * @param extraInput extra input.
	 * @return true if adding is successful.
	 */
	public boolean addExtraInput(Object extraInput) {
		return this.extraInputs.add(extraInput);
	}
	
	
	/**
	 * Removing extra input at specified index.
	 * @param index specified index.
	 * @return previous extra input.
	 */
	public Object removeExtraInput(int index) {
		return this.extraInputs.remove(index);
	}
	
	
	/**
	 * Clearing extra inputs.
	 */
	public void clearExtraInputs() {
		extraInputs.clear();
	}
	
	
	/**
	 * Creating input records.
	 * @param inputs input matrices.
	 * @return input records.
	 */
	public static Record[] createInputs(Matrix...inputs) {
		if (inputs == null || inputs.length == 0) return null;
		Record[] inputRecords = new Record[inputs.length];
		for (int i = 0; i < inputs.length; i++) inputRecords[i] = new Record(inputs[i]);
		return inputRecords;
	}
	
	
}


