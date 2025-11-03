/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.transformer;

import java.io.Serializable;
import java.util.List;

import net.ea.ann.core.Util;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.mane.Record.Inout;

/**
 * This class represents record for training transformer.
 * @author Loc Nguyen
 * @version 1.0
 */
public class Record implements Cloneable, Serializable {
	
	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * List of internal records.
	 */
	public List<Record0> records = Util.newList(0);
	
	
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
	 * Constructor of array of records.
	 * @param inouts array of records.
	 */
	public Record(Record0...records) {
		if (records == null || records.length == 0) return;
		for (int i = 0; i < records.length; i++) {
			if (records[i] != null) this.records.add(records[i]);
		}
	}
	
	
	/**
	 * Constructor with Y input data, attention output data, X input data, and input mask.
	 * @param inputY Y input data.
	 * @param outputA attention output data.
	 * @param inputX X input data.
	 * @param inputMask input mask.
	 */
	public Record(Matrix inputY, Matrix outputA, Matrix inputX, boolean[][] inputMask) {
		this(new Record0(inputY, outputA, inputX, inputMask));
	}
	

	/**
	 * Constructor with Y input data, attention output data, and X input data.
	 * @param inputY Y input data.
	 * @param outputA attention output data.
	 * @param inputX X input data.
	 */
	public Record(Matrix inputY, Matrix outputA, Matrix inputX) {
		this(inputY, outputA, inputX, null);
	}
	

	/**
	 * Constructor with Y input data and attention output data.
	 * @param inputY Y input data.
	 * @param outputA attention output data.
	 */
	public Record(Matrix inputY, Matrix outputA) {
		this(inputY, outputA, null);
	}


	/**
	 * Getting size of record.
	 * @return size of record.
	 */
	public int size() {return records.size();}
	
	
	/**
	 * Getting record at specified index.
	 * @param index specified index.
	 * @return record at specified index.
	 */
	public Record0 get(int index) {return records.get(index);}
	
	
	/**
	 * Adding record.
	 * @param record record.
	 * @return true if adding is successful.
	 */
	public boolean add(Record0 record) {
		return this.records.add(record);
	}
	
	
	/**
	 * Adding record.
	 * @param inputY Y input data.
	 * @param outputA attention output data.
	 * @param inputX X input data.
	 * @param inputMask input mask.
	 */
	public boolean add(Matrix inputY, Matrix outputA, Matrix inputX, boolean[][] inputMask) {
		return add(new Record0(inputY, outputA, inputX, inputMask));
	}

	
	/**
	 * Adding record.
	 * @param inputY Y input data.
	 * @param outputA attention output data.
	 * @param inputX X input data.
	 */
	public boolean add(Matrix inputY, Matrix outputA, Matrix inputX) {
		return add(new Record0(inputY, outputA, inputX));
	}
	

	/**
	 * Adding record.
	 * @param inputY Y input data.
	 * @param outputA attention output data.
	 */
	public boolean add(Matrix inputY, Matrix outputA) {
		return add(new Record0(inputY, outputA));
	}

	
	/**
	 * Removing record at specified index.
	 * @param index specified index.
	 * @return previous record.
	 */
	public Record0 remove(int index) {
		return this.records.remove(index);
	}
	
	
	/**
	 * Clearing inputs and outputs.
	 */
	public void clear() {
		this.records.clear();
	}


	/**
	 * Getting first record.
	 * @return first record.
	 */
	public Record0 get() {return size() > 0 ? get(0) : null;}


	/**
	 * Getting second record.
	 * @return second record.
	 */
	public Record0 get2() {return size() > 1 ? get(1) : null;}


	/**
	 * Getting Y input.
	 * @return Y input.
	 */
	public Matrix inputY() {
		Record0 record = get();
		return record != null ? record.inputY : null;
	}
	
	
	/**
	 * Getting attention output.
	 * @return attention output.
	 */
	public Matrix outputA() {
		Record0 record = get();
		return record != null ? record.outputA : null;
	}

	
	/**
	 * Getting X input.
	 * @return X input.
	 */
	public Matrix inputX() {
		Record0 record = get();
		return record != null ? record.inputX : null;
	}

	
	/**
	 * Getting input mask.
	 * @return input mask.
	 */
	public boolean[][] inputMask() {
		Record0 record = get();
		return record != null ? record.inputMask : null;
	}
	
	
	/**
	 * Getting Y input.
	 * @return Y input.
	 */
	public Matrix inputY2() {
		Record0 record = get2();
		return record != null ? record.inputY : null;
	}
	
	
	/**
	 * Getting attention output.
	 * @return attention output.
	 */
	public Matrix outputA2() {
		Record0 record = get2();
		return record != null ? record.outputA : null;
	}

	
	/**
	 * Getting X input.
	 * @return X input.
	 */
	public Matrix inputX2() {
		Record0 record = get2();
		return record != null ? record.inputX : null;
	}

	
	/**
	 * Getting input mask.
	 * @return input mask.
	 */
	public boolean[][] inputMask2() {
		Record0 record = get2();
		return record != null ? record.inputMask : null;
	}

	
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
	 * Creating transformer record from matrix neural network record.
	 * @param maneRecord matrix neural network records.
	 * @return transformer record.
	 */
	public static Record create(net.ea.ann.mane.Record maneRecord) {
		if (maneRecord == null || maneRecord.size() == 0) return null;
		List<Record0> record0s = Util.newList(maneRecord.size());
		for (int i = 0; i < maneRecord.size(); i++) {
			Record0 record0 = null;
			if (i == 0) {
				record0 = new Record0(maneRecord.input(), maneRecord.output(), maneRecord.input2());
				Object extraInput = maneRecord.extraInput();
				if ((extraInput != null) && (extraInput instanceof boolean[][])) record0.inputMask = (boolean[][])extraInput;
			}
			else {
				record0 = new Record0(maneRecord.input(i), maneRecord.output(i));
			}
			if (record0 != null) record0s.add(record0);
		}

		return record0s.size() > 0 ? new Record(record0s.toArray(new Record0[] {})) : null;
	}
	
	
	/**
	 * Creating transformer records from matrix neural network records.
	 * @param maneRecords matrix neural network records.
	 * @return transformer records.
	 */
	public static List<Record> create(Iterable<net.ea.ann.mane.Record> maneRecords) {
		List<Record> records = Util.newList(0);
		for (net.ea.ann.mane.Record maneRecord : maneRecords) {
			if (maneRecord == null) continue;
			Record record = create(maneRecord);
			if (record != null) records.add(record);
		}
		return records;
	}
	
	
	/**
	 * Creating record with Y input data, X input data, and input mask.
	 * @param inputY Y input data.
	 * @param inputX X input data.
	 * @param inputMask input mask.
	 * @return input record.
	 */
	public static Record createInput(Matrix inputY, Matrix inputX, boolean[][] inputMask) {
		Record0 record = Record0.createInput(inputY, inputX, inputMask);
		return record != null ? new Record(record) : null;
	}
	
	
	/**
	 * Creating record with Y input data and X input data.
	 * @param inputY Y input data.
	 * @param inputX X input data.
	 * @return input record.
	 */
	public static Record createInput(Matrix inputY, Matrix inputX) {
		return createInput(inputY, inputX, null);
	}

	
	/**
	 * Creating record with input data.
	 * @param input input data.
	 * @return input record.
	 */
	public static Record createInput(Matrix input) {
		return createInput(input, null, null);
	}

	
	/**
	 * Creating record with output.
	 * @param output output.
	 * @return output.
	 */
	public static Record createOutput(Matrix output) {
		Record0 record = Record0.createOutput(output);
		return record != null ? new Record(record) : null;
	}
	
	
	/**
	 * Converting transformer record into matrix neural network record.
	 * @param record transformer record.
	 * @return matrix neural network record.
	 */
	public net.ea.ann.mane.Record convert() {
		if (size() == 0) return null;
		if (size() == 1) {
			Record0 record0 = get();
			if (record0 == null) return null;
			net.ea.ann.mane.Record maneRecord = new net.ea.ann.mane.Record(record0.inputY, record0.outputA, record0.inputX, null);
			if (this.inputMask() != null) maneRecord.addExtraInput(this.inputMask());
			return maneRecord;
		}
		
		List<Inout> inouts = Util.newList(0);
		for (int i = 0; i < size(); i++) {
			Record0 record0 = get(i);
			if (record0 == null) continue;
			inouts.add(new Inout(record0.inputY, record0.outputA));
			inouts.add(new Inout(record0.inputX, null));
		}
		if (inouts.size() == 0) return null;
		net.ea.ann.mane.Record maneRecord = new net.ea.ann.mane.Record(inouts.toArray(new Inout[] {}));
		if (this.inputMask() != null) maneRecord.addExtraInput(this.inputMask());
		return maneRecord;
	}
	
	
	/**
	 * Converting transformer records into matrix neural network records.
	 * @param records transformer records.
	 * @return matrix neural network records.
	 */
	public static List<net.ea.ann.mane.Record> convert(Iterable<Record> records) {
		List<net.ea.ann.mane.Record> maneRecords = Util.newList(0);
		for (Record record : records) {
			if (record == null) continue;
			net.ea.ann.mane.Record maneRecord = record.convert();
			if (maneRecord != null) maneRecords.add(maneRecord);
		}
		return maneRecords;
	}


}



/**
 * This class represents basic record for training transformer.
 * @author Loc Nguyen
 * @version 1.0
 */
class Record0 implements Cloneable, Serializable {
	
	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Y input data.
	 */
	public Matrix inputY = null;
	
	
	/**
	 * Attention output data.
	 */
	public Matrix outputA = null;
	
	
	/**
	 * X input data.
	 */
	public Matrix inputX = null;
	
	
	/**
	 * Masked input matrix.
	 */
	boolean[][] inputMask = null;
	
	
	/**
	 * Default constructor.
	 */
	public Record0() {
		
	}
	
	
	/**
	 * Constructor with Y input data, attention output data, X input data, and input mask.
	 * @param inputY Y input data.
	 * @param outputA attention output data.
	 * @param inputX X input data.
	 * @param inputMask input mask.
	 */
	public Record0(Matrix inputY, Matrix outputA, Matrix inputX, boolean[][] inputMask) {
		this.inputY = inputY;
		this.outputA = outputA;
		this.inputX = inputX;
		this.inputMask = inputMask;
	}

	
	/**
	 * Constructor with Y input data, attention output data, and X input data.
	 * @param inputY Y input data.
	 * @param outputA attention output data.
	 * @param inputX X input data.
	 */
	public Record0(Matrix inputY, Matrix outputA, Matrix inputX) {
		this(inputY, outputA, inputX, null);
	}
	

	/**
	 * Constructor with Y input data and attention output data.
	 * @param inputY Y input data.
	 * @param outputA attention output data.
	 */
	public Record0(Matrix inputY, Matrix outputA) {
		this(inputY, outputA, null);
	}

	
	/**
	 * Creating transformer record from matrix neural network record.
	 * @param maneRecord matrix neural network records.
	 * @return transformer record.
	 */
	public static Record0 create(net.ea.ann.mane.Record maneRecord) {
		Record0 record = new Record0(maneRecord.input(), maneRecord.output(), maneRecord.input2());
		Object extraInput = maneRecord.extraInput();
		if ((extraInput != null) && (extraInput instanceof boolean[][])) record.inputMask = (boolean[][])extraInput;
		return record;
	}
	
	
	/**
	 * Creating transformer records from matrix neural network records.
	 * @param maneRecords matrix neural network records.
	 * @return transformer records.
	 */
	public static List<Record0> create(Iterable<net.ea.ann.mane.Record> maneRecords) {
		List<Record0> records = Util.newList(0);
		for (net.ea.ann.mane.Record maneRecord : maneRecords) {
			if (maneRecord == null) continue;
			Record0 record = create(maneRecord);
			if (record != null) records.add(record);
		}
		return records;
	}
	
	
	/**
	 * Creating record with Y input data, X input data, and input mask.
	 * @param inputY Y input data.
	 * @param inputX X input data.
	 * @param inputMask input mask.
	 * @return input record.
	 */
	public static Record0 createInput(Matrix inputY, Matrix inputX, boolean[][] inputMask) {
		Record0 record = new Record0();
		record.inputY = inputY;
		record.inputX = inputX;
		record.inputMask = inputMask;
		return record;
	}
	
	
	/**
	 * Creating record with Y input data and X input data.
	 * @param inputY Y input data.
	 * @param inputX X input data.
	 * @return input record.
	 */
	public static Record0 createInput(Matrix inputY, Matrix inputX) {
		return createInput(inputY, inputX, null);
	}

	
	/**
	 * Creating record with input data.
	 * @param input input data.
	 * @return input record.
	 */
	public static Record0 createInput(Matrix input) {
		return createInput(input, null, null);
	}

	
	/**
	 * Creating record with output.
	 * @param output output.
	 * @return output.
	 */
	public static Record0 createOutput(Matrix output) {
		Record0 record = new Record0();
		record.outputA = output;
		return record;
	}
	
	
	/**
	 * Converting transformer record into matrix neural network record.
	 * @param record transformer record.
	 * @return matrix neural network record.
	 */
	public net.ea.ann.mane.Record convert() {
		net.ea.ann.mane.Record maneRecord = new net.ea.ann.mane.Record(this.inputY, this.outputA, this.inputX, null);
		if (this.inputMask != null) maneRecord.addExtraInput(this.inputMask);
		return maneRecord;
	}
	
	
	/**
	 * Converting transformer records into matrix neural network records.
	 * @param records transformer records.
	 * @return matrix neural network records.
	 */
	public static List<net.ea.ann.mane.Record> convert(Iterable<Record0> records) {
		List<net.ea.ann.mane.Record> maneRecords = Util.newList(0);
		for (Record0 record : records) {
			if (record == null) continue;
			net.ea.ann.mane.Record maneRecord = record.convert();
			if (maneRecord != null) maneRecords.add(maneRecord);
		}
		return maneRecords;
	}
	
	
}



