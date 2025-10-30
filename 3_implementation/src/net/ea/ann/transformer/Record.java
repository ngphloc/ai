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

/**
 * This class represents record for training attention.
 * @author Loc Nguyen
 * @version 1.0
 */
public class Record implements Cloneable, Serializable {
	
	
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
	public Record() {
		
	}
	
	
	/**
	 * Constructor with Y input data, attention output data, X input data, and input mask.
	 * @param inputY Y input data.
	 * @param outputA attention output data.
	 * @param inputX X input data.
	 * @param inputMask input mask.
	 */
	public Record(Matrix inputY, Matrix outputA, Matrix inputX, boolean[][] inputMask) {
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
	 * Creating record with Y input data, X input data, and input mask.
	 * @param inputY Y input data.
	 * @param inputX X input data.
	 * @param inputMask input mask.
	 * @return input record.
	 */
	public static Record createInput(Matrix inputY, Matrix inputX, boolean[][] inputMask) {
		Record record = new Record();
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
		Record record = new Record();
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
		if (this.inputMask != null) maneRecord.addExtraInput(maneRecord);
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
	
	
	/**
	 * Creating transformer record from matrix neural network record.
	 * @param maneRecord matrix neural network records.
	 * @return transformer record.
	 */
	public static Record create(net.ea.ann.mane.Record maneRecord) {
		Record record = new Record(maneRecord.input(), maneRecord.output(), maneRecord.input2());
		Object extraInput = maneRecord.extraInput();
		if ((extraInput != null) && (extraInput instanceof boolean[][])) record.inputMask = (boolean[][])extraInput;
		return record;
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
	
	
}


