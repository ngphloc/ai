/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import java.io.Serializable;
import java.util.List;
import java.util.Map;
import java.util.Random;

import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Raster;

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
	 * Backbone input. Two most important fields are {@link #input} and {@link #output}.
	 */
	public NeuronValue[] input = null;
	
	
	/**
	 * Backbone output. It can be null.
	 */
	public NeuronValue[] output = null;
	
	
	/**
	 * Rib-in bone input whose key is attached layer index. It can be null.
	 */
	public Map<Integer, NeuronValue[]> ribinInput = Util.newMap(0);

	
	/**
	 * Rib-in bone output whose key is attached layer index. It can be null.
	 */
	public Map<Integer, NeuronValue[]> ribinOutput = Util.newMap(0);
	
	
	/**
	 * Rib-out bone input whose key is attached layer index. It can be null.
	 */
	public Map<Integer, NeuronValue[]> riboutInput = Util.newMap(0);

	
	/**
	 * Rib-out bone output whose key is attached layer index. It can be null.
	 */
	public Map<Integer, NeuronValue[]> riboutOutput = Util.newMap(0);

	
	/**
	 * Memory input. It can be null.
	 */
	public NeuronValue[] memInput = null;

	
	/**
	 * Memory output. It can be null.
	 */
	public NeuronValue[] memOutput = null;
	
	
	/**
	 * Undefined input. It is often raster.
	 */
	public Object undefinedInput = null;

	
	/**
	 * Undefined output. It is often raster.
	 */
	public Object undefinedOutput = null;

	
	/**
	 * Default constructor.
	 */
	public Record() {

	}

	
	/**
	 * Default constructor with input.
	 * @param input specified input.
	 */
	public Record(NeuronValue[] input) {
		this.input = input;
	}


	/**
	 * Default constructor with input and output.
	 * @param input specified input.
	 * @param output specified output.
	 */
	public Record(NeuronValue[] input, NeuronValue[] output) {
		this.input = input;
		this.output = output;
	}

	
	/**
	 * Constructor with raster input.
	 * @param rasterInput raster input.
	 */
	public Record(Raster rasterInput) {
		this.undefinedInput = rasterInput;
	}
	

	/**
	 * Constructor with raster input and raster output.
	 * @param rasterInput raster input.
	 * @param rasterOutput raster output.
	 */
	public Record(Raster rasterInput, Raster rasterOutput) {
		this.undefinedInput = rasterInput;
		this.undefinedOutput = rasterOutput;
	}
	
	
	/**
	 * Constructor with matrix input.
	 * @param matrixInput matrix input.
	 */
	public Record(Matrix matrixInput) {
		this.undefinedInput = matrixInput;
	}

	
	/**
	 * Constructor with matrix input and matrix output.
	 * @param matrixInput matrix input.
	 * @param matrixOutput matrix output.
	 */
	public Record(Matrix matrixInput, Matrix matrixOutput) {
		this.undefinedInput = matrixInput;
		this.undefinedOutput = matrixOutput;
	}

	
	/**
	 * Constructor with other record.
	 * @param record other record.
	 */
	public Record(Record record) {
		this();
		if (record != null) record.transfer(this);
	}

	
	/**
	 * Reversing input into output and vice versa.
	 */
	public void reverse( ) {
		NeuronValue[] inputTemp = this.input;
		Map<Integer, NeuronValue[]> ribinInputTemp = this.ribinInput;
		Map<Integer, NeuronValue[]> riboutInputTemp = this.riboutInput;
		NeuronValue[] memInputTemp = this.memInput;
		Object undefinedInputTemp = this.undefinedInput;

		this.input = this.output;
		this.ribinInput = this.ribinOutput;
		this.riboutInput = this.riboutOutput;
		this.memInput = this.memOutput;
		this.undefinedInput = this.undefinedOutput;
		
		this.output = inputTemp;
		this.ribinOutput = ribinInputTemp;
		this.riboutOutput = riboutInputTemp;
		this.memOutput = memInputTemp;
		this.undefinedOutput = undefinedInputTemp;
	}
	
	
	/**
	 * Transferring inputs to other record.
	 * @param outRecord other record.
	 */
	public void transferInputs(Record outRecord) {
		if (outRecord == null) return;
		
		outRecord.input = this.input;
		outRecord.ribinInput = this.ribinInput;
		outRecord.riboutInput = this.riboutInput;
		outRecord.memInput = this.memInput;
		outRecord.undefinedInput = this.undefinedInput;
	}
	
	
	/**
	 * Transferring outputs to other record.
	 * @param outRecord other record.
	 */
	public void transferOutputs(Record outRecord) {
		if (outRecord == null) return;
		
		outRecord.output = this.output;
		outRecord.ribinOutput = this.ribinOutput;
		outRecord.riboutOutput = this.riboutOutput;
		outRecord.memOutput = this.memOutput;
		outRecord.undefinedOutput = this.undefinedOutput;
	}


	/**
	 * Transferring inputs and outputs to other record.
	 * @param outRecord other record.
	 */
	public void transfer(Record outRecord) {
		transferInputs(outRecord);
		transferOutputs(outRecord);
	}
	

	/**
	 * Getting raster input.
	 * @return raster input.
	 */
	public Raster getRasterInput() {
		return (undefinedInput != null) && (undefinedInput instanceof Raster) ? (Raster)undefinedInput : null;
	}
	
	
	/**
	 * Setting raster input.
	 * @param rasterInput raster input.
	 */
	public void setRasterInput(Raster rasterInput) {
		this.undefinedInput = rasterInput;
	}
	
	
	/**
	 * Getting raster output.
	 * @return raster output.
	 */
	public Raster getRasterOutput() {
		return (undefinedOutput != null) && (undefinedOutput instanceof Raster) ? (Raster)undefinedOutput : null;
	}


	/**
	 * Setting raster output.
	 * @param rasterOutput raster output.
	 */
	public void setRasterOutput(Raster rasterOutput) {
		this.undefinedOutput = rasterOutput;
	}


	/**
	 * Getting matrix input.
	 * @return matrix input.
	 */
	public Matrix getMatrixInput() {
		return (undefinedInput != null) && (undefinedInput instanceof Matrix) ? (Matrix)undefinedInput : null;
	}
	
	
	/**
	 * Setting matrix input.
	 * @param matrixInput matrix input.
	 */
	public void setMatrixInput(Matrix matrixInput) {
		this.undefinedInput = matrixInput;
	}
	
	
	/**
	 * Getting matrix output.
	 * @return matrix output.
	 */
	public Matrix getMatrixOutput() {
		return (undefinedOutput != null) && (undefinedOutput instanceof Matrix) ? (Matrix)undefinedOutput : null;
	}


	/**
	 * Setting matrix output.
	 * @param matrixOutput matrix output.
	 */
	public void setMatrixOutput(Matrix matrixOutput) {
		this.undefinedOutput = matrixOutput;
	}

	
	/**
	 * Randomizing sample flag.
	 */
	public static boolean RANDOM_SAMPLE = false;
	
	
	/**
	 * Getting size of collection.
	 * @param <T> type.
	 * @param collection collection.
	 * @return size of collection.
	 */
	public static <T> int sizeOf(Iterable<T> collection) {
		int size = 0;
		for (T element : collection) {
			if (element != null) size++;
		}
		return size;
	}

	
	/**
	 * Converting collection into list. 
	 * @param <T> type.
	 * @param collection collection.
	 * @return list.
	 */
	public static <T> List<T> listOf(Iterable<T> collection) {
		List<T> list = Util.newList(0); 
		if (collection == null) return list;
		for (T element : collection) {
			if (element != null) list.add(element);
		}
		return list;
	}
	
	
	/**
	 * Re-sampling records.
	 * @param <T> type.
	 * @param records specified records.
	 * @param size sample size.
	 * @return re-sampled record list.
	 */
	private static <T> List<T> resample(List<T> records, int size) {
		int n = records.size();
		if (n == 0) return records;
		size = size < 1 || size > n ? n : size;
		List<T> sample = Util.newList(size);
		Random rnd = new Random();
		for (int i = 0; i < size; i++) {
			int index = rnd.nextInt(n);
			sample.add(records.get(index));
		}
		return sample;
	}
	
	
	/**
	 * Re-sampling records.
	 * @param <T> type.
	 * @param records specified records.
	 * @param size sample size.
	 * @return re-sampled record list.
	 */
	private static <T> List<T> resample(Iterable<T> records, int size) {
		List<T> list = listOf(records);
		return resample(list, size);
	}


	/**
	 * Re-sampling records.
	 * @param <T> type.
	 * @param records specified records.
	 * @return re-sampled record list.
	 */
	public static <T> List<T> resample(Iterable<T> records, double rate) {
		rate = 0 < rate && rate <= 1 ? rate : 1;
		int size = (int) (rate * (double)sizeOf(records));
		return resample(records, size);
	}
	
	
}
