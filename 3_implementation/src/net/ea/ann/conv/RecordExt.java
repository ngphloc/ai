/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import net.ea.ann.core.Record;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Raster;

/**
 * This class is extensive sample record for learning neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class RecordExt extends Record {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Content input for backbone.
	 */
	public Content[] contentInput = null;
	
	
	/**
	 * Content output for backbone. It can be null.
	 */
	public Content[] contentOutput = null;

	
	/**
	 * Default constructor.
	 */
	public RecordExt() {

	}

	
	/**
	 * Default constructor with input.
	 * @param input specified input.
	 */
	public RecordExt(NeuronValue[] input) {
		super(input);
	}


	/**
	 * Default constructor with input and output.
	 * @param input specified input.
	 * @param output specified output.
	 */
	public RecordExt(NeuronValue[] input, NeuronValue[] output) {
		super(input, output);
	}


	/**
	 * Default constructor with content input.
	 * @param contentInput specified content input.
	 */
	public RecordExt(Content[] contentInput) {
		this.contentInput = contentInput;
	}


	/**
	 * Default constructor with content input and content output.
	 * @param contentInput specified content input.
	 * @param contentOutput specified content output.
	 */
	public RecordExt(Content[] contentInput, Content[] contentOutput) {
		this.contentInput = contentInput;
		this.contentOutput = contentOutput;
	}


	/**
	 * Constructor with raster input.
	 * @param rasterInput raster input.
	 */
	public RecordExt(Raster rasterInput) {
		super(rasterInput);
	}


	/**
	 * Constructor with raster input and raster output.
	 * @param rasterInput raster input.
	 * @param rasterOutput raster output.
	 */
	public RecordExt(Raster rasterInput, Raster rasterOutput) {
		super(rasterInput, rasterOutput);
	}


	/**
	 * Constructor with other record.
	 * @param record other record.
	 */
	public RecordExt(RecordExt record) {
		this();
		if (record != null) record.transfer(this);
	}


	@Override
	public void reverse() {
		super.reverse();
		Content[] contentInputTemp = this.contentInput;
		this.contentInput = this.contentOutput;
		this.contentOutput = contentInputTemp;
	}


	@Override
	public void transferInputs(Record outRecord) {
		super.transferInputs(outRecord);
		
		RecordExt recordExt = (outRecord != null) && (outRecord instanceof RecordExt) ? (RecordExt)outRecord : null;
		if (recordExt == null) return;
		
		recordExt.contentInput = this.contentInput;
	}


	@Override
	public void transferOutputs(Record outRecord) {
		super.transferOutputs(outRecord);
		
		RecordExt recordExt = (outRecord != null) && (outRecord instanceof RecordExt) ? (RecordExt)outRecord : null;
		if (recordExt == null) return;
		
		recordExt.contentOutput = this.contentOutput;
	}


}
