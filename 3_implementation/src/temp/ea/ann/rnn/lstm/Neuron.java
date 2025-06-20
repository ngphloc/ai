/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package temp.ea.ann.rnn.lstm;

import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.NeuronStandardImpl;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.indexed.IndexedNeuronValue;

/**
 * This class represents an neuron in long short-term memory.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Neuron extends NeuronStandardImpl {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Number of parameters
	 */
	public final static int PARAMS_NUM = 4;
	
	
	/**
	 * Index of forget gate.
	 */
	public final static int PARAMS_FORGETGATE_INDEX = 0;


	/**
	 * Index of input gate.
	 */
	public final static int PARAMS_INPUTGATE_INDEX = 1;


	/**
	 * Index of output gate.
	 */
	public final static int PARAMS_OUTPUTGATE_INDEX = 2;

	
	/**
	 * Index of activation gate.
	 */
	public final static int PARAMS_ACTIVATEGATE_INDEX = 3;

	
	/**
	 * Cell input activation vector.
	 */
	protected NeuronValue activateCell = null;


	/**
	 * Constructor with standard layer.
	 * @param layer this layer.
	 */
	public Neuron(Layer layer) {
		super(layer);
		this.activateCell = layer.newCellValue();
	}

	
	/**
	 * Updating value.
	 * @return indexed output.
	 */
	private NeuronValue updateValue() {
		NeuronValue output = getOutput();
		if ((output  == null) || !(output instanceof IndexedNeuronValue)) return output;
		IndexedNeuronValue indexedOutput = (IndexedNeuronValue)output;
		if (indexedOutput.size() == 0) return output;
		if (indexedOutput.size() <= PARAMS_FORGETGATE_INDEX) return output;
		if (indexedOutput.size() <= PARAMS_INPUTGATE_INDEX) return output;
		if (indexedOutput.size() <= PARAMS_OUTPUTGATE_INDEX) return output;
		if (indexedOutput.size() <= PARAMS_ACTIVATEGATE_INDEX) return output;
		
		this.activateCell = this.activateCell.multiply(indexedOutput.get(PARAMS_FORGETGATE_INDEX)).add(
				indexedOutput.get(PARAMS_INPUTGATE_INDEX).multiply(indexedOutput.get(PARAMS_ACTIVATEGATE_INDEX))
			);
		
		NeuronValue piece = indexedOutput.get(PARAMS_OUTPUTGATE_INDEX).multiply(getActivateRef().evaluate(this.activateCell));
		indexedOutput.set(PARAMS_FORGETGATE_INDEX, piece);
		indexedOutput.set(PARAMS_INPUTGATE_INDEX, piece);
		indexedOutput.set(PARAMS_OUTPUTGATE_INDEX, piece);
		indexedOutput.set(PARAMS_ACTIVATEGATE_INDEX, piece);
		
		setInput(indexedOutput);
		setOutput(indexedOutput);
		return indexedOutput;
	}


	@Override
	public NeuronValue evaluate() {
		LayerStandard standardLayer = getLayer();
		if ((standardLayer == null) || !(standardLayer instanceof Layer)) return super.evaluate();
		Layer layer = (Layer)standardLayer;
		int oldParamIndex = layer.getParamIndex();
		
		layer.setParamIndex(PARAMS_FORGETGATE_INDEX); super.evaluate();
		layer.setParamIndex(PARAMS_INPUTGATE_INDEX); super.evaluate();
		layer.setParamIndex(PARAMS_OUTPUTGATE_INDEX); super.evaluate();
		layer.setParamIndex(PARAMS_ACTIVATEGATE_INDEX); super.evaluate();
		NeuronValue output = updateValue();
		
		layer.setParamIndex(oldParamIndex);
		return output;
	}

	
}
