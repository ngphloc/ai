/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.rnn.lstm;

import java.util.List;

import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.LayerStandardAbstract;
import net.ea.ann.core.NetworkStandardAbstract;
import net.ea.ann.core.NeuronStandardAssoc;
import net.ea.ann.core.WeightedNeuron;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.function.ReLU;
import net.ea.ann.core.generator.GeneratorStandard.Neuron;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.vector.NeuronValueVector;
import net.ea.ann.rnn.RecurrentNetwork;
import net.ea.ann.rnn.State;

/**
 * This class represents a cell in long short-term memory.
 * 
 * @author Loc Nguyen
 * @version 1.0
 */
public class Cell extends Neuron {
	
	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Number of gates.
	 */
	public final static int GATE_NUMBERS = 4;

	
	/**
	 * Input gate.
	 */
	public final static int INPUT_GATE = 0;
	
	
	/**
	 * Forget gate.
	 */
	public final static int FORGET_GATE =1;

	
	/**
	 * Output gate.
	 */
	public final static int OUTPUT_GATE = 2;

	
	/**
	 * Cell gate.
	 */
	public final static int CELL_GATE = 3;

	
	/**
	 * Getting stored memory.
	 */
	protected NeuronValue c = null;
	
	
	/**
	 * Getting displayed memory.
	 */
	protected NeuronValue h = null;

	
	/**
	 * Constructor with layer.
	 * @param layer layer.
	 */
	public Cell(LayerStandard layer) {
		super(layer);
		resetCellState();
	}


	/**
	 * Getting zero element.
	 * @return zero element.
	 */
	private NeuronValue getZeroElement() {
		NeuronValue zero = getBias().zero();
		if (zero instanceof NeuronValueVector)
			return ((NeuronValueVector)zero).get(0);
		else
			return zero;
	}
	
	
	/**
	 * Getting state.
	 * @return current state.
	 */
	public State getState() {
		LayerStandard layer = getLayer();
		if (layer == null) return null;
		if (!(layer instanceof LayerStandardAbstract)) return null;
		NetworkStandardAbstract network = ((LayerStandardAbstract)layer).getNetwork();
		return network != null && network instanceof State ? (State)network : null;
	}


	@Override
	public void setInput(NeuronValue value) {
		super.setInput(RecurrentNetwork.verify(value, this));
	}


	@Override
	public void setOutput(NeuronValue value) {
		super.setOutput(RecurrentNetwork.verify(value, this));
	}


	/**
	 * Getting gate at specified index.
	 * @param v specified value.
	 * @param gateIndex specified gate index.
	 * @return gate at specified index.
	 */
	private static NeuronValue getGate(NeuronValue v, int gateIndex) {
		if ((v == null) || !(v instanceof NeuronValueVector)) return null;
		NeuronValueVector value = (NeuronValueVector)v;
		return value.get(gateIndex);
	}
	
	
	/**
	 * Setting gate at specified index.
	 * @param v specified value.
	 * @param gateIndex specified gate index.
	 * @param element element.
	 * @return replaced element.
	 */
	private static NeuronValue setGate(NeuronValue v, int gateIndex, NeuronValue element) {
		if ((v == null) || !(v instanceof NeuronValueVector)) return null;
		NeuronValueVector value = (NeuronValueVector)v;
		return value.set(gateIndex, element);
	}

	
	/**
	 * Getting gate at specified index.
	 * @param gateIndex specified gate index.
	 * @return gate at specified index.
	 */
	private NeuronValue getGate(int gateIndex) {
		return getGate(getOutput(), gateIndex);
	}
	
	
	/**
	 * Setting gate at specified index.
	 * @param gateIndex specified gate index.
	 * @param element element.
	 * @return replaced element.
	 */
	private NeuronValue setGate(int gateIndex, NeuronValue element) {
		return setGate(getOutput(), gateIndex, element);
	}
	
	
	/**
	 * Getting input gate.
	 * @return input gate.
	 */
	public NeuronValue getInputGate() {
		return getGate(INPUT_GATE);
	}
	
	
	/**
	 * Setting input gate.
	 * @param value specified gate value.
	 * @return replaced gate value.
	 */
	protected NeuronValue setInputGate(NeuronValue value) {
		return setGate(INPUT_GATE, value);
	}

	
	/**
	 * Getting forget gate.
	 * @return forget gate.
	 */
	public NeuronValue getForgetGate() {
		return getGate(FORGET_GATE);
	}

	
	/**
	 * Setting forget gate.
	 * @param value specified gate value.
	 * @return replaced gate value.
	 */
	protected NeuronValue setForgetGate(NeuronValue value) {
		return setGate(FORGET_GATE, value);
	}

	
	/**
	 * Getting output gate.
	 * @return output gate.
	 */
	public NeuronValue getOutputGate() {
		return getGate(OUTPUT_GATE);
	}

	
	/**
	 * Setting output gate.
	 * @param value specified gate value.
	 * @return replaced gate value.
	 */
	protected NeuronValue setOutputGate(NeuronValue value) {
		return setGate(OUTPUT_GATE, value);
	}

	
	/**
	 * Getting cell gate.
	 * @return cell gate.
	 */
	public NeuronValue getCellGate() {
		return getGate(CELL_GATE);
	}

	
	/**
	 * Setting cell gate.
	 * @param value specified gate value.
	 * @return replaced gate value.
	 */
	protected NeuronValue setCellGate(NeuronValue value) {
		return setGate(CELL_GATE, value);
	}

	
	/**
	 * Getting stored memory.
	 * @return stored memory.
	 */
	public NeuronValue getCellState() {
		return c;
	}
	
	
	/**
	 * Getting displayed memory.
	 * @return displayed memory.
	 */
	public NeuronValue getCellStateOutput() {
		return h;
	}

	
	/**
	 * Resetting cell state.
	 */
	protected void resetCellState() {
		this.h = this.c = getZeroElement();
	}
	
	
	@Override
	public NeuronValue evaluate() {
		super.evaluate();
			
		resetCellState();

		NeuronValue remember = getInputGate().multiply(getCellGate());
		List<WeightedNeuron> sources = new NeuronStandardAssoc(this).getSources();
		if (sources.size() == 0)
			this.c = remember;
		else {
			NeuronValue sourcesCellState = getZeroElement();
			for (WeightedNeuron source : sources) {
				if (!(source.neuron instanceof Cell)) continue;
				sourcesCellState = sourcesCellState.add(((Cell)source.neuron).getCellState());
			}
			this.c = getForgetGate().multiply(sourcesCellState).add(remember);
		}
		this.h = getOutputGate().multiply(getActivateRef().evaluate(this.c));
		
		NeuronValue input = null;
		Function auxActivateRef = getAuxActivateRef();
		if (auxActivateRef == null || auxActivateRef instanceof ReLU) {
			if (auxActivateRef != null) this.h = auxActivateRef.evaluate(this.h);
			updateOutputByCellStateOutput();
			input = getOutput();
		}
		else {
			updateOutputByCellStateOutput();
			input = getOutput().duplicate();
			this.h = auxActivateRef.evaluate(this.h);
			updateOutputByCellStateOutput();
		}
		
		setInput(input);
		return getOutput();
	}
	
	
	/**
	 * Updating output by cell state output.
	 */
	private void updateOutputByCellStateOutput() {
		setInputGate(this.h);
		setForgetGate(this.h);
		setOutputGate(this.h);
		setCellGate(this.h);
	}
	
	
}
