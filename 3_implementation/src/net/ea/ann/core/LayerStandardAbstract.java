/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import java.util.List;

import net.ea.ann.core.function.Function;
import net.ea.ann.core.function.IdentityFunction1;

/**
 * This class is abstract implementation of standard layer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class LayerStandardAbstract extends LayerAbstract implements LayerStandard {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;
	
	
	/**
	 * Activation function reference.
	 */
	protected Function activateRef = null;


	/**
	 * Internal neurons.
	 */
	protected List<NeuronStandard> neurons = Util.newList(0);
	
	
	/**
	 * Previous layer.
	 */
	protected LayerStandard prevLayer = null;
	
	
	/**
	 * Implicit previous layer.
	 */
	protected LayerStandard prevLayerImplicit = null;

	
	/**
	 * Next layer.
	 */
	protected LayerStandard nextLayer = null;
	
	
	/**
	 * Input rib layer.
	 */
	protected LayerStandard ribinLayer = null;

	
	/**
	 * Output rib layer.
	 */
	protected LayerStandard riboutLayer = null;
	
	
	/**
	 * Constructor with neuron channel, activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param idRef identifier reference.
	 */
	protected LayerStandardAbstract(int neuronChannel, Function activateRef, Id idRef) {
		super(idRef);
		
		if (neuronChannel <= 1 && activateRef == null) {
			this.neuronChannel = 1;
			this.activateRef = new IdentityFunction1();
		}
		else {
			this.neuronChannel = neuronChannel;
			this.activateRef = activateRef;
		}
	}


	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	protected LayerStandardAbstract(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Default constructor.
	 * @param neuronChannel neuron channel.
	 */
	protected LayerStandardAbstract(int neuronChannel) {
		this(neuronChannel, null, null);
	}

	
	@Override
	public NeuronStandard newNeuron() {
		return new NeuronStandardImpl(this);
	}

	
	/**
	 * Create a new weight.
	 * @return new weight.
	 */
	private Weight newWeight() {
		return new Weight(newNeuronValue().newWeightValue().zero());
	}
	
	
	@Override
	public int size() {
		return neurons.size();
	}

	
	@Override
	public NeuronStandard get(int index) {
		return neurons.get(index);
	}

	
	@Override
	public boolean add(NeuronStandard neuron) {
		return neurons.add(neuron);
	}

	
	@Override
	public NeuronStandard remove(int index) {
		NeuronStandard neuron = neurons.get(index);
		neuron.clearNextNeurons();
		neuron.clearRiboutNeurons();
		
		return neurons.remove(index);
	}

	
	@Override
	public void clear() {
		while (neurons.size() > 0) {
			remove(0);
		}
	}


	@Override
	public int indexOf(NeuronStandard neuron) {
		return neurons.indexOf(neuron);
	}

	
	@Override
	public int indexOf(int neuronId) {
		for (int i = 0; i < neurons.size(); i++) {
			if (neurons.get(i).id() == neuronId) return i;
		}
		
		return -1;
	}


	@Override
	public LayerStandard getPrevLayer() {
		return prevLayer;
	}

	
	@Override
	public LayerStandard getPrevLayerImplicit() {
		return prevLayerImplicit;
	}
	
	
	@Override
	public boolean setPrevLayer(LayerStandard prevLayer) {
		if (prevLayer == this.prevLayer) return false;
		if (this.prevLayer == null && this.prevLayerImplicit != null) return false;
		if (prevLayer != null && prevLayer == getRibinLayer()) return false;

		LayerStandard oldPrevLayer = this.prevLayer;
		LayerStandard oldPrevPrevLayer = null;
		if (oldPrevLayer != null) {
			oldPrevPrevLayer = oldPrevLayer.getPrevLayer();
			clearNextNeurons(oldPrevLayer);
		}

		this.prevLayer = prevLayer;
		if (prevLayer == null) return true;

		clearNextNeurons(prevLayer);
		((LayerStandardImpl)prevLayer).nextLayer = this;
		for (int i = 0; i < prevLayer.size(); i++) {
			NeuronStandard neuron = prevLayer.get(i);
			for (int j = 0; j < size(); j++) {
				neuron.setNextNeuron(get(j), newWeight());
			}
		}
		
		if (oldPrevPrevLayer == null) return true;
		clearNextNeurons(oldPrevPrevLayer);
		((LayerStandardImpl)oldPrevPrevLayer).nextLayer = prevLayer;
		((LayerStandardImpl)prevLayer).prevLayer = oldPrevPrevLayer;
		for (int i = 0; i < oldPrevPrevLayer.size(); i++) {
			NeuronStandard neuron = oldPrevPrevLayer.get(i);
			for (int j = 0; j < prevLayer.size(); j++) {
				neuron.setNextNeuron(prevLayer.get(j), newWeight());
			}
		}
		
		return true;
	}


	@Override
	public LayerStandard getNextLayer() {
		return nextLayer;
	}

	
	@Override
	public boolean setNextLayer(LayerStandard nextLayer) {
		if (nextLayer == this.nextLayer) return false;
		if (nextLayer != null) {
			if (nextLayer == getRiboutLayer()) return false;
			if (nextLayer.getRibinLayer() == this) return false;
		}
		
		clearNextNeurons(this);
		
		LayerStandard oldNextLayer = this.nextLayer;
		LayerStandard oldNextNextLayer = null;
		if (oldNextLayer != null) {
			oldNextNextLayer = oldNextLayer.getNextLayer();
			clearNextNeurons(oldNextLayer);
		}

		this.nextLayer = nextLayer;
		if (nextLayer == null) return true;

		clearNextNeurons(nextLayer);
		((LayerStandardImpl)nextLayer).prevLayer = this;
		for (int i = 0; i < size(); i++) {
			NeuronStandard neuron = get(i);
			for (int j = 0; j < nextLayer.size(); j++) {
				neuron.setNextNeuron(nextLayer.get(j), newWeight());
			}
		}
		
		if (oldNextNextLayer == null) return true;
		((LayerStandardImpl)oldNextNextLayer).prevLayer = nextLayer;
		((LayerStandardImpl)nextLayer).nextLayer = oldNextNextLayer;
		for (int i = 0; i < oldNextNextLayer.size(); i++) {
			NeuronStandard neuron = oldNextNextLayer.get(i);
			for (int j = 0; j < nextLayer.size(); j++) {
				neuron.setNextNeuron(nextLayer.get(j), newWeight());
			}
		}
		
		return true;
	}


	/**
	 * Clearing next neurons of specified layer.
	 * @param layer specified layer.
	 */
	private static void clearNextNeurons(LayerStandard layer) {
		if (layer == null) return;
		for (int i = 0; i < layer.size(); i++) {
			layer.get(i).clearNextNeurons();
		}
	}
	
	
	@Override
	public LayerStandard getRibinLayer() {
		return ribinLayer;
	}


	@Override
	public boolean setRibinLayer(LayerStandard ribinLayer) {
		if (this.ribinLayer == ribinLayer) return false;
		if (ribinLayer != null) {
			if (ribinLayer.getNextLayer() != null) return false;
			if (ribinLayer == getPrevLayer()) return false;
		}
		
		this.ribinLayer = ribinLayer;
		if (ribinLayer == null) return true;
			
		clearNextNeurons(ribinLayer);
		LayerStandard oldNextLayer = ribinLayer.getNextLayer();
		if (oldNextLayer != null) clearNextNeurons(oldNextLayer);

		((LayerStandardImpl)ribinLayer).nextLayer = this;
		for (int i = 0; i < ribinLayer.size(); i++) {
			NeuronStandard ribbinNeuron = ribinLayer.get(i);
			for (int j = 0; j < size(); j++) {
				ribbinNeuron.setNextNeuron(get(j), newWeight());
			}
		}
		
		return true;
	}

	
	@Override
	public LayerStandard getRiboutLayer() {
		return riboutLayer;
	}


	@Override
	public boolean setRiboutLayer(LayerStandard riboutLayer) {
		if (this.riboutLayer == riboutLayer) return false;
		if (riboutLayer != null && riboutLayer.getPrevLayer() != null) return false;
		if (riboutLayer != null) {
			if (riboutLayer.getPrevLayer() != null) return false;
			if (riboutLayer == getNextLayer()) return false;
		}
		
		LayerStandard oldRiboutLayer = this.riboutLayer;
		this.riboutLayer = riboutLayer;
		for (NeuronStandard neuron : neurons) ((NeuronStandardImpl)neuron).riboutNeurons.clear();
		
		if (oldRiboutLayer != null) ((LayerStandardImpl)oldRiboutLayer).prevLayerImplicit = null;
		if (riboutLayer == null) return true;
		
		for (NeuronStandard neuron : neurons) {
			for (int i = 0; i < riboutLayer.size(); i++) {
				WeightedNeuron wn = new WeightedNeuron(riboutLayer.get(i), newWeight());
				((NeuronStandardImpl)neuron).riboutNeurons.add(wn);
			}
		}
		
		((LayerStandardImpl)riboutLayer).prevLayerImplicit = this;
		
		return true;
	}


	@Override
	public Function getActivateRef() {
		return this.activateRef;
	}

	
	@Override
	public Function setActivateRef(Function activateRef) {
		return this.activateRef = activateRef;
	}


	@Override
	public NeuronValue[] getInput() {
		if (neurons.size() == 0) return null;
		NeuronValue[] array = new NeuronValue[neurons.size()];
		for (int j = 0; j < array.length; j++) {
			array[j] = neurons.get(j).getInput();
		}
		return array;
	}


	@Override
	public NeuronValue[] getOutput() {
		if (neurons.size() == 0) return null;
		NeuronValue[] array = new NeuronValue[neurons.size()];
		for (int j = 0; j < array.length; j++) {
			array[j] = neurons.get(j).getOutput();
		}
		return array;
	}


	/**
	 * Verbalize layer.
	 * @param layer specific layer.
	 * @param tab tab text.
	 * @return verbalized text.
	 */
	protected static String toText(LayerStandard layer, String tab) {
		StringBuffer buffer = new StringBuffer();
		String internalTab = "    ";
		buffer.append("layer l## (id=" + layer.id() + "):");
		for (int i = 0; i < layer.size(); i++) {
			buffer.append("\n");

			String neuronText = NeuronStandardImpl.toText(layer.get(i), internalTab);
			neuronText = neuronText.replaceAll("n##", "" + (i+1));
			buffer.append(neuronText);
		}
		
		String text = buffer.toString();
		if (tab != null && !tab.isEmpty()) {
			text = tab + text; text = text.replaceAll("\n", "\n" + tab);
		}
		return text;
	}


	/**
	 * Converting this layer to text.
	 * @return converted text.
	 */
	public String toText() {
		try {
			String text = toText(this, null);
			text = text.replaceAll("l##", "");
			return text;
		}
		catch (Throwable e) {}
		
		return super.toString();
	}

	
//	/**
//	 * Getting previous neurons of specified neuron.
//	 * @param neuron specified neuron.
//	 * @return previous neurons of specified neuron.
//	 */
//	protected WeightedNeuron[] getImplicitPrevNeurons(Neuron neuron) {
//		if (neuron == null || prevLayer != null || prevLayerImplicit == null)
//			return new WeightedNeuron[] {};
//		
//		if (!neurons.contains(neuron)) return new WeightedNeuron[] {};
//		Layer rLayer = prevLayerImplicit.getRiboutLayer();
//		if (this != rLayer) return new WeightedNeuron[] {};
//		
//		List<WeightedNeuron> wns = Util.newList(0);
//		for (int i = 0; i < prevLayerImplicit.size(); i++) {
//			Neuron prevNeuron = prevLayerImplicit.get(i);
//			WeightedNeuron nw = prevNeuron.findRiboutNeuron(neuron);
//			if (nw != null) {
//				wns.add(new WeightedNeuron(prevNeuron, nw.weight));
//			}
//		}
//		
//		return wns.toArray(new WeightedNeuron[] {});
//	}


}
