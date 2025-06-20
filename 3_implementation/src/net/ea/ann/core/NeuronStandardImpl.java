/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Set;

import net.ea.ann.core.function.Function;
import net.ea.ann.core.function.FunctionDelay;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.Weight;

/**
 * This class is default implementation of standard neuron.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class NeuronStandardImpl implements NeuronStandard, TextParsable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Identifier.
	 */
	protected int id = -1;
	
	
	/**
	 * Main layer.
	 */
	protected LayerStandard layer = null;
	
	
	/**
	 * Input value.
	 */
	protected NeuronValue input = null;
	
	
	/**
	 * Bias.
	 */
	protected NeuronValue bias = null;

	
	/**
	 * Output value.
	 */
	protected NeuronValue output = null;
			
	
	/**
	 * Activation function reference.
	 */
	protected Function activateRef = null;
	
	
	/**
	 * Next neurons.
	 */
	protected List<WeightedNeuron> nextNeurons = Util.newList(0);
	
	
	/**
	 * Output rib neurons.
	 */
	protected List<WeightedNeuron> riboutNeurons = Util.newList(0);

	
	/**
	 * Inside next neurons.
	 */
	protected List<WeightedNeuron> insideNextNeurons = Util.newList(0);

	
	/**
	 * Outside previous neurons.
	 */
	protected Set<WeightedNeuron> outsidePrevNeurons = Util.newSet(0);

	
	/**
	 * Outside next neurons.
	 */
	protected Set<WeightedNeuron> outsideNextNeurons = Util.newSet(0);

	
	/**
	 * Constructor with standard layer.
	 * @param layer this layer.
	 */
	public NeuronStandardImpl(LayerStandard layer) {
		this.layer = layer;
		this.id = layer.getIdRef().get();
		this.setActivateRef(layer.getActivateRef());
		this.setBias(layer.newBias().zero());
		
		NeuronValue zero = layer.newNeuronValue().zero();
		this.setInput(zero);
		this.setOutput(zero);
	}

	
	@Override
	public int id() {
		return id;
	}

	
	@Override
	public NeuronValue getValue() {
		return getOutput();
	}


	@Override
	public NeuronValue getInput() {
		return input;
	}

	
	@Override
	public void setInput(NeuronValue value) {
		this.input = value;
	}

	
	@Override
	public NeuronValue getBias() {
		return bias;
	}


	@Override
	public void setBias(NeuronValue bias) {
		this.bias = bias;
	}


	@Override
	public NeuronValue getOutput() {
		return output;
	}

	
	@Override
	public void setOutput(NeuronValue value) {
		this.output = value;
	}

	
	@Override
	public Function getActivateRef() {
		return activateRef;
	}

	
	@Override
	public Function setActivateRef(Function activateRef) {
		return this.activateRef = activateRef;
	}

	
	@Override
	public WeightedNeuron[] getPrevNeurons() {
		List<WeightedNeuron> sources = Util.newList(0);
		if (layer == null) return sources.toArray(new WeightedNeuron[] {});
		
		LayerStandard prevLayer = layer.getPrevLayer();
		if (prevLayer == null) return sources.toArray(new WeightedNeuron[] {});
		
		for (int i = 0; i < prevLayer.size(); i++) {
			NeuronStandard prevNeuron = prevLayer.get(i);
			WeightedNeuron found = prevNeuron.findNextNeuron(this);
			if (found != null) {
				WeightedNeuron wn = new WeightedNeuron(prevNeuron, found.weight);
				sources.add(wn);
			}
		}
		
		return sources.toArray(new WeightedNeuron[] {});
	}
	

	@Override
	public WeightedNeuron[] getPrevNeurons(LayerStandard prevLayer) {
		if (layer == null || prevLayer == null || prevLayer == layer.getPrevLayer())
			return getPrevNeurons();
		
		if (!(layer instanceof LayerStandardImpl)) return new WeightedNeuron[] {};
		LayerStandard prevLayerImplicit = layer.getPrevLayerImplicit();
		if (prevLayer == prevLayerImplicit)
			return getPrevNeuronsImplicit();
		else
			return new WeightedNeuron[] {};
	}


	@Override
	public WeightedNeuron[] getPrevNeuronsImplicit() {
		if (layer == null || !(layer instanceof LayerStandardImpl)) return new WeightedNeuron[] {};
		
		LayerStandard prevLayerImplicit = layer.getPrevLayerImplicit();
		//Although given a rib-out layer, by default, its implicit previous layer is the layer on backbone to which it attaches,
		//The condition prevLayerImplicit.getRiboutLayer() != layer will be improved because it is too strict.
		if (prevLayerImplicit == null || prevLayerImplicit.getRiboutLayer() != layer)
			return new WeightedNeuron[] {};
		
		List<WeightedNeuron> wns = Util.newList(0);
		for (int i = 0; i < prevLayerImplicit.size(); i++) {
			NeuronStandard prevNeuron = prevLayerImplicit.get(i);
			WeightedNeuron nw = prevNeuron.findRiboutNeuron(this); //This code line will be improved.
			if (nw != null) {
				wns.add(new WeightedNeuron(prevNeuron, nw.weight));
			}
		}
		
		return wns.toArray(new WeightedNeuron[] {});
	}


	@Override
	public WeightedNeuron[] getNextNeurons() {
		return nextNeurons.toArray(new WeightedNeuron[] {});
	}


	@Override
	public WeightedNeuron[] getNextNeurons(LayerStandard nextLayer) {
		if (nextLayer == null || layer == null || layer.getNextLayer() == nextLayer)
			return getNextNeurons();
		else if (nextLayer == layer.getRiboutLayer())
			return riboutNeurons.toArray(new WeightedNeuron[] {});
		else
			return new WeightedNeuron[] {};
	}


	@Override
	public boolean setNextNeuron(NeuronStandard neuron, Weight weight) {
		LayerStandard nextLayer = layer != null ? layer.getNextLayer() : null;
		if (nextLayer == null || neuron == null || weight == null)
			return false;
		if (nextLayer.indexOf(neuron) < 0) return false;
		
		WeightedNeuron wn = findNextNeuron(neuron);
		if (wn == null) {
			wn = new WeightedNeuron(neuron, weight);
			nextNeurons.add(wn);
		}
		else {
			wn.weight.value = weight.value;
		}
		
		return true;
	}

	
	@Override
	public boolean removeNextNeuron(NeuronStandard neuron) {
		if (neuron == null) return false;
		for (int i = 0; i < nextNeurons.size(); i++) {
			if (nextNeurons.get(i).neuron == neuron) {
				nextNeurons.remove(i);
				return true;
			}
		}

		return false;
	}

	
	@Override
	public void clearNextNeurons() {
		List<WeightedNeuron> wns = Util.newList(this.nextNeurons.size());
		wns.addAll(this.nextNeurons);
		
		for (WeightedNeuron wn : wns) {
			removeNextNeuron(wn.neuron);
		}
		
		this.nextNeurons.clear();
	}


	@Override
	public WeightedNeuron findNextNeuron(NeuronStandard neuron) {
		for (int i = 0; i < nextNeurons.size(); i++) {
			WeightedNeuron wn = nextNeurons.get(i);
			if (wn.neuron == neuron) return wn;
		}
		
		return null;
	}
	
	
	@Override
	public WeightedNeuron findNextNeuron(int neuronId) {
		for (int i = 0; i < nextNeurons.size(); i++) {
			WeightedNeuron wn = nextNeurons.get(i);
			if (wn.neuron != null && wn.neuron.id() == neuronId) return wn;
		}
		
		return null;
	}


	@Override
	public WeightedNeuron[] getRibinNeurons() {
		LayerStandard ribinLayer = layer != null ? layer.getRibinLayer() : null;
		if (ribinLayer == null) return new WeightedNeuron[] {};

		List<WeightedNeuron> ribinNeurons = Util.newList(0);
		for (int i = 0; i < ribinLayer.size(); i++) {
			NeuronStandard ribinNeuron = ribinLayer.get(i);
			WeightedNeuron[] wns = ribinNeuron.getNextNeurons();
			for (WeightedNeuron wn : wns) {
				if (wn.neuron == this) {
					ribinNeurons.add(new WeightedNeuron(ribinNeuron, wn.weight));
				}
			}
		}

		return ribinNeurons.toArray(new WeightedNeuron[] {});
	}


	@Override
	public boolean setRibinNeuron(NeuronStandard ribinNeuron, Weight weight) {
		LayerStandard ribinLayer = layer != null ? layer.getRibinLayer() : null;
		if (ribinLayer == null || ribinNeuron == null || weight == null) return false;
		if (ribinLayer.indexOf(ribinNeuron) < 0) return false;
		
		return ribinNeuron.setNextNeuron(this, weight);
	}


	@Override
	public boolean removeRibinNeuron(NeuronStandard ribinNeuron) {
		LayerStandard ribinLayer = layer != null ? layer.getRibinLayer() : null;
		if (ribinLayer == null || ribinNeuron == null) return false;
		if (ribinLayer.indexOf(ribinNeuron) < 0) return false;

		return ribinNeuron.removeNextNeuron(this);
	}


	@Override
	public WeightedNeuron findRibinNeuron(NeuronStandard ribinNeuron) {
		LayerStandard ribinLayer = layer != null ? layer.getRibinLayer() : null;
		if (ribinLayer == null || ribinNeuron == null) return null;
		if (ribinLayer.indexOf(ribinNeuron) < 0) return null;

		WeightedNeuron wn = ribinNeuron.findNextNeuron(this);
		if (wn == null)
			return null;
		else
			return new WeightedNeuron(ribinNeuron, wn.weight);
	}


	@Override
	public WeightedNeuron findRibinNeuron(int ribinNeuronId) {
		LayerStandard ribinLayer = layer != null ? layer.getRibinLayer() : null;
		if (ribinLayer == null) return null;
		int index = ribinLayer.indexOf(ribinNeuronId);
		if (index < 0) return null;

		NeuronStandard ribinNeuron = ribinLayer.get(index);
		WeightedNeuron wn = ribinNeuron.findNextNeuron(this);
		if (wn == null)
			return null;
		else
			return new WeightedNeuron(ribinNeuron, wn.weight);
	}


	@Override
	public void clearRibinNeurons() {
		LayerStandard ribinLayer = layer != null ? layer.getRibinLayer() : null;
		if (ribinLayer == null) return;
		for (int i = 0; i < ribinLayer.size(); i++) {
			ribinLayer.get(i).removeNextNeuron(this);
		}
	}


	@Override
	public WeightedNeuron[] getRiboutNeurons() {
		return riboutNeurons.toArray(new WeightedNeuron[] {});
	}


	@Override
	public boolean setRiboutNeuron(NeuronStandard riboutNeuron, Weight weight) {
		LayerStandard ribinLayer = layer != null ? layer.getRiboutLayer() : null;
		if (ribinLayer == null || riboutNeuron == null || weight == null)
			return false;
		if (ribinLayer.indexOf(riboutNeuron) < 0) return false;
		
		WeightedNeuron wn = findRiboutNeuron(riboutNeuron);
		if (wn == null) {
			wn = new WeightedNeuron(riboutNeuron, weight);
			riboutNeurons.add(wn);
		}
		else {
			wn.weight.value = weight.value;
		}
		
		return true;
	}


	@Override
	public boolean removeRiboutNeuron(NeuronStandard riboutNeuron) {
		if (riboutNeuron == null) return false;
		for (int i = 0; i < riboutNeurons.size(); i++) {
			if (riboutNeurons.get(i).neuron == riboutNeuron) {
				riboutNeurons.remove(i);
				return true;
			}
		}

		return false;
	}


	@Override
	public WeightedNeuron findRiboutNeuron(NeuronStandard riboutNeuron) {
		for (int i = 0; i < riboutNeurons.size(); i++) {
			WeightedNeuron wn = riboutNeurons.get(i);
			if (wn.neuron == riboutNeuron) return wn;
		}
		
		return null;
	}


	@Override
	public WeightedNeuron findRiboutNeuron(int riboutNeuronId) {
		for (int i = 0; i < riboutNeurons.size(); i++) {
			WeightedNeuron wn = riboutNeurons.get(i);
			if (wn.neuron != null && wn.neuron.id() == riboutNeuronId) return wn;
		}
		
		return null;
	}


	@Override
	public void clearRiboutNeurons() {
		riboutNeurons.clear();
	}


	/**
	 * Finding neuron.
	 * @param wns collection of neurons.
	 * @param neuron specified neuron.
	 * @return found neuron.
	 */
	private static WeightedNeuron findNeuron(Collection<WeightedNeuron> wns, NeuronStandard neuron) {
		for (WeightedNeuron wn : wns) {
			if (wn.neuron == neuron) return wn;
		}
		return null;
	}


	/**
	 * Adding neuron.
	 * @param wns collection of neurons.
	 * @param neuron specified neuron.
	 * @param weight specified weight.
	 * @return true if adding is successful.
	 */
	private static boolean addNeuron(Collection<WeightedNeuron> wns, NeuronStandard neuron, Weight weight) {
		if (neuron == null || weight == null) return false;
		if (findNeuron(wns, neuron) != null) return false;
		return wns.add(new WeightedNeuron(neuron, weight));
	}


	/**
	 * Removing neuron.
	 * @param wns collection of neurons.
	 * @param neuron specified neuron.
	 * @return true if removal is successful.
	 */
	private static boolean removeNeuron(Collection<WeightedNeuron> wns, NeuronStandard neuron) {
		WeightedNeuron found = findNeuron(wns, neuron);
		return found != null ? wns.remove(found) : false;
	}


	@Override
	public List<WeightedNeuron> getInsidePrevNeurons() {
		List<WeightedNeuron> prevNeurons = Util.newList(0);
		for (WeightedNeuron wn : insideNextNeurons) {
			if (wn.neuron == this) prevNeurons.add(wn);
		}
		return prevNeurons;
	}


	@Override
	public List<WeightedNeuron> getInsideNextNeurons() {
		return insideNextNeurons;
	}

	
	@Override
	public Set<WeightedNeuron> getOutsidePrevNeurons() {
		return outsidePrevNeurons;
	}


	@Override
	public WeightedNeuron findOutsidePrevNeuron(NeuronStandard outsidePrevNeuron) {
		return findNeuron(outsidePrevNeurons, outsidePrevNeuron);
	}


	@Override
	public boolean addOutsidePrevNeuron(NeuronStandard outsidePrevNeuron, Weight weight) {
		boolean added = addNeuron(outsidePrevNeurons, outsidePrevNeuron, weight);
		if (!added) return added;
		added = added && addNeuron(((NeuronStandardImpl)outsidePrevNeuron).outsideNextNeurons, this, weight);
		return added;
	}


	@Override
	public boolean removeOutsidePrevNeuron(NeuronStandard outsidePrevNeuron) {
		boolean removed = removeNeuron(outsidePrevNeurons, outsidePrevNeuron);
		if (!removed) return removed;
		removed = removed && removeNeuron(((NeuronStandardImpl)outsidePrevNeuron).outsideNextNeurons, this);
		return removed;
	}


	@Override
	public void clearOutsidePrevNeurons() {
		Set<WeightedNeuron> tempOutsidePrevNeurons = Util.newSet(outsidePrevNeurons.size());
		tempOutsidePrevNeurons.addAll(outsidePrevNeurons);
		for (WeightedNeuron wn : tempOutsidePrevNeurons) removeOutsidePrevNeuron(wn.neuron);

		outsidePrevNeurons.clear();
	}


	@Override
	public Set<WeightedNeuron> getOutsideNextNeurons() {
		return outsideNextNeurons;
	}


	@Override
	public Set<WeightedNeuron> getOutsideNextNeurons(LayerStandard layer) {
		Set<WeightedNeuron> wns = getOutsideNextNeurons();
		Set<WeightedNeuron> nextNeurons = Util.newSet(0);
		for (WeightedNeuron wn : wns) {
			if (wn.neuron.getLayer() == layer) nextNeurons.add(wn);
		}
		return nextNeurons;
	}
	
	
	@Override
	public WeightedNeuron findOutsideNextNeuron(NeuronStandard outsideNextNeuron) {
		return findNeuron(outsideNextNeurons, outsideNextNeuron);
	}


	@Override
	public boolean addOutsideNextNeuron(NeuronStandard outsideNextNeuron, Weight weight) {
		boolean added = addNeuron(outsideNextNeurons, outsideNextNeuron, weight);
		if (!added) return added;
		added = added && addNeuron(((NeuronStandardImpl)outsideNextNeuron).outsidePrevNeurons, this, weight);
		return added;
	}


	@Override
	public boolean removeOutsideNextNeuron(NeuronStandard outsideNextNeuron) {
		boolean removed = removeNeuron(outsideNextNeurons, outsideNextNeuron);
		if (!removed) return removed;
		removed = removed && removeNeuron(((NeuronStandardImpl)outsideNextNeuron).outsidePrevNeurons, this);
		return removed;
	}


	@Override
	public void clearOutsideNextNeurons() {
		Set<WeightedNeuron> tempOutsideNextNeurons = Util.newSet(outsideNextNeurons.size());
		tempOutsideNextNeurons.addAll(outsideNextNeurons);
		for (WeightedNeuron wn : tempOutsideNextNeurons) removeOutsideNextNeuron(wn.neuron);
		
		outsideNextNeurons.clear();
	}

	
	@Override
	public NeuronStandard getPrevSibling() {
		if (layer == null) return null;
		
		int index = layer.indexOf(this);
		if (index <= 0)
			return null;
		else
			return layer.get(index - 1);
	}

	
	@Override
	public NeuronStandard getNextSibling() {
		if (layer == null) return null;
		
		int index = layer.indexOf(this);
		if (index < 0 || index >= layer.size() - 1)
			return null;
		else
			return layer.get(index + 1);
	}

	
	@Override
	public LayerStandard getLayer() {
		return layer;
	}


	/**
	 * Getting all weighted source neurons.
	 * @return all weighted source neurons.
	 */
	List<WeightedNeuron> getSources() {
		Set<WeightedNeuron> sources = Util.newSet(0);
		sources.addAll(Arrays.asList(getPrevNeurons()));
		sources.addAll(Arrays.asList(getRibinNeurons()));
		sources.addAll(Arrays.asList(getPrevNeuronsImplicit()));
		
		sources.addAll(getInsidePrevNeurons());
		sources.addAll(getOutsidePrevNeurons());
		
		List<WeightedNeuron> list = Util.newList(sources.size());
		list.addAll(sources);
		return list;
	}
	
	
	@Override
	public NeuronValue evaluate() {
		List<WeightedNeuron> sources = getSources();
		if (sources.size() == 0) {
			NeuronValue out = getInput();
			setOutput(out);
			return out;
		}
		
		NeuronValue in = getBias();
		for (WeightedNeuron source : sources) {
			NeuronValue element = source.neuron.getOutput().multiply(source.weight.value);
			in = in.add(element);
		}
		
		setInput(in);
		Function f = getActivateRef();
		NeuronValue out = ((f != null) && !(f instanceof FunctionDelay)) ? in.evaluate(f) : in;
		setOutput(out);
		return out;
	}


	@Override
	public NeuronValue derivative() {
		Function f = getActivateRef();
		if (f == null) return getBias().unit();
		NeuronValue derivativeInput = NeuronStandard.getDerivativeInput(this);
		return derivativeInput != null ? derivativeInput.derivative(f) : null;
	}


	/**
	 * Verbalize neuron.
	 * @param neuron specific neuron.
	 * @param tab tab text.
	 * @return verbalized text.
	 */
	protected static String toText(NeuronStandard neuron, String tab) {
		StringBuffer buffer = new StringBuffer();
		String internalTab = "    ";
		buffer.append("neuron n## (id=" + neuron.id() + "):");
		
		buffer.append("\n" + internalTab);
		NeuronValue input = neuron.getInput();
		buffer.append("input = " + (input != null ? input.toString() : null));

		buffer.append("\n" + internalTab);
		NeuronValue output = neuron.getOutput();
		buffer.append("output = " + (output != null ? output.toString() : null));

		buffer.append("\n" + internalTab);
		NeuronValue bias = neuron.getBias();
		buffer.append("bias = " + (bias != null ? bias.toString() : null));

		WeightedNeuron[] nexts = neuron.getNextNeurons();
		for (int i = 0; i < nexts.length; i++) {
			buffer.append("\n" + internalTab);
			buffer.append(nexts[i].weight + " -> neuron id=" + nexts[i].neuron.id() + " (layer id=" + nexts[i].neuron.getLayer().id() + ")");
		}
		
		String text = buffer.toString();
		if (tab != null && !tab.isEmpty()) {
			text = tab + text; text = text.replaceAll("\n", "\n" + tab);
		}
		return text;
		
	}
	

	@Override
	public String toText() {
		try {
			String text = toText(this, null);
			text = text.replaceAll("n##", "");
			return text;
		}
		catch (Throwable e) {}
		
		return super.toString();
	}

	
}
