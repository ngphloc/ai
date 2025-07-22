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
import java.util.Set;

import net.ea.ann.core.function.Function;
import net.ea.ann.core.function.FunctionDelay;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValue1;
import net.ea.ann.core.value.Weight;
import net.ea.ann.raster.Raster;

/**
 * This class is abstract implementation of standard layer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class LayerStandardAbstract extends LayerAbstract implements LayerStandard, TextParsable {


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
	 * By default, given a rib-out layer, its implicit previous layer is the layer (in the backbone) connecting to it.
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
		
		if (neuronChannel < 1)
			this.neuronChannel = neuronChannel = 1;
		else
			this.neuronChannel = neuronChannel;
		this.activateRef = activateRef == null ? (activateRef = Raster.toActivationRef(this.neuronChannel, true)) : activateRef;
	}


	/**
	 * Getting network if possible.
	 * @return network if possible.
	 */
	public NetworkStandardAbstract getNetwork() {return null;}
	
	
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

	
	@Override
	public Weight newWeight() {
		return new Weight(newNeuronValue().newWeightValue().zeroW());
	}
	
	
	@Override
	public NeuronValue newBias() {
		return newNeuronValue().zero();
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
	public boolean hasSomePrevLayers() {
		return getAllPrevLayers().size() > 0;
	}


	@Override
	public Set<LayerStandard> getAllPrevLayers() {
		Set<LayerStandard> prevLayers = Util.newSet(0);
		if (prevLayer != null && prevLayer.size() > 0) prevLayers.add(prevLayer);
		if (ribinLayer != null && ribinLayer.size() > 0) prevLayers.add(ribinLayer);
		if (prevLayerImplicit != null && prevLayerImplicit.size() > 0) prevLayers.add(prevLayerImplicit);
		
		LayerStandard insidePrevVirtualLayer = getInsidePrevVirtualLayer();
		if (insidePrevVirtualLayer != null && insidePrevVirtualLayer.size() > 0) prevLayers.add(insidePrevVirtualLayer);
		LayerStandard outsidePrevVirtualLayer = getOutsidePrevVirtualLayer();
		if (outsidePrevVirtualLayer != null && outsidePrevVirtualLayer.size() > 0) prevLayers.add(outsidePrevVirtualLayer);
		return prevLayers;
	}


	@Override
	public boolean setPrevLayer(LayerStandard prevLayer) {
		return setPrevLayer(prevLayer, false);
	}


	/**
	 * Setting previous layer.
	 * @param prevLayer previous layer. It can be null.
	 * @param injective if this parameter is true, there is only one connection between two neurons.
	 * @return true if setting is successful.
	 */
	protected boolean setPrevLayer(LayerStandard prevLayer, boolean injective) {
		if (prevLayer == this.prevLayer) return false;
		if (this.prevLayer == null && this.prevLayerImplicit != null) return false;
		if (prevLayer != null && prevLayer == getRibinLayer()) return false;

		LayerStandard oldPrevLayer = this.prevLayer;
		if (oldPrevLayer != null) {
			clearNextNeurons(oldPrevLayer);
			((LayerStandardImpl)oldPrevLayer).nextLayer = null;
		}
		this.prevLayer = prevLayer;
		if (prevLayer == null) return true;

		clearNextNeurons(prevLayer);
		LayerStandard nextLayerOfPrevLayer = ((LayerStandardImpl)prevLayer).nextLayer;
		if (nextLayerOfPrevLayer != null) ((LayerStandardImpl)nextLayerOfPrevLayer).prevLayer = null;
		
		((LayerStandardImpl)prevLayer).nextLayer = this;
		if (injective) {
			int n = Math.min(prevLayer.size(), size());
			for (int i = 0; i < n; i++) {
				prevLayer.get(i).setNextNeuron(get(i), newWeight());
			}
		}
		else {
			for (int i = 0; i < prevLayer.size(); i++) {
				NeuronStandard neuron = prevLayer.get(i);
				for (int j = 0; j < size(); j++) {
					neuron.setNextNeuron(get(j), newWeight());
				}
			}
		}
		
		return true;
	}

	
	/**
	 * Replacing previous layer.
	 * @param prevLayer previous layer. It can be null.
	 * @return true if setting is successful.
	 */
	protected boolean replacePrevLayer(LayerStandard prevLayer) {
		return replacePrevLayer(prevLayer, false);
	}


	/**
	 * Replacing previous layer.
	 * @param prevLayer previous layer. It can be null.
	 * @param injective if this parameter is true, there is only one connection between two neurons.
	 * @return true if setting is successful.
	 */
	protected boolean replacePrevLayer(LayerStandard prevLayer, boolean injective) {
		if (prevLayer == this.prevLayer) return false;
		if (this.prevLayer == null && this.prevLayerImplicit != null) return false;
		if (prevLayer != null && prevLayer == getRibinLayer()) return false;

		LayerStandard oldPrevLayer = this.prevLayer;
		LayerStandard oldPrevPrevLayer = null;
		if (oldPrevLayer != null) {
			oldPrevPrevLayer = oldPrevLayer.getPrevLayer();
			clearNextNeurons(oldPrevLayer);
			((LayerStandardImpl)oldPrevLayer).nextLayer = null;
		}
		this.prevLayer = prevLayer;
		if (prevLayer == null) return true;

		clearNextNeurons(prevLayer);
		LayerStandard nextLayerOfPrevLayer = ((LayerStandardImpl)prevLayer).nextLayer;
		if (nextLayerOfPrevLayer != null) ((LayerStandardImpl)nextLayerOfPrevLayer).prevLayer = null;
		
		((LayerStandardImpl)prevLayer).nextLayer = this;
		if (injective) {
			int n = Math.min(prevLayer.size(), size());
			for (int i = 0; i < n; i++) {
				prevLayer.get(i).setNextNeuron(get(i), newWeight());
			}
		}
		else {
			for (int i = 0; i < prevLayer.size(); i++) {
				NeuronStandard neuron = prevLayer.get(i);
				for (int j = 0; j < size(); j++) {
					neuron.setNextNeuron(get(j), newWeight());
				}
			}
		}
		
		if (oldPrevPrevLayer == null) return true;
		clearNextNeurons(oldPrevPrevLayer);
		((LayerStandardImpl)oldPrevPrevLayer).nextLayer = prevLayer;
		((LayerStandardImpl)prevLayer).prevLayer = oldPrevPrevLayer;
		if (injective) {
			int n = Math.min(oldPrevPrevLayer.size(), prevLayer.size());
			for (int i = 0; i < n; i++) {
				oldPrevPrevLayer.get(i).setNextNeuron(prevLayer.get(i), newWeight());
			}
		}
		else {
			for (int i = 0; i < oldPrevPrevLayer.size(); i++) {
				NeuronStandard neuron = oldPrevPrevLayer.get(i);
				for (int j = 0; j < prevLayer.size(); j++) {
					neuron.setNextNeuron(prevLayer.get(j), newWeight());
				}
			}
		}
		
		return true;
	}

	
	@Override
	public LayerStandard getNextLayer() {
		return nextLayer;
	}

	
	@Override
	public boolean hasSomeNextLayers() {
		return getAllNextLayers().size() > 0;
	}


	@Override
	public Set<LayerStandard> getAllNextLayers() {
		Set<LayerStandard> nextLayers = Util.newSet(0);
		if (nextLayer != null && nextLayer.size() > 0) nextLayers.add(nextLayer);
		if (riboutLayer != null && riboutLayer.size() > 0) nextLayers.add(riboutLayer);
		
		LayerStandard insideNextVirtualLayer = getInsideNextVirtualLayer();
		if (insideNextVirtualLayer != null && insideNextVirtualLayer.size() > 0) nextLayers.add(insideNextVirtualLayer);
		LayerStandard outsideNextVirtualLayer = getOutsideNextVirtualLayer();
		if (outsideNextVirtualLayer != null && outsideNextVirtualLayer.size() > 0) nextLayers.add(outsideNextVirtualLayer);
		return nextLayers;
	}


	@Override
	public boolean setNextLayer(LayerStandard nextLayer) {
		return setNextLayer(nextLayer, null, false);
	}


	/**
	 * This interface represent a method to set next layer.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public static interface NextLayerSetter extends Serializable, Cloneable {
		
		/**
		 * Setting next layer (1st).
		 * @param thisLayer this layer.
		 * @param nextLayer next layer.
		 */
		void setNextLayer(LayerStandard thisLayer, LayerStandard nextLayer);
		
		/**
		 * Setting next layer (2nd). This method is ignored when setting network from left to right.
		 * Indeed this method is the same to method {@link #setNextLayer(LayerStandard, LayerStandard)}.
		 * @param nextLayer next layer.
		 * @param nextNextLayer next layer of the next layer.
		 */
		void setNextNextLayer(LayerStandard nextLayer, LayerStandard nextNextLayer);
		
	}

	
	/**
	 * Setting next layer.
	 * @param nextLayer next layer. It can be null.
	 * @param setter specified setter. It can be null.
	 * @param injective if this parameter is true, there is only one connection between two neurons.
	 * @return true if setting is successful.
	 */
	private boolean setNextLayer(LayerStandard nextLayer, NextLayerSetter setter, boolean injective) {
		if (nextLayer == this.nextLayer) return false;
		if (nextLayer != null) {
			if (nextLayer == getRiboutLayer()) return false;
			if (nextLayer.getRibinLayer() == this) return false;
		}
		
		clearNextNeurons(this);
		LayerStandard oldNextLayer = this.nextLayer;
		if (oldNextLayer != null) ((LayerStandardImpl)oldNextLayer).prevLayer = null;
		this.nextLayer = nextLayer;
		if (nextLayer == null) return true;
		
		LayerStandard prevLayerOfNextLayer = ((LayerStandardImpl)nextLayer).prevLayer;
		if (prevLayerOfNextLayer != null) {
			clearNextNeurons(prevLayerOfNextLayer);
			((LayerStandardImpl)prevLayerOfNextLayer).nextLayer = null;
		}
		
		((LayerStandardImpl)nextLayer).prevLayer = this;
		if (setter != null) {
			setter.setNextLayer(this, nextLayer);
		}
		else {
			if (injective) {
				int n = Math.min(size(), nextLayer.size());
				for (int i = 0; i < n; i++) {
					get(i).setNextNeuron(nextLayer.get(i), newWeight());
				}
			}
			else {
				for (int i = 0; i < size(); i++) {
					NeuronStandard neuron = get(i);
					for (int j = 0; j < nextLayer.size(); j++) {
						neuron.setNextNeuron(nextLayer.get(j), newWeight());
					}
				}
			}
		}
		
		return true;
	}

	
	/**
	 * Setting next layer.
	 * @param nextLayer next layer. It can be null.
	 * @param setter specified setter. It can be null.
	 * @return true if setting is successful.
	 */
	protected boolean setNextLayer(LayerStandard nextLayer, NextLayerSetter setter) {
		return setNextLayer(nextLayer, setter, false);
	}
	
	
	/**
	 * Setting next layer.
	 * @param nextLayer next layer. It can be null.
	 * @param injective if this parameter is true, there is only one connection between two neurons.
	 * @return true if setting is successful.
	 */
	protected boolean setNextLayer(LayerStandard nextLayer, boolean injective) {
		return setNextLayer(nextLayer, null, injective);
	}
	
	
	/**
	 * Replacing next layer.
	 * @param nextLayer next layer. It can be null.
	 * @return true if setting is successful.
	 */
	protected boolean replaceNextLayer(LayerStandard nextLayer) {
		return replaceNextLayer(nextLayer, null, false);
	}


	/**
	 * Replacing next layer.
	 * @param nextLayer next layer. It can be null.
	 * @param setter specified setter. It can be null.
	 * @param injective if this parameter is true, there is only one connection between two neurons.
	 * @return true if setting is successful.
	 */
	private boolean replaceNextLayer(LayerStandard nextLayer, NextLayerSetter setter, boolean injective) {
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
			((LayerStandardImpl)oldNextLayer).prevLayer = null;
		}
		this.nextLayer = nextLayer;
		if (nextLayer == null) return true;

		LayerStandard prevLayerOfNextLayer = ((LayerStandardImpl)nextLayer).prevLayer;
		if (prevLayerOfNextLayer != null) {
			clearNextNeurons(prevLayerOfNextLayer);
			((LayerStandardImpl)prevLayerOfNextLayer).nextLayer = null;
		}
		clearNextNeurons(nextLayer);
		
		((LayerStandardImpl)nextLayer).prevLayer = this;
		if (setter != null) {
			setter.setNextLayer(this, nextLayer);
		}
		else {
			if (injective) {
				int n = Math.min(size(), nextLayer.size());
				for (int i = 0; i < n; i++) {
					get(i).setNextNeuron(nextLayer.get(i), newWeight());
				}
			}
			else {
				for (int i = 0; i < size(); i++) {
					NeuronStandard neuron = get(i);
					for (int j = 0; j < nextLayer.size(); j++) {
						neuron.setNextNeuron(nextLayer.get(j), newWeight());
					}
				}
			}
		}
		
		if (oldNextNextLayer == null) return true;
		((LayerStandardImpl)oldNextNextLayer).prevLayer = nextLayer;
		((LayerStandardImpl)nextLayer).nextLayer = oldNextNextLayer;
		if (setter != null) {
			setter.setNextNextLayer(nextLayer, oldNextNextLayer);
		}
		else {
			if (injective) {
				int n = Math.min(nextLayer.size(), oldNextNextLayer.size());
				for (int i = 0; i < n; i++) {
					nextLayer.get(i).setNextNeuron(oldNextNextLayer.get(i), newWeight());
				}
			}
			else {
				for (int i = 0; i < nextLayer.size(); i++) {
					NeuronStandard neuron = nextLayer.get(i);
					for (int j = 0; j < oldNextNextLayer.size(); j++) {
						neuron.setNextNeuron(oldNextNextLayer.get(j), newWeight());
					}
				}
			}
		}
		
		return true;
	}

	
	/**
	 * Replacing next layer.
	 * @param nextLayer next layer. It can be null.
	 * @param setter specified setter. It can be null.
	 * @return true if setting is successful.
	 */
	protected boolean replaceNextLayer(LayerStandard nextLayer, NextLayerSetter setter) {
		return replaceNextLayer(nextLayer, setter, false);
	}
	
	
	/**
	 * Replacing next layer.
	 * @param nextLayer next layer. It can be null.
	 * @param injective if this parameter is true, there is only one connection between two neurons.
	 * @return true if setting is successful.
	 */
	protected boolean replaceNextLayer(LayerStandard nextLayer, boolean injective) {
		return replaceNextLayer(nextLayer, null, injective);
	}

	
	/**
	 * Removing this layer from the list.
	 * @return this layer from the list.
	 */
	protected boolean cutOffLayer() {
		return cutOffLayer(false);
	}


	/**
	 * Removing this layer from the list.
	 * @param injective if this parameter is true, there is only one connection between two neurons.
	 * @return this layer from the list.
	 */
	protected boolean cutOffLayer(boolean injective) {
		LayerStandardAbstract prevLayer = (LayerStandardAbstract)getPrevLayer();
		if (prevLayer != null) prevLayer.setNextLayer(null, injective);
		LayerStandardAbstract nextLayer = (LayerStandardAbstract)getNextLayer();
		if (nextLayer != null) nextLayer.setPrevLayer(null, injective);
		
		return prevLayer != null && nextLayer != null ? prevLayer.setNextLayer(nextLayer, injective) : true;
	}
	
	
	/**
	 * Clearing next neurons of specified layer.
	 * @param layer specified layer.
	 */
	public static void clearNextNeurons(LayerStandard layer) {
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
		return setRibinLayer(ribinLayer, false);
	}

	
//	/**
//	 * Setting input rib layer. This method needs to be checked more carefully.
//	 * @param ribinLayer input rib layer. It can be null.
//	 * @param injective if this parameter is true, there is only one connection between two neurons.
//	 * @return true if setting is successful.
//	 */
//	protected boolean setRibinLayer(LayerStandard ribinLayer, boolean injective) {
//		if (this.ribinLayer == ribinLayer) return false;
//		if (ribinLayer != null) {
//			//Rib-in layer has only one next layer which is this layer, which is a strict condition.
//			//Therefore setting rib-in layer is stricter setting rib-out layer.
//			if (ribinLayer.getNextLayer() != null) return false;
//			
//			if (ribinLayer == getPrevLayer()) return false;
//		}
//		
//		this.ribinLayer = ribinLayer;
//		if (ribinLayer == null) return true;
//			
//		clearNextNeurons(ribinLayer);
//		LayerStandard oldNextLayer = ribinLayer.getNextLayer();
//		if (oldNextLayer != null) clearNextNeurons(oldNextLayer); //This code line cutting off network to assert the strict condition that rib-in layer has only one next layer.
//
//		((LayerStandardImpl)ribinLayer).nextLayer = this;
//		if (injective) {
//			int n = Math.min(ribinLayer.size(), size());
//			for (int i = 0; i < n; i++) {
//				ribinLayer.get(i).setNextNeuron(get(i), newWeight());
//			}
//		}
//		else {
//			for (int i = 0; i < ribinLayer.size(); i++) {
//				NeuronStandard ribbinNeuron = ribinLayer.get(i);
//				for (int j = 0; j < size(); j++) {
//					ribbinNeuron.setNextNeuron(get(j), newWeight());
//				}
//			}
//		}
//		
//		return true;
//	}

	
	/**
	 * Setting input rib layer. This method needs to be checked more carefully.
	 * @param ribinLayer input rib layer. It can be null.
	 * @param injective if this parameter is true, there is only one connection between two neurons.
	 * @return true if setting is successful.
	 */
	protected boolean setRibinLayer(LayerStandard ribinLayer, boolean injective) {
		if (this.ribinLayer == ribinLayer) return false;
		if (ribinLayer != null) {
			//Rib-in layer has only one next layer which is this layer, which is a strict condition.
			//Therefore setting rib-in layer is stricter setting rib-out layer.
			if (ribinLayer.getNextLayer() != null) return false;
			
			if (ribinLayer == getPrevLayer()) return false;
		}
		
		this.ribinLayer = ribinLayer;
		if (ribinLayer == null) return true;
			
		clearNextNeurons(ribinLayer);
		LayerStandard oldNextLayer = ribinLayer.getNextLayer();
		if (oldNextLayer != null) ((LayerStandardImpl)oldNextLayer).ribinLayer = null; //This code is not necessary because of the strict condition that rib-in layer has only one next layer.

		((LayerStandardImpl)ribinLayer).nextLayer = this;
		if (injective) {
			int n = Math.min(ribinLayer.size(), size());
			for (int i = 0; i < n; i++) {
				ribinLayer.get(i).setNextNeuron(get(i), newWeight());
			}
		}
		else {
			for (int i = 0; i < ribinLayer.size(); i++) {
				NeuronStandard ribbinNeuron = ribinLayer.get(i);
				for (int j = 0; j < size(); j++) {
					ribbinNeuron.setNextNeuron(get(j), newWeight());
				}
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
		return setRiboutLayer(riboutLayer, null, false);
	}


	/**
	 * Setting output rib layer.
	 * @param riboutLayer output rib layer. It can be null.
	 * @param setter specified setter.
	 * @param injective if this parameter is true, there is only one connection between two neurons.
	 * @return true if setting is successful.
	 */
	private boolean setRiboutLayer(LayerStandard riboutLayer, NextLayerSetter setter, boolean injective) {
		if (this.riboutLayer == riboutLayer) return false;
		if (riboutLayer != null) {
//			//This condition is removed because it is too strict.
//			//Therefore, rib-out layer can have explicit previous layer besides this layer as its implicit previous layer.
//			if (riboutLayer.getPrevLayer() != null) return false;
			
			if (riboutLayer == getNextLayer()) return false;
		}
		
		LayerStandard oldRiboutLayer = this.riboutLayer;
		this.riboutLayer = riboutLayer;
		for (NeuronStandard neuron : neurons) ((NeuronStandardImpl)neuron).riboutNeurons.clear();
		
		if (oldRiboutLayer != null) ((LayerStandardImpl)oldRiboutLayer).prevLayerImplicit = null;
		if (riboutLayer == null) return true;
		
		if (setter != null) {
			setter.setNextLayer(this, riboutLayer);
		}
		else {
			if (injective) {
				int n = Math.min(size(), riboutLayer.size());
				for (int i = 0; i < n; i++) {
					WeightedNeuron wn = new WeightedNeuron(riboutLayer.get(i), newWeight());
					((NeuronStandardImpl)get(i)).riboutNeurons.add(wn);
				}
			}
			else {
				for (NeuronStandard neuron : neurons) {
					for (int i = 0; i < riboutLayer.size(); i++) {
						WeightedNeuron wn = new WeightedNeuron(riboutLayer.get(i), newWeight());
						((NeuronStandardImpl)neuron).riboutNeurons.add(wn);
					}
				}
			}
		}
		
		((LayerStandardImpl)riboutLayer).prevLayerImplicit = this;
		
		return true;
	}

	
	/**
	 * Setting output rib layer.
	 * @param riboutLayer output rib layer. It can be null.
	 * @param setter specified setter.
	 * @return true if setting is successful.
	 */
	protected boolean setRiboutLayer(LayerStandard riboutLayer, NextLayerSetter setter) {
		return setRiboutLayer(riboutLayer, setter, false);
	}
	
	
	/**
	 * Setting output rib layer.
	 * @param riboutLayer output rib layer. It can be null.
	 * @param injective if this parameter is true, there is only one connection between two neurons.
	 * @return true if setting is successful.
	 */
	protected boolean setRiboutLayer(LayerStandard riboutLayer, boolean injective) {
		return setRiboutLayer(riboutLayer, null, injective);
	}
	
	
	@Override
	public Set<WeightedNeuron> getInsidePrevNeurons() {
		Set<WeightedNeuron> wns = Util.newSet(0);
		for (NeuronStandard neuron : neurons) wns.addAll(neuron.getInsidePrevNeurons());
		return wns;
	}

	
	@Override
	public LayerStandard getInsidePrevVirtualLayer() {
		Set<WeightedNeuron> wns = getInsidePrevNeurons();
		LayerStandardImpl insidePrevVirtualLayer = getNetwork() != null ? (LayerStandardImpl)getNetwork().newLayer() : new LayerStandardImpl(neuronChannel, activateRef, idRef);
		for (WeightedNeuron wn : wns) insidePrevVirtualLayer.add(wn.neuron);
		return insidePrevVirtualLayer;
	}

	
	@Override
	public Set<WeightedNeuron> getInsideNextNeurons() {
		Set<WeightedNeuron> wns = Util.newSet(0);
		for (NeuronStandard neuron : neurons) wns.addAll(neuron.getInsideNextNeurons());
		return wns;
	}


	@Override
	public LayerStandard getInsideNextVirtualLayer() {
		Set<WeightedNeuron> wns = getInsideNextNeurons();
		LayerStandardImpl insideNextVirtualLayer = getNetwork() != null ? (LayerStandardImpl)getNetwork().newLayer() : new LayerStandardImpl(neuronChannel, activateRef, idRef);
		for (WeightedNeuron wn : wns) insideNextVirtualLayer.add(wn.neuron);
		return insideNextVirtualLayer;
	}

	
	@Override
	public Set<WeightedNeuron> getOutsidePrevNeurons() {
		Set<WeightedNeuron> wns = Util.newSet(0);
		for (NeuronStandard neuron : neurons) wns.addAll(neuron.getOutsidePrevNeurons());
		return wns;
	}

	
	@Override
	public LayerStandard getOutsidePrevVirtualLayer() {
		Set<WeightedNeuron> wns = getOutsidePrevNeurons();
		LayerStandardImpl outsidePrevVirtualLayer = getNetwork() != null ? (LayerStandardImpl)getNetwork().newLayer() : new LayerStandardImpl(neuronChannel, activateRef, idRef);
		for (WeightedNeuron wn : wns) outsidePrevVirtualLayer.add(wn.neuron);
		return outsidePrevVirtualLayer;
	}

	
	@Override
	public boolean addOutsidePrevVirtualLayer(LayerStandard outsidePrevVirtualLayer) {
		if (outsidePrevVirtualLayer == null) return false;
		for (NeuronStandard neuron : neurons) {
			for (int i = 0; i < outsidePrevVirtualLayer.size(); i++) {
				((NeuronStandardImpl)neuron).addOutsidePrevNeuron(outsidePrevVirtualLayer.get(i), newWeight());
			}
		}
		return true;
	}


	@Override
	public void removeOutsidePrevVirtualLayer(LayerStandard outsidePrevVirtualLayer) {
		if (outsidePrevVirtualLayer == null) return;
		for (NeuronStandard neuron : neurons) {
			for (int i = 0; i < outsidePrevVirtualLayer.size(); i++) {
				((NeuronStandardImpl)neuron).removeOutsidePrevNeuron(outsidePrevVirtualLayer.get(i));
			}
		}
	}


	@Override
	public Set<WeightedNeuron> getOutsideNextNeurons() {
		Set<WeightedNeuron> wns = Util.newSet(0);
		for (NeuronStandard neuron : neurons) wns.addAll(neuron.getOutsideNextNeurons());
		return wns;
	}


	@Override
	public LayerStandard getOutsideNextVirtualLayer() {
		Set<WeightedNeuron> wns = getOutsideNextNeurons();
		LayerStandardImpl outsideNextVirtualLayer = getNetwork() != null ? (LayerStandardImpl)getNetwork().newLayer() : new LayerStandardImpl(neuronChannel, activateRef, idRef);
		for (WeightedNeuron wn : wns) outsideNextVirtualLayer.add(wn.neuron);
		return outsideNextVirtualLayer;
	}


	@Override
	public boolean addOutsideNextVirtualLayer(LayerStandard outsideNextVirtualLayer) {
		if (outsideNextVirtualLayer == null) return false;
		for (NeuronStandard neuron : neurons) {
			for (int i = 0; i < outsideNextVirtualLayer.size(); i++) {
				((NeuronStandardImpl)neuron).addOutsideNextNeuron(outsideNextVirtualLayer.get(i), newWeight());
			}
		}
		return true;
	}


	@Override
	public void removeOutsideNextVirtualLayer(LayerStandard outsideNextVirtualLayer) {
		if (outsideNextVirtualLayer == null) return;
		for (NeuronStandard neuron : neurons) {
			for (int i = 0; i < outsideNextVirtualLayer.size(); i++) {
				((NeuronStandardImpl)neuron).removeOutsideNextNeuron(outsideNextVirtualLayer.get(i));
			}
		}
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
	public void setInput(NeuronValue...input) {
		if (neurons.size() == 0 || input == null) return;
		int n = Math.min(neurons.size(), input.length);
		for (int j = 0; j < n; j++) {
			if (input[j] != null) neurons.get(j).setInput(input[j]);
		}
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


	@Override
	public void setOutput(NeuronValue...output) {
		if (neurons.size() == 0 || output == null) return;
		int n = Math.min(neurons.size(), output.length);
		for (int j = 0; j < n; j++) {
			if (output[j] != null) neurons.get(j).setOutput(output[j]);
		}
	}

	
	@Override
	public NeuronValue[] evaluate(NeuronValue[] input) {
		if (!hasSomePrevLayers()) {
			input = NeuronValue.adjustArray(input, size(), this);
			for (int j = 0; j < size(); j++) {
				NeuronStandard neuron = get(j);
				neuron.setInput(input[j]);
				neuron.setOutput(input[j]);
			}
		}
		else {
			for (NeuronStandard neuron : neurons) neuron.evaluate();
			postEvaluate();
		}
		return getOutput();
	}

	
	@Override
	public NeuronValue[] evaluate() {
		if (!hasSomePrevLayers()) return getOutput();
		for (NeuronStandard neuron : neurons) neuron.evaluate();
		postEvaluate();
		return getOutput();
	}

	
	/**
	 * Post evaluation.
	 */
	protected void postEvaluate() {
		if (activateRef == null || !(activateRef instanceof FunctionDelay)) return;
		for (NeuronStandard neuron : neurons) {
			if (neuron.getActivateRef() != activateRef) return;
		}
		postEvaluate(this, activateRef);
	}
	
	
	/**
	 * Post evaluation.
	 * @param layer specified layer.
	 * @param activateRef specified activation reference.
	 */
	protected static void postEvaluate(LayerStandard layer, Function activateRef) {
		if (layer == null || activateRef == null || layer.size() == 0) return;
		NeuronValue[] outputs = new NeuronValue[layer.size()];
		NeuronValue sum = layer.get(0).getOutput().zero();
		for (int i = 0; i < outputs.length; i++) {
			NeuronStandard neuron = layer.get(i);
			NeuronValue output = neuron.getOutput();
			neuron.setInput(output);
			
			outputs[i] = output.evaluate(activateRef);
			sum = sum.add(outputs[i]);
		}
		double v = sum.mean();
		if (!Double.isFinite(v)) {
			for (int i = 0; i < outputs.length; i++) outputs[i] = new NeuronValue1(1.0/(double)outputs.length);
			System.out.println("LayerStandardAbstract.postEvaluate(LayerStandard, Function) produces non-finite numbers");
		}
		else if (v > 1.0 + 10*Network.LEARN_TERMINATED_THRESHOLD_DEFAULT) {
			for (int i = 0; i < outputs.length; i++) outputs[i] = outputs[i].divide(sum);
			System.out.println("LayerStandardAbstract.postEvaluate(LayerStandard, Function) produces larger-than-1 numbers");
		}
		
		for (int i = 0; i < outputs.length; i++) layer.get(i).setOutput(outputs[i]);
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


	@Override
	public String toText() {
		try {
			String text = toText(this, null);
			text = text.replaceAll("l##", "");
			return text;
		}
		catch (Throwable e) {}
		
		return super.toString();
	}

	
}
