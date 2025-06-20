/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.stack;

import java.util.List;

import net.ea.ann.conv.Content;
import net.ea.ann.conv.ConvLayer;
import net.ea.ann.conv.ConvLayerSingle;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.core.Id;
import net.ea.ann.core.LayerAbstract;
import net.ea.ann.core.Neuron;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.Weight;
import net.ea.ann.raster.NeuronRaster;
import net.ea.ann.raster.Raster;

/**
 * This class represents an abstract class of stack element layer.
 * Element layer in this current version does not support rib-in, rib-out, inside, and outside element layers.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class ElementLayerAbstract extends LayerAbstract implements ElementLayer {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Neuron channel or depth.
	 */
	protected int neuronChannel = 1;

	
	/**
	 * Current stack.
	 */
	protected Stack stack = null;
	
	
	/**
	 * Next layers.
	 */
	protected List<WeightedElementLayer> nextLayers = Util.newList(0);

	
	/**
	 * Content of this layer.
	 */
	protected Content content = null;
	
	
	/**
	 * Content bias associated with weights.
	 */
	protected NeuronValue bias = null;
	
	
	/**
	 * Activation function.
	 */
	protected Function activateRef = null;
	
	
	/**
	 * Constructor with neuron channel, stack, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param stack specified stack.
	 * @param content layer content.
	 * @param activateRef activation reference.
	 * @param idRef ID reference.
	 */
	protected ElementLayerAbstract(int neuronChannel, Stack stack, Content content, Function activateRef, Id idRef) {
		super(idRef);
		if (neuronChannel < 1)
			this.neuronChannel = neuronChannel = 1;
		else
			this.neuronChannel = neuronChannel;
		this.activateRef = activateRef == null ? (activateRef = Raster.toConvActivationRef(this.neuronChannel, true)) : activateRef;
		
		this.stack = stack;
		this.content = content;
		this.bias = stack != null ? stack.newBias() : content.newNeuronValue().zero();
	}

	
	/**
	 * Constructor with neuron channel and stack.
	 * @param neuronChannel neuron channel or depth.
	 * @param stack specified stack.
	 * @param content layer content.
	 * @param activateRef activation reference.
	 */
	protected ElementLayerAbstract(int neuronChannel, Stack stack, Content content, Function activateRef) {
		this(neuronChannel, stack, content, activateRef, null);
	}

	
	@Override
	public List<WeightedElementLayer> getPrevLayers() {
		List<WeightedElementLayer> sources = Util.newList(0);
		if (stack == null) return sources;
		
		Stack prevStack = stack.getPrevStack();
		if (prevStack == null) return sources;
		
		for (int i = 0; i < prevStack.size(); i++) {
			ElementLayer prevLayer = prevStack.get(i);
			WeightedElementLayer found = prevLayer.findNextLayer(this);
			if (found != null) {
				WeightedElementLayer wl = new WeightedElementLayer(prevLayer, found.weight);
				sources.add(wl);
			}
		}
		
		return sources;
	}

	
	/**
	 * Getting previous content.
	 * @return previous content.
	 */
	protected Content getPrevContent() {
		return content != null ? (Content)content.getPrevLayer() : null;
	}
	
	

	@Override
	public List<WeightedElementLayer> getNextLayers() {
		List<WeightedElementLayer> actualNextLayers = Util.newList(0);
		if (this.content == null) return actualNextLayers;
		for (WeightedElementLayer nextLayer : nextLayers) {
			Content nextContent = nextLayer.layer != null ? nextLayer.layer.getContent() : null;
			if (nextContent != null && this.content.indexOfNextContent(nextContent) >= 0)
				actualNextLayers.add(nextLayer);
		}
		
		return actualNextLayers;
	}

	
	@Override
	public List<WeightedElementLayer> getNextLayers(Stack nextStack) {
		if (nextStack == null || stack == null || stack.getNextStack() == nextStack)
			return getNextLayers();
		else
			return Util.newList(0);
	}

	
	/**
	 * Getting next content.
	 * @return next content.
	 */
	protected Content getNextContent() {
		return content != null ? (Content)content.getNextLayer() : null;
	}
	
	
	@Override
	public boolean setNextLayer(ElementLayer layer, Weight weight, Filter filter) {
		Stack nextStack = stack != null ? stack.getNextStack() : null;
		if (nextStack == null || layer == null || weight == null)
			return false;
		if (nextStack.indexOf(layer) < 0) return false;
		
		WeightedElementLayer wl = findNextLayer(layer);
		if (wl == null) {
			wl = new WeightedElementLayer(layer, weight, filter);
			this.getContent().addNextContent(layer.getContent());
			nextLayers.add(wl);
		}
		else {
			wl.weight = weight;
		}
		
		return true;
	}


	@Override
	public boolean removeNextLayer(ElementLayer layer) {
		if (layer == null) return false;
		for (int i = 0; i < nextLayers.size(); i++) {
			if (nextLayers.get(i).layer == layer) {
				this.getContent().removeNextContent(nextLayers.get(i).layer.getContent());
				nextLayers.remove(i);
				return true;
			}
		}

		return false;
	}

	
	@Override
	public void clearNextLayers() {
		List<WeightedElementLayer> wls = Util.newList(this.nextLayers.size());
		wls.addAll(this.nextLayers);
		
		for (WeightedElementLayer wl : wls) {
			removeNextLayer(wl.layer);
		}
		
		this.nextLayers.clear();
	}

	
	@Override
	public WeightedElementLayer findNextLayer(ElementLayer layer) {
		for (int i = 0; i < nextLayers.size(); i++) {
			WeightedElementLayer wl = nextLayers.get(i);
			if (wl.layer == layer) return wl;
		}
		
		return null;
	}


	@Override
	public Stack getStack() {
		return stack;
	}


	@Override
	public Content getContent() {
		return content;
	}


	@Override
	public NeuronValue[] setContent(NeuronValue[] data) {
		if (content != null)
			return content.setData(data);
		else
			return null;
	}
	
	
	@Override
	public boolean isPadZeroFilter() {
		return content != null ? content.isPadZeroFilter() : false;
	}


	@Override
	public void setPadZeroFilter(boolean isPadZeroFilter) {
		if (content != null) content.setPadZeroFilter(isPadZeroFilter);
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
	public Function getActivateRef() {
		return activateRef;
	}
	
	
	@Override
	public ConvLayer forward() {
		if (content == null) return null;
		List<WeightedElementLayer> nextLayers = getNextLayers();
		boolean forwarded = false;
		ElementLayer forwardedLayer = null;
		for (WeightedElementLayer nextLayer : nextLayers) {
			if (nextLayer.filter == null) continue;
			content.forward(nextLayer.layer.getContent(), nextLayer.filter);
			forwarded = true;
			forwardedLayer = nextLayer.layer;
		}
		
		if (forwarded)
			return forwardedLayer.getStack();
		else
			return content.forward();
	}


	@Override
	public ConvLayerSingle evaluate() {
		if (this.content == null) return null;
		List<WeightedElementLayer> sources = getPrevLayers();
		if (sources.size() == 0) return null;
		
		List<Object[]> targets = Util.newList(0);
		for (WeightedElementLayer source : sources) {
			Content sourceContent = source.layer.getContent();
			if (sourceContent == null) continue;
			
			NeuronRaster raster = sourceContent.forward(this.content, source.filter);
			if (raster == null) continue;
			targets.add(new Object[] {raster.getNeurons(), source.weight});
		}
		if (targets.size() == 0) return null;
		
		NeuronValue[] sum = new NeuronValue[this.content.length()];
		for (int i = 0; i < sum.length; i++) sum[i] = bias;
		
		for (Object[] target : targets) {
			Neuron[] neurons = (Neuron[])target[0];
			double weight = (double)target[1]; 
			int n = Math.min(neurons.length, sum.length);
			for (int i = 0; i < n; i++) {
				sum[i] = sum[i].add(neurons[i].getValue().multiply(weight));
			}
		}
		
		for (int i = 0; i < sum.length; i++) {
			if (activateRef != null) sum[i] = activateRef.evaluate(sum[i]);
			this.content.set(i, sum[i]);
		}
		return this.content;
	}

	
}
