/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.stack;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

import net.ea.ann.conv.Content;
import net.ea.ann.conv.ConvLayer;
import net.ea.ann.conv.ConvNeuron;
import net.ea.ann.conv.ConvNeuronImpl;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.core.Id;
import net.ea.ann.core.LayerAbstract;
import net.ea.ann.core.Util;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.Weight;

/**
 * This class is an abstract implementation of stack of convolutional layers. It is a convolutional layer too.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class StackAbstract extends LayerAbstract implements Stack {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Neuron channel or depth.
	 */
	protected int neuronChannel = 1;
	
	
	/**
	 * List of stack elements.
	 */
	protected List<ElementLayer> layers = Util.newList(0);
	
	
	/**
	 * Previous stack.
	 */
	protected Stack prevStack = null;
	
	
	/**
	 * Next stack.
	 */
	protected Stack nextStack = null;

	
	/**
	 * Flag to indicate whether to pad zero when filering.
	 */
	protected boolean isPadZeroFilter = false;

	
	/**
	 * Constructor with neuron channel and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param idRef identifier reference.
	 */
	protected StackAbstract(int neuronChannel, Id idRef) {
		super(idRef);
		this.neuronChannel = neuronChannel;
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	protected StackAbstract(int neuronChannel) {
		this(neuronChannel, null);
	}

	
	@Override
	public Weight newWeight() {
		return new Weight(newNeuronValue().newWeightValue().zeroW());
	}

	
	@Override
	public NeuronValue newBias() {
		return newNeuronValue();
	}


	@Override
	public ConvNeuron newNeuron() {
		return new ConvNeuronImpl(this);
	}

	
	@Override
	public int getNeuronChannel() {
		return neuronChannel;
	}


	@Override
	public int size() {
		return layers.size();
	}

	
	@Override
	public ElementLayer get(int index) {
		return layers.get(index);
	}

	
	@Override
	public boolean add(ElementLayer layer) {
		return layers.add(layer);
	}

	
	@Override
	public ElementLayer remove(int index) {
		ElementLayer layer = layers.get(index);
		layer.clearNextLayers();
		
		return layers.remove(index);
	}

	
	@Override
	public void clear() {
		while (layers.size() > 0) {
			remove(0);
		}
	}

	
	@Override
	public int indexOf(ElementLayer layer) {
		return layers.indexOf(layer);
	}

	
	/**
	 * Checking whether padding zero when filtering.
	 * @return whether padding zero when filtering.
	 */
	protected boolean isPaddZeroFilter() {
		return isPadZeroFilter;
	}

	
	/**
	 * Setting whether to pad zero when filtering.
	 * @param isPadZeroFilter flag to indicate whether to pad zero when filtering.
	 */
	protected void setPadZeroFilter(boolean isPadZeroFilter) {
		this.isPadZeroFilter = isPadZeroFilter;
	}

	
	@Override
	public Stack getPrevStack() {
		return prevStack;
	}

	
	@Override
	public Set<Stack> getAllPrevStacks() {
		Set<Stack> prevStacks = Util.newSet(0);
		if (prevStack != null) prevStacks.add(prevStack);
		return prevStacks;
	}


	@Override
	public ConvLayer getPrevLayer() {
		return getPrevStack();
	}


	@Override
	public Stack getNextStack() {
		return nextStack;
	}

	
	@Override
	public ConvLayer getNextLayer() {
		return getNextStack();
	}


	@Override
	public boolean setNextStack(Stack nextStack) {
		return setNextStack(nextStack, null, false);
	}

	
	@Override
	public boolean setNextStack(Stack nextStack, boolean injective, Filter...filters) {
		return setNextStack(nextStack, null, injective, filters);
	}

		
	/**
	 * This interface represent a method to set next stack.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public static interface NextStackSetter {
		
		/**
		 * Setting next stack (1st).
		 * @param thisStack this stack.
		 * @param nextStack next stack.
		 */
		void setNextStack(Stack thisStack, Stack nextStack);
		
		/**
		 * Setting next stack (2nd). This method is ignored when setting network from left to right.
		 * @param nextStack next stack.
		 * @param nextNextStack next stack of the next stack.
		 */
		void setNextNextStack(Stack nextStack, Stack nextNextStack);

	}
	
	
	/**
	 * Setting next stack.
	 * @param nextStack next stack.
	 * @param setter specified setter.
	 * @param injective if this parameter is true, there is only one connection between two layers.
	 * @param filters array of filters.
	 * @return true if setting is successful.
	 */
	private boolean setNextStack(Stack nextStack, NextStackSetter setter, boolean injective, Filter...filters) {
		if (nextStack == this.nextStack) return false;

		clearNextLayers(this);
		Stack oldNextStack = this.nextStack;
		if (oldNextStack != null) ((StackAbstract)oldNextStack).prevStack = null;
		this.nextStack = nextStack;
		if (nextStack == null) return true;

		Stack prevStactOfNextStack = ((StackAbstract)nextStack).prevStack;
		if (prevStactOfNextStack != null) {
			clearNextLayers(prevStactOfNextStack);
			((StackAbstract)prevStactOfNextStack).nextStack = null;
		}

		((StackAbstract)nextStack).prevStack = this;
		int k = 0;
		if (setter != null) {
			setter.setNextStack(this, nextStack);
		}
		else {
			if (injective) {
				int n = Math.min(size(), nextStack.size());
				for (int i = 0; i < n; i++) {
					Filter filter = (filters != null && k < filters.length) ? filters[k] : null;
					get(i).setNextLayer(nextStack.get(i), newWeight(), filter);
					k++;
				}
			}
			else {
				for (int i = 0; i < size(); i++) {
					ElementLayer layer = get(i);
					for (int j = 0; j < nextStack.size(); j++) {
						Filter filter = (filters != null && k < filters.length) ? filters[k] : null;
						layer.setNextLayer(nextStack.get(j), newWeight(), filter);
						k++;
					}
				}
			}
		}
		
		return true;
	}

	
	/**
	 * Replacing next stack.
	 * @param nextStack next stack.
	 * @return true if setting is successful.
	 */
	protected boolean replaceNextStack(Stack nextStack) {
		return replaceNextStack(nextStack, null, false);
	}

	
	/**
	 * Replacing next stack.
	 * @param nextStack next stack.
	 * @param injective if this parameter is true, there is only one connection between two layers.
	 * @param filters array of filters.
	 * @return true if setting is successful.
	 */
	protected boolean replaceNextStack(Stack nextStack, boolean injective, Filter...filters) {
		return replaceNextStack(nextStack, null, injective, filters);
	}
	
	
	/**
	 * Replacing next stack.
	 * @param nextStack next stack.
	 * @param setter specified setter.
	 * @param injective if this parameter is true, there is only one connection between two layers.
	 * @param filters array of filters.
	 * @return true if setting is successful.
	 */
	private boolean replaceNextStack(Stack nextStack, NextStackSetter setter, boolean injective, Filter...filters) {
		if (nextStack == this.nextStack) return false;

		clearNextLayers(this);
		Stack oldNextStack = this.nextStack;
		Stack oldNextNextStack = null;
		if (oldNextStack != null) {
			oldNextNextStack = oldNextStack.getNextStack();
			clearNextLayers(oldNextStack);
			((StackAbstract)oldNextStack).prevStack = null;
		}
		this.nextStack = nextStack;
		if (nextStack == null) return true;

		Stack prevStactOfNextStack = ((StackAbstract)nextStack).prevStack;
		if (prevStactOfNextStack != null) {
			clearNextLayers(prevStactOfNextStack);
			((StackAbstract)prevStactOfNextStack).nextStack = null;
		}

		clearNextLayers(nextStack);
		((StackAbstract)nextStack).prevStack = this;
		int k = 0;
		if (setter != null) {
			setter.setNextStack(this, nextStack);
		}
		else {
			if (injective) {
				int n = Math.min(size(), nextStack.size());
				for (int i = 0; i < n; i++) {
					Filter filter = (filters != null && k < filters.length) ? filters[k] : null;
					get(i).setNextLayer(nextStack.get(i), newWeight(), filter);
					k++;
				}
			}
			else {
				for (int i = 0; i < size(); i++) {
					ElementLayer layer = get(i);
					for (int j = 0; j < nextStack.size(); j++) {
						Filter filter = (filters != null && k < filters.length) ? filters[k] : null;
						layer.setNextLayer(nextStack.get(j), newWeight(), filter);
						k++;
					}
				}
			}
		}
		
		if (oldNextNextStack == null) return true;
		((StackAbstract)oldNextNextStack).prevStack = nextStack;
		((StackAbstract)nextStack).nextStack = oldNextNextStack;
		if (setter != null) {
			setter.setNextNextStack(nextStack, oldNextNextStack);
		}
		else {
			if (injective) {
				int n = Math.min(nextStack.size(), oldNextNextStack.size());
				for (int i = 0; i < n; i++) {
					Filter filter = (filters != null && k < filters.length) ? filters[k] : null;
					nextStack.get(i).setNextLayer(oldNextNextStack.get(i), newWeight(), filter);
					k++;
				}
			}
			else {
				for (int i = 0; i < nextStack.size(); i++) {
					ElementLayer layer = nextStack.get(i);
					for (int j = 0; j < oldNextNextStack.size(); j++) {
						Filter filter = (filters != null && k < filters.length) ? filters[k] : null;
						layer.setNextLayer(oldNextNextStack.get(j), newWeight(), filter);
						k++;
					}
				}
			}
		}
		
		return true;
	}

	
	@Override
	public boolean setNextLayer(ConvLayer nextLayer) {
		if ((nextLayer != null) && (nextLayer instanceof Stack))
			return setNextStack((Stack)nextLayer);
		else
			return false;
	}


	/**
	 * Clearing next neurons of specified layer.
	 * @param layer specified layer.
	 */
	private static void clearNextLayers(Stack stack) {
		if (stack == null) return;
		for (int i = 0; i < stack.size(); i++) {
			stack.get(i).clearNextLayers();
		}
	}


	@Override
	public NeuronValue[] setContent(NeuronValue[]...datas) {
		if (datas == null || datas.length == 0) return null;
		
		if (datas.length == 1) {
			NeuronValue[] adjustedData = null;
			for (ElementLayer layer : layers) {
				adjustedData = layer.setContent(datas[0]);
			}
			
			return adjustedData;
		}
		else {
			NeuronValue[] adjustedData = null;
			for (int i = 0; i < layers.size(); i++) {
				ElementLayer layer = layers.get(i);
				if (i < datas.length)
					adjustedData = layer.setContent(datas[i]);
				else
					adjustedData = layer.setContent(datas[datas.length - 1]);
			}
			
			return adjustedData;
		}
		
	}
	
	
	@Override
	public ConvLayer forward() {
		ConvLayer result = null;
		for (int i = 0; i < size(); i++) result = get(i).forward();
		return result;
	}


	@Override
	public List<Content> evaluate() {
		List<Content> output = Util.newList(0); 
		for (ElementLayer layer : layers) {
			layer.evaluate();
			output.add(layer.getContent());
		}
		
		return output;
	}
	
	
	
	/**
	 * Create content with specified index.
	 * @param index specified index.
	 * @param stack specified stack.
	 * @return content.
	 */
	public static Content newContent(int index, Stack stack) {
		if (index < 0 || index >= stack.size()) return null;
		Content content = stack.get(index).getContent();
		return stack.newContent(content.getActivateRef(), content.getSize(), content.getFilter());
	}

	
	/**
	 * Create last content.
	 * @param stack specified stack.
	 * @return last content.
	 */
	public static Content newOutputContent(Stack stack) {
		return newContent(stack.size() - 1, stack);
	}
	
	
	/**
	 * Create an array of content.
	 * @param length array length.
	 * @param stack referred stack.
	 * @return array of contents.
	 */
	public static Content[] makeArray(int length, Stack stack) {
		Content[] array = new Content[length];
		for (int j = 0; j < length; j++) array[j] = newContent(j, stack);
		
		return array;
	}


	/**
	 * Adjusting array by length.
	 * @param array specified array.
	 * @param length specified length.
	 * @param stack referred stack.
	 * @return adjusted array.
	 */
	public static Content[] adjustArray(Content[] array, int length, Stack stack) {
		if (length <= 0) return array;
		
		if (array == null || array.length == 0)
			array = makeArray(length, stack);
		else if (array.length < length) {
			int originLength = array.length;
			array = Arrays.copyOfRange(array, 0, length);
			for (int j = originLength; j < length; j++) {
				if (array[j] == null) array[j] = newContent(j, stack);
			}
		}
		
		return array;
	}


}
