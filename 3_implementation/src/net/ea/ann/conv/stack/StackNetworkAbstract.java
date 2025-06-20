/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv.stack;

import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;

import net.ea.ann.conv.Content;
import net.ea.ann.conv.ContentImpl;
import net.ea.ann.conv.RecordExt;
import net.ea.ann.conv.filter.BiasFilter;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.conv.filter.FilterFactory;
import net.ea.ann.conv.filter.FilterFactoryImpl;
import net.ea.ann.conv.stack.bp.Backpropagator;
import net.ea.ann.conv.stack.bp.BackpropagatorAbstract;
import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.NetworkDoEvent;
import net.ea.ann.core.NetworkDoEvent.Type;
import net.ea.ann.core.NetworkDoEventImpl;
import net.ea.ann.core.NetworkInfoEvent;
import net.ea.ann.core.NetworkListener;
import net.ea.ann.core.NetworkStandardImpl;
import net.ea.ann.core.NormSupporter;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Image;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Size;

/**
 * This class is an abstract implementation of convolutional stack network.<br>
 * <br>
 * Output of the fully connected network (FCN) {@link #fullNetwork} is some feature/some class and output of the reserved fully connected network (RFCN) {@link #reversedFullNetwork} is (flat) unified content.
 * Note, the unified content is output of convolutional/deconvolutional network (main network) {@link #stacks} and the dimension of feature is always the same to the dimension of FCN and RFCN.<br>
 * <br>
 * If FCN is not null and RFCN is not null, FCN will transform the (flat) unified content to some feature
 * and RFCN will transform the feature back to (flat) unified content (exactly, the flat unified content which is input of FCN).<br>
 * <br>
 * If FCN is not null but RFCN is null, FCN will classify the (flat) unified content to some class/feature.<br>
 * <br>
 * If FCN is null but RFCN is not null, RFCN will transform some feature to (flat) input of the main network.
 * Therefore, the main network is often deconvolutional network whose output is finer than its input.<br>
 * <br>
 * If FCN is null and RFCN is null, the main network is normal convolutional network or normal deconvolutional network.
 * In this case, the output of the convolutional network or the input of the deconvolutional network can be considered as feature for some other applications.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class StackNetworkAbstract extends NetworkAbstract implements StackNetwork, NetworkListener {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Name of learning filter field.
	 */
	protected static final String LEARNING_FILTERS_FIELD = "conv_learning_filters";

	
	/**
	 * Name of learning filter field.
	 */
	protected static final boolean LEARNING_FILTERS_DEFAULT = false;

	
	/**
	 * List of layer stacks. This is the main convolutional network.
	 * The output of the the main convolutional network is unified as content called unified content.
	 * The input of the the main convolutional network is raster flattened as array of neuron values.
	 */
	protected List<Stack> stacks = Util.newList(0);
	
	
	/**
	 * Fully connected network (full network) which converts output to feature.
	 * Input of the full network is flatten from the unified output of convolutional network (the main network).
	 * The output of fully connected network is always feature/class.
	 */
	protected NetworkStandardImpl fullNetwork = null;
	
	
	/**
	 * Reversed fully connected network (reserved full network).
	 * If the full network is not null, this reversed full network will convert feature to input of the full network.
	 * If the full network is null, this reversed full network will convert feature to input of the convolutional network.
	 * The input of reversed fully connected network is always feature.
	 */
	protected NetworkStandardImpl reversedFullNetwork = null;

	
	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;
	

	/**
	 * Neuron channel of full network.
	 */
	protected int fullNetworkNeuronChannel = 1;
	
	
	/**
	 * Activation function reference.
	 */
	protected Function activateRef = null;

	
	/**
	 * Content activation function reference, which is often activation function related to convolutional pixel like ReLU function.
	 */
	protected Function contentActivateRef = null;

	
	/**
	 * Flag to indicate whether to pad zero when filering.
	 */
	protected boolean isPadZeroFilter = false;

	
	/**
	 * If this parameter is true, the evaluation process is performed only forward in the convolutional network {@link #stacks}.
	 * In other words, weights are ignored in the convolutional network {@link #stacks} if this parameter is true so that only convolutional filters are focused.
	 * Therefore, learning task is only done if this parameter is false.
	 * In current version, this parameter is nearly set to be fixed as constant (true).
	 */
	protected boolean onlyForward = true;

	
	/**
	 * Unified output content which is the unified output of the main convolutional network.
	 */
	private Content unifiedOutputContent = null;
	
	
	/**
	 * Backpropagation algorithm.
	 */
	protected Backpropagator bp = null;

	
	/**
	 * Constructor with neuron channel, activation functions, and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function, which is often activation function related to weights like sigmod function.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 * @param idRef ID reference.
	 */
	protected StackNetworkAbstract(int neuronChannel, Function activateRef, Function contentActivateRef, Id idRef) {
		super(idRef);
		
		this.config.put(LEARN_MAX_ITERATION_FIELD, 1);
		this.config.put(LEARNING_FILTERS_FIELD, LEARNING_FILTERS_DEFAULT);
		this.config.put(Raster.NORM_FIELD, Raster.NORM_DEFAULT);
		this.config.put(Image.ALPHA_FIELD, Image.ALPHA_DEFAULT);

		if (neuronChannel < 1)
			this.neuronChannel = neuronChannel = 1;
		else
			this.neuronChannel = neuronChannel;
		this.fullNetworkNeuronChannel = this.neuronChannel;
		
		if (activateRef == null && contentActivateRef == null)
			this.contentActivateRef = this.activateRef = contentActivateRef = activateRef = Raster.toConvActivationRef(this.neuronChannel, isNorm());
		else if (activateRef != null && contentActivateRef != null) {
			this.activateRef = activateRef;
			this.contentActivateRef = contentActivateRef;
		}
		else if (activateRef != null)
			this.contentActivateRef = this.activateRef = contentActivateRef = activateRef; 
		else
			this.contentActivateRef = this.activateRef = activateRef = contentActivateRef;
		
		if (this.activateRef != null && this.contentActivateRef != null && this.activateRef instanceof NormSupporter && this.contentActivateRef instanceof NormSupporter) {
			NormSupporter ns1 = (NormSupporter)this.activateRef;
			NormSupporter ns2 = (NormSupporter)this.contentActivateRef;
			if (ns1.isNorm() == ns2.isNorm() && ns1.isNorm() != isNorm()) this.config.put(Raster.NORM_FIELD, ns1.isNorm());
		}
		
		this.bp = new BackpropagatorAbstract() {
			
			/**
			 * Serial version UID for serializable class. 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			protected Content calcOutputError(ElementLayer outputLayer, Content realOutput, Stack outputStack) {
				return thisNetwork().calcOutputError(outputLayer, realOutput, outputStack);
			}

		};
	}

	
	/**
	 * Constructor with neuron channel and activation functions.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function, which is often activation function related to weights like sigmod function.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 */
	protected StackNetworkAbstract(int neuronChannel, Function activateRef, Function contentActivateRef) {
		this(neuronChannel, activateRef, contentActivateRef, null);
	}


	/**
	 * Getting this network.
	 * @return this network.
	 */
	protected StackNetworkAbstract thisNetwork() {return this;}

	
	/**
	 * Resetting this network.
	 */
	public void reset() {
		stacks.clear();
		
		if (fullNetwork != null) {
			try {
				fullNetwork.removeListener(this);
			} catch (Throwable e) {Util.trace(e);}
		}
		fullNetwork = null;
		
		if (reversedFullNetwork != null) {
			try {
				reversedFullNetwork.removeListener(this);
			} catch (Throwable e) {Util.trace(e);}
		}
		reversedFullNetwork = null;
		
		unifiedOutputContent = null;
	}

	
	/**
	 * Initialize with image/raster specification and filters.
	 * @param size stack content size.
	 * @param filters specific filters.
	 * @param nFullHiddenOutputNeuron numbers of hidden neurons and output neurons of fully connected network.
	 * @param initReverse flag to indicate whether to initialize the reversed full network.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			Filter[] filters,
			int[] nFullHiddenOutputNeuron,
			boolean initReverse) {
		stacks.clear();
		Size newSize = new Size(size.width, size.height, size.depth, size.time);
		if (filters == null || filters.length == 0) { //Allowing to have only one stack.
			stacks.add(newStack(newSize));
		}
		else {
			Stack stack = null;
			stack = addStacks(filters, newSize, stack);
			if (stack == null) return false;
			
			Filter.calcSize(newSize, filters[filters.length-1]);
			Stack lastStack = newStack(newSize);
			if (lastStack != null) {
				stacks.add(lastStack);
				stack.setNextStack(lastStack);
				stack = lastStack;
			}
		}
		
		if (nFullHiddenOutputNeuron == null || nFullHiddenOutputNeuron.length < 1) return true;

		return initializeFullNetwork(nFullHiddenOutputNeuron, initReverse);
	}

	
	/**
	 * Initialize with image/raster specification and filters.
	 * @param size stack content size.
	 * @param filters specific filters.
	 * @param nFullHiddenOutputNeuron numbers of hidden neurons and output neurons of fully connected network.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			Filter[] filters,
			int[] nFullHiddenOutputNeuron) {
		return initialize(size, filters, nFullHiddenOutputNeuron, false);
	}
	
	
	/**
	 * Initialize with image/raster specification and filters.
	 * @param size stack content size.
	 * @param filters specific filters.
	 * @param initReverse flag to indicate whether to initialize the reversed full network.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			Filter[] filters,
			boolean initReverse) {
		return initialize(size, filters, null, initReverse);
	}

	
	/**
	 * Initialize with image/raster specification without reversed fully connected network.
	 * @param size stack content size.
	 * @param filters specific filters.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			Filter[] filters) {
		return initialize(size, filters, null, false);
	}

	
	/**
	 * Initialize with image/raster specification.
	 * @param size stack content size.
	 * @param nFullHiddenOutputNeuron numbers of hidden neurons and output neurons of fully connected network.
	 * @param initReverse flag to indicate whether to initialize the reversed full network.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			int[] nFullHiddenOutputNeuron,
			boolean initReverse) {
		return initialize(size, (Filter[])null, nFullHiddenOutputNeuron, initReverse);
	}
	
	
	/**
	 * Initialize with image/raster specification.
	 * @param size stack content size.
	 * @param nFullHiddenOutputNeuron numbers of hidden neurons and output neurons of fully connected network.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			int[] nFullHiddenOutputNeuron) {
		return initialize(size, (Filter[])null, nFullHiddenOutputNeuron, false);
	}

	
	/**
	 * Initialize with image/raster specification.
	 * @param size stack content size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size) {
		return initialize(size, (Filter[])null, null, false);
	}

	
	/**
	 * Initialize with image/raster specification and filters.
	 * @param size stack content size.
	 * @param filterArrays arrays of filters. Filters in the same array have the same size.
	 * @param nFullHiddenOutputNeuron numbers of hidden neurons and output neurons of fully connected network.
	 * @param initReverse flag to indicate whether to initialize the reversed full network.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			Filter[][] filterArrays,
			int[] nFullHiddenOutputNeuron,
			boolean initReverse) {
		if (filterArrays == null || filterArrays.length == 0) return initialize(size, (Filter[])null, nFullHiddenOutputNeuron, initReverse);
		
		stacks.clear();
		Size newSize = new Size(size.width, size.height, size.depth, size.time);
		Stack stack = null;
		stack = addStacks(filterArrays, newSize, stack);
		if (stack == null) return false;
		
		Filter.calcSize(newSize, filterArrays[filterArrays.length-1][0]);
		Stack lastStack = newStack(newSize, null, filterArrays[filterArrays.length-1].length);
		if (lastStack != null) {
			stacks.add(lastStack);
			stack.setNextStack(lastStack, true);
			stack = lastStack;
		}
		
		if (nFullHiddenOutputNeuron == null || nFullHiddenOutputNeuron.length < 1) return true;

		return initializeFullNetwork(nFullHiddenOutputNeuron, initReverse);
	}

	
	/**
	 * Initialize with image/raster specification and filters.
	 * @param size stack content size.
	 * @param filterArrays array of filters. Filters in the same row have the same size.
	 * @param nFullHiddenOutputNeuron numbers of hidden neurons and output neurons of fully connected network.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			Filter[][] filterArrays,
			int[] nFullHiddenOutputNeuron) {
		return initialize(size, filterArrays, nFullHiddenOutputNeuron, false);
	}
	
	
	/**
	 * Initialize with image/raster specification.
	 * @param size stack content size.
	 * @param filterArrays array of filters. Filters in the same row have the same size.
	 * @param initReverse flag to indicate whether to initialize the reversed full network.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			Filter[][] filterArrays,
			boolean initReverse) {
		return initialize(size, filterArrays, null, initReverse);
	}

	
	/**
	 * Initialize with image/raster specification without reversed fully connected network.
	 * @param size stack content size.
	 * @param filterArrays array of filters. Filters in the same row have the same size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			Filter[][] filterArrays) {
		return initialize(size, filterArrays, null, false);
	}

	
	/**
	 * Initializing full network (FN).
	 * @param nFullHiddenOutputNeuron numbers of hidden neurons and output neurons of fully connected network.
	 * @param initReverse flag to indicate whether to initialize the reversed full network.
	 * @return true if initialization is successful.
	 */
	protected boolean initializeFullNetwork(int[] nFullHiddenOutputNeuron, boolean initReverse) {
		if (nFullHiddenOutputNeuron == null || nFullHiddenOutputNeuron.length < 1) return false;
		
		Content content = unifyOutputContent();
		if (content == null) return false;

		int rInputNeuron = 0, rOutputNeuron = 0;
		int[] rHiddenNeuron = null;
		fullNetwork = new NetworkStandardImpl(fullNetworkNeuronChannel, activateRef);
		int nInputNeuron = getFullNetworkNeuronChannelRatio() * content.getWidth() * content.getHeight() * content.getDepth() * content.getTime();
		if (nFullHiddenOutputNeuron.length == 1) {
			if (!fullNetwork.initialize(nInputNeuron, nFullHiddenOutputNeuron[0])) return false;
			rInputNeuron = nFullHiddenOutputNeuron[0];
			rOutputNeuron = nInputNeuron;
		}
		else {
			int length = nFullHiddenOutputNeuron.length;
			int[] nHiddenNeuron = Arrays.copyOf(nFullHiddenOutputNeuron, length-1);
			if (!fullNetwork.initialize(nInputNeuron, nFullHiddenOutputNeuron[length-1], nHiddenNeuron)) return false;
			
			rInputNeuron = nFullHiddenOutputNeuron[length-1];
			rOutputNeuron = nInputNeuron;
			rHiddenNeuron = new int[nHiddenNeuron.length];
			for (int i = 0; i < nHiddenNeuron.length; i++) rHiddenNeuron[i] = nHiddenNeuron[nHiddenNeuron.length - i - 1];
		}
		try {
			fullNetwork.addListener(this);
		} catch (Throwable e) {Util.trace(e);}
		
		if (!initReverse) return true;
		
		if (rInputNeuron == 0 || rOutputNeuron == 0) return false;
		reversedFullNetwork = new NetworkStandardImpl(fullNetworkNeuronChannel, activateRef);
		if (!reversedFullNetwork.initialize(rInputNeuron, rOutputNeuron, rHiddenNeuron)) return false;
		try {
			reversedFullNetwork.addListener(this);
		} catch (Throwable e) {Util.trace(e);}
		
		return true;
	}
	
	
	/**
	 * Getting ratio of full network neuron channel to neuron channel.
	 * @return ratio of full network neuron channel to neuron channel.
	 */
	protected int getFullNetworkNeuronChannelRatio() {
		int ratio = neuronChannel / fullNetworkNeuronChannel;
		return ratio < 1 ? 1 : ratio;
	}

	
	/**
	 * Adding stacks according to filters.
	 * @param filters array of filters
	 * @param size size of stack content. This is also output parameter.
	 * @param prevStack previous stack.
	 * @return current added stack, which is the last stack.
	 */
	private Stack addStacks(Filter[] filters, Size size, Stack prevStack) {
		if (filters == null || filters.length == 0) return prevStack;
		for (int i = 0; i < filters.length; i++) {
			if (i > 0) Filter.calcSize(size, filters[i-1]);
			Stack newStack = newStack(size, filters[i]);
			stacks.add(newStack);

			if (prevStack != null) prevStack.setNextStack(newStack);
			prevStack = newStack;
		}
		
		return prevStack;
	}

	
	/**
	 * Adding stacks according to filters.
	 * @param filterArrays array of filters. Filters in the same row have the same size.
	 * @param size size of stack content. This is also output parameter.
	 * @param prevStack previous stack.
	 * @return current added stack, which is the last stack.
	 */
	private Stack addStacks(Filter[][] filterArrays, Size size, Stack prevStack) {
		if (filterArrays == null || filterArrays.length == 0) return prevStack;
		for (int i = 0; i < filterArrays.length; i++) {
			if (i > 0) Filter.calcSize(size, filterArrays[i-1][0]);
			Stack newStack = newStack(size, filterArrays[i]);
			stacks.add(newStack);

			if (prevStack != null) prevStack.setNextStack(newStack, true);
			prevStack = newStack;
		}
		
		return prevStack;
	}

	
	/**
	 * Creating stack. Derived class can override this method.
	 * @return created stack.
	 */
	protected Stack newStack() {
		return StackImpl.create(neuronChannel, idRef);
	}
	
	
	/**
	 * Creating stack with specified size and filters.
	 * @param size specified size.
	 * @param filters specified filters.
	 * @return created stack.
	 */
	public Stack newStack(Size size, Filter...filters) {
		Stack stack = newStack();
		if (filters == null || filters.length == 0) {
			ElementLayer layer = stack.newLayer(activateRef, contentActivateRef, size, null);
			layer.setPadZeroFilter(isPadZeroFilter);
			stack.add(layer);
		}
		else {
			for (Filter filter : filters) {
				ElementLayer layer = stack.newLayer(activateRef, contentActivateRef, size, filter);
				layer.setPadZeroFilter(isPadZeroFilter);
				stack.add(layer);
			}
		}
		
		return stack;
	}
	
	
	/**
	 * Creating stack with specified size, filter, and number of layers.
	 * @param size specified size.
	 * @param filter specified filter. It can be null.
	 * @param nLayers number of layers.
	 * @return created stack.
	 */
	private Stack newStack(Size size, Filter filter, int nLayers) {
		if (nLayers < 1) return null;
		Stack stack = newStack();
		for (int i = 0; i < nLayers; i++) {
			ElementLayer layer = stack.newLayer(activateRef, contentActivateRef, size, filter);
			layer.setPadZeroFilter(isPadZeroFilter);
			stack.add(layer);
		}
		
		return stack;
	}
	
	
	/**
	 * Creating content with neuron channel, activation function, size, and filter. This content can be different from the one created from stack.
	 * Derived class can override this method.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param size specified size.
	 * @param filter kernel filter.
	 * @return created content.
	 */
	protected Content newContent(int neuronChannel, Function activateRef, Size size, Filter filter) {
		Content content = ContentImpl.create(neuronChannel, activateRef, size, filter, idRef);
		content.setPadZeroFilter(isPadZeroFilter);
		return content;
	}

	
	/**
	 * Getting filter factory.
	 * @return filter factory.
	 */
	public FilterFactory getFilterFactory() {
		Stack stack = newStack(new Size(1, 1, 1, 1));
		return new FilterFactoryImpl(stack.get(0).getContent());
	}
	
	
	@Override
	public synchronized NeuronValue[] evaluateRaster(Raster inputRaster) throws RemoteException {
		if (inputRaster == null || stacks.size() == 0) return null;
		Stack inputStack = stacks.get(0);
		Content content = inputStack.get(0).getContent();
		if (content == null) return null;
		
		NeuronValue[] input = inputRaster.toNeuronValues(content, isNorm());
		return evaluate(input);
	}

	
	/*
	 * Order of input to be evaluated in the input record is:
	 * 1. Record.input
	 * 2. RecordExt.contentInput
	 * 3. RecordExt.undefinedInput as raster input.
	 */
	@Override
	public NeuronValue[] evaluate(Record inputRecord) throws RemoteException {
		if (inputRecord == null)
			return null;
		else if (inputRecord.input != null)
			return evaluate(inputRecord.input);
		else if (inputRecord instanceof RecordExt) {
			RecordExt inputRecordExt = (RecordExt)inputRecord;
			if (inputRecordExt.contentInput != null)
				return evaluate(inputRecordExt.contentInput);
			else if (inputRecord.getRasterInput() != null)
				return evaluateRaster(inputRecord.getRasterInput());
			else
				return null;
		}
		else if (inputRecord.getRasterInput() != null)
			return evaluateRaster(inputRecord.getRasterInput());
		else
			return null;
	}
	
	
	/**
	 * Evaluate convolutional stack network by content array.
	 * @param input array of content as input. This is array of contents and so each element is a content that is input for a element layer of the first stack.
	 * @return evaluated array of neuron values.
	 */
	private NeuronValue[] evaluate(Content...input) {
		if (input == null || input.length == 0) return null;
		
		List<NeuronValue[]> datas = Util.newList(0);
		for (Content content : input) {
			NeuronValue[] data = content.getData();
			if (data != null) datas.add(data);
		}
		
		return evaluate(datas.toArray(new NeuronValue[] {}));
	}
	
	
	/**
	 * Evaluate convolutional stack network by value input array.
	 * @param input array of neuron values as input.
	 * It is array of neuron values array, which are contents for the input stack where each array is a content for an element layer of the first stack.
	 * @return evaluated array of neuron values.
	 */
	public synchronized NeuronValue[] evaluate(NeuronValue[]...input) {
		if (stacks.size() == 0 || input == null || input.length == 0) return null;
		
		Stack inputStack = stacks.get(0);
		if (inputStack == null || inputStack.size() == 0) return null;
		NeuronValue[] input0 = null;
		if (stacks.size() == 1 && inputStack.size() == 1)
			input0 = inputStack.setContent(input);
		else
			inputStack.setContent(input);
		
		if (stacks.size() > 1) {
			for (int i = 0; i < stacks.size(); i++) {
				if ((onlyForward) && (i < stacks.size()-1)) stacks.get(i).forward();
				if ((!onlyForward) && (i > 0)) stacks.get(i).evaluate();
			}
		}

		//This code line is important for updating unified output content.
		unifyOutputContent();
		
		if (fullNetwork != null) {
			Record record = new Record();
			record.input = convertUnifiedContentToFullNetworkInput(null); //input0 != null ? input0 : convertUnifiedContentToFullNetworkInput(null);
			try {
				return fullNetwork.evaluate(record);
			} catch (Throwable e) {Util.trace(e);}
			return null;
		}
		else if (input0 != null && reversedFullNetwork == null)
			return input0;
		else { //Setting more for reserved full network in the next version.
			Content content = null;
			try {
				content = getFeature();
			} catch (Throwable e) {Util.trace(e);}
			return (content != null ? content.getData() : null);
		}
	}

	
	/**
	 * Converting feature into unified content data. The content data can be set to the unified content
	 * by calling {@link #getUnifiedOutputContent(boolean)}.setData({@link #convertFeatureToUnifiedContentData(NeuronValue[])}.
	 * @param feature specified feature represented by array of neuron values.
	 * @return array of neuron values as data of unified content.
	 */
	NeuronValue[] convertFeatureToUnifiedContentData(NeuronValue[] feature) {
		NeuronValue[] contentData = null;
		if (fullNetwork != null) {
			try {
				if (reversedFullNetwork != null) {
					contentData = reversedFullNetwork.evaluate(new Record(feature));
					contentData = convertFullNetworkInputToUnifiedContent(contentData).getData();
				}
				else {
					//In the case that the full network is not null but the reserved full network is null,
					//then the feature is considered as input of the full network and output of the full network may be classes in classification learning. 
					contentData = convertFullNetworkInputToUnifiedContent(feature).getData();;
				}
			} catch (Throwable e) {Util.trace(e);}
		}
		else {
			try {
				//In the case that the full network is null, the output of the reversed full network is considered is the first feature and
				//the output of the convolutional network (the entire network) is the final feature. note, the output of the reversed full network is input of the convolutional network.
				contentData = reversedFullNetwork != null ? reversedFullNetwork.evaluate(new Record(feature)) : feature;
				contentData = evaluate(convertReversedFullNetworkOutputToConvInput(contentData));
			} catch (Throwable e) {Util.trace(e);}
		}
		
		return contentData;
	}
	
	
	/**
	 * Converting unified content to full network input. This method, which is flattening method, should be overridden.
	 * @param unifiedContent unified content.
	 * @return converted unified content.
	 */
	NeuronValue[] convertUnifiedContentToFullNetworkInput(Content unifiedContent) {
		if (unifiedContent == null) unifiedContent = getUnifiedOutputContent();
		if (unifiedContent == null) return null;
		
		NeuronValue[] input = unifiedContent.getData();
		if (input == null || fullNetwork == null || fullNetwork.getNeuronChannel() == unifiedContent.getNeuronChannel())
			return input;
		else
			return input[0].flatten(input, fullNetwork.getNeuronChannel());
	}


	/**
	 * Converting full network input to unified content. This method, which is aggregation method, should be overridden.
	 * @param fnInput full network input.
	 * @return converted full network input, which is the content of the first element layer of the first stack.
	 */
	Content convertFullNetworkInputToUnifiedContent(NeuronValue[] fnInput) {
		Content unifiedContent = getUnifiedOutputContent();
		if (unifiedContent == null) return null;

		if (fnInput == null || fullNetwork == null || fullNetwork.getNeuronChannel() == unifiedContent.getNeuronChannel())
			return unifiedContent.newContent(fnInput, null);
		else {
			fnInput = fnInput[0].aggregate(fnInput, unifiedContent.getNeuronChannel());
			return unifiedContent.newContent(fnInput, null);
		}
	}

	
	/**
	 * Converting reversed full network output to convolutional network input (maybe main network input). This method, which is aggregation method, should be overridden.
	 * This method is valid regardless of that the full network is null or not null but is is often used when the full network is null.
	 * If the full network is not null, the return value can be flattened one more time to become input of the full network.
	 * @param rfnOutput reversed full network output.
	 * @return convolutional network input converted from reversed full network output.
	 */
	NeuronValue[] convertReversedFullNetworkOutputToConvInput(NeuronValue[] rfnOutput) {
		if (rfnOutput == null || rfnOutput.length == 0 || stacks.size() == 0 || stacks.get(0).size() == 0) return rfnOutput;
		int neuronChannel = stacks.get(0).get(0).getContent().getNeuronChannel();
		if (rfnOutput[0].length() == neuronChannel)
			return rfnOutput;
		else
			return rfnOutput[0].aggregate(rfnOutput, neuronChannel);
	}

	
	/**
	 * Unifying output content. This method should be overridden.
	 * @return unified output content is output of the last stack of the convolutional network which is unified into one content. 
	 */
	protected Content unifyOutputContent() {
		if (stacks.size() == 0) return (unifiedOutputContent = null);
		Stack lastStack = stacks.get(stacks.size() - 1);
		if (lastStack.size() == 0) return (unifiedOutputContent = null);
		if (lastStack.size() == 1) return (unifiedOutputContent = lastStack.get(0).getContent());
		
		List<Content> contents = Util.newList(lastStack.size());
		for (int i = 0; i < lastStack.size(); i++) {
			contents.add(lastStack.get(i).getContent());
		}
		return (unifiedOutputContent = ContentImpl.aggregate(contents));
	}
	

	/**
	 * Getting unified output content.
	 * @param update flag to indicate whether to update unified content.
	 * @return unified output content is output of the last stack of the convolutional network which is unified into one content. 
	 */
	private Content getUnifiedOutputContent(boolean update) {
		if (unifiedOutputContent == null || update) unifyOutputContent();
		return unifiedOutputContent;
	}
	
	
	/**
	 * Getting unified output content.
	 * @return unified output content is output of the last stack of the convolutional network which is unified into one content. 
	 */
	public Content getUnifiedOutputContent() {
		return getUnifiedOutputContent(true);
	}
	
	
	/**
	 * Getting size of unified output content.
	 * @return unified output content size.
	 */
	public Size getUnifiedOutputContentSize() {
		Content content = getUnifiedOutputContent();
		return content != null ? new Size(content.getWidth(), content.getHeight(), content.getDepth(), content.getTime()) : null;
	}
	
	
	@Override
	public Content getFeature() throws RemoteException {
		if (fullNetwork == null) {
			if (reversedFullNetwork == null) return getUnifiedOutputContent();
			
			//Feature now is considered as a flattened 1-dimension vector.
			Content feature = newContent(reversedFullNetwork.getNeuronChannel(), reversedFullNetwork.getActivateRef(),
				new Size(reversedFullNetwork.getInputLayer().size(), 1, 1, 1), null);
			feature.setData(reversedFullNetwork.getInputLayer().getOutput());
			return feature;
		}
		else {
			//Feature now is considered as a flattened 1-dimension vector.
			Content feature = newContent(fullNetwork.getNeuronChannel(), fullNetwork.getActivateRef(),
					new Size(fullNetwork.getOutputLayer().size(), 1, 1, 1), null);
			feature.setData(fullNetwork.getOutputLayer().getOutput());
			return feature;
		}
	}


	/**
	 * Getting feature fit to neuron channel of this network.
	 * This method is more precise but slower than the called method {@link #getFeature()}.
	 * @return feature fit to neuron channel.
	 */
	public Content getFeatureFitChannel() {
		Content content = null;
		try {
			content = getFeature();
		} catch (Throwable e) {Util.trace(e);}
		if (content == null || content.length() == 0 || content.getNeuronChannel() == this.neuronChannel) return content;
		
		NeuronValue[] data = new NeuronValue[content.length()];
		for (int i = 0; i < data.length; i++) {
			data[i] = content.get(i).getValue().resize(this.neuronChannel);
		}
		Content newContent = newContent(this.neuronChannel, content.getActivateRef(),
			new Size(content.getWidth(), content.getHeight(), content.getDepth(), content.getTime()), null);
		newContent.setData(data);
		return newContent;
	}
	
	
	/**
	 * Get size of feature.
	 * @return feature size.
	 */
	public Size getFeatureSize() {
		if (fullNetwork == null) {
			Content feature = null;
			try {
				feature = getFeature();
				return new Size(feature.getWidth(), feature.getHeight(), feature.getDepth(), feature.getTime());
			} catch (Throwable e) {Util.trace(e);}
			return null;
		}
		else {
			return new Size(fullNetwork.getOutputLayer().size(), 1, 1, 1);
		}
	}
	
	
	/**
	 * Creating raster from feature when feature is represented as array of neuron values.
	 * @param feature specified feature.
	 * @return raster.
	 */
	public Raster createRaster(NeuronValue[] feature) {
		NeuronValue[] contentData = convertFeatureToUnifiedContentData(feature);
		if (contentData == null) return null;
		Content content = getUnifiedOutputContent();
		return content != null ? content.createRaster(contentData, isNorm(), getDefaultAlpha()) : null;
	}
	
	
	/**
	 * Checking whether padding zero when filtering.
	 * @return whether padding zero when filtering.
	 */
	public boolean isPadZeroFilter() {
		return isPadZeroFilter;
	}
	
	
	/**
	 * Setting whether to pad zero when filtering.
	 * @param isPadZeroFilter flag to indicate whether to pad zero when filtering.
	 */
	public void setPadZeroFilter(boolean isPadZeroFilter) {
		this.isPadZeroFilter = isPadZeroFilter;
	}

	
	@Override
	public NeuronValue[] learnOne(Iterable<Record> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = config.getAsReal(LEARN_RATE_FIELD);
		return learnOne(sample, learningRate, terminatedThreshold, maxIteration);
	}


	@Override
	public NeuronValue[] learn(Iterable<Record> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = config.getAsReal(LEARN_RATE_FIELD);
		return learn(sample, learningRate, terminatedThreshold, maxIteration);
	}

	
	/**
	 * Learning neural network by back propagate algorithm, one-by-one record over sample.<br>
	 * One of these fields 1. {@link Record#input}, or 2. {@link RecordExt#contentInput}, or 3. raster {@link Record#undefinedInput} is used to evaluate the convolutional network (main network).<br>
	 * The field {@link Record#output} as feature is used to train full network and reserved full network.<br>
	 * The field {@link RecordExt#contentOutput} is used to train the convolutional network (filters).
	 * @param sample learning sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	public NeuronValue[] learnOne(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		if (stacks.size() < 1) return null;
		if (stacks.size() < 2 && !onlyForward) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_DEFAULT;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			sample = resample(sample, iteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration);

			for (Record record : sample) {
				if (record == null) continue;
				
				Content[] output = null; //This is the second output which is content output for learning convolutional network.
				Stack lastStack = stacks.get(stacks.size() - 1);
				if (record instanceof RecordExt) {
					RecordExt recordExt = (RecordExt)record;
					output = recordExt.contentOutput != null? StackAbstract.adjustArray(recordExt.contentOutput, lastStack.size(), lastStack) : null;
				}
				
				//Evaluating layers.
				try {
					evaluate(record);
				} catch (Throwable e) {Util.trace(e);}
				
				Content[] contentError = null;
				if (!onlyForward && output != null) {
					//Learning stack list.
					contentError = bp.updateWeightsBiases(stacks, output, lr);
				}
				
				Content unifiedContent = getUnifiedOutputContent(false);
				boolean realError = false;
				if (fullNetwork != null) {
					try {
						error = fullNetwork.learn(convertUnifiedContentToFullNetworkInput(unifiedContent), record.output, lr, terminatedThreshold, 1);
						realError = true;
					} catch (Throwable e) {Util.trace(e);}
				}
				else if (!onlyForward && contentError != null) {
					error = new NeuronValue[contentError.length];
					for (int i = 0; i < error.length; i++) error[i] = contentError[i].mean0();
					realError = true;
				}
				else
					error = new NeuronValue[] {lastStack.newNeuronValue().zero()};
				
				if (reversedFullNetwork != null) {
					try {
						NeuronValue[] rfnOutput = null;
						if (fullNetwork != null)
							rfnOutput = convertUnifiedContentToFullNetworkInput(unifiedContent);
						else if (record != null && record.input != null)
							rfnOutput = record.input[0].flatten(reversedFullNetwork.getNeuronChannel()); 
						NeuronValue[] rfnError = reversedFullNetwork.learn(record.output, rfnOutput, lr, terminatedThreshold, 1);
						if (!realError) error = rfnError;
					} catch (Throwable e) {Util.trace(e);}
				}
				
				//Be careful to learn filter because filter is only learned from large layer to small layer in this current version.
				if (config.getAsBoolean(LEARNING_FILTERS_FIELD)) learnFilters(lr, 1);
				
			} //End for
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "stacknn_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else {
				double errorMean = 0;
				for (NeuronValue r : error) errorMean += r.norm();
				errorMean = errorMean / error.length;
				if (errorMean < terminatedThreshold) doStarted = false; 
			}
			
			synchronized (this) {
				while (doPaused) {
					notifyAll();
					try {
						wait();
					} catch (Exception e) {Util.trace(e);}
				}
			}

		}
		
		synchronized (this) {
			doStarted = false;
			doPaused = false;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "stacknn_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}

	
	/**
	 * Learning neural network by back propagate algorithm.<br>
	 * One of these fields 1. {@link Record#input}, or 2. {@link RecordExt#contentInput}, or 3. raster {@link Record#undefinedInput} is used to evaluate the convolutional network (main network).<br>
	 * The field {@link Record#output} as feature is used to train full network and reserved full network.<br>
	 * The field {@link RecordExt#contentOutput} is used to train the convolutional network (filters).
	 * @param sample learning sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	public NeuronValue[] learn(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		if (stacks.size() < 1) return null;
		if (stacks.size() < 2 && !onlyForward) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_DEFAULT;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			sample = resample(sample, iteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration);

			Content[] contentError = null;
			if (!onlyForward) {
				//Learning stack list.
				contentError = bp.updateWeightsBiases(sample, stacks, lr, this);
			}
			
			List<Record> fnSample = Util.newList(0), rfnSample = Util.newList(0);
			for (Record record : sample) {
				NeuronValue[] fnInput = null, rfnOutput = null;
				try {
					//Evaluating layers.
					evaluate(record);
					Content unifiedContent = getUnifiedOutputContent(false);
					if (fullNetwork != null) fnInput = convertUnifiedContentToFullNetworkInput(unifiedContent);
					if (reversedFullNetwork != null) {
						if (fullNetwork != null)
							rfnOutput = convertUnifiedContentToFullNetworkInput(unifiedContent);
						else if (record != null && record.input != null)
							rfnOutput = record.input[0].flatten(reversedFullNetwork.getNeuronChannel()); 
					}
				} catch (Throwable e) {Util.trace(e);}
				
				if (fnInput != null) fnSample.add(new Record(fnInput, record.output));
				if (rfnOutput != null) rfnSample.add(new Record(record.output, rfnOutput));
			}
			
			if (fullNetwork != null)
				error = fullNetwork.learn(fnSample, lr, terminatedThreshold, 1);
			else if (!onlyForward && contentError != null) {
				error = new NeuronValue[contentError.length];
				for (int i = 0; i < error.length; i++) error[i] = contentError[i].mean0();
			}
			else
				error = new NeuronValue[] {stacks.get(stacks.size() - 1).newNeuronValue().zero()};
			
			if (reversedFullNetwork != null) 
				error = reversedFullNetwork.learn(rfnSample, lr, terminatedThreshold, 1);
			
			//Be careful to learn filter because filter is only learned from large layer to small layer in this current version.
			if (config.getAsBoolean(LEARNING_FILTERS_FIELD)) learnFilters(lr, 1);

			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "stacknn_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else {
				double errorMean = 0;
				for (NeuronValue r : error) errorMean += r.norm();
				errorMean = errorMean / error.length;
				if (errorMean < terminatedThreshold) doStarted = false; 
			}
			
			synchronized (this) {
				while (doPaused) {
					notifyAll();
					try {
						wait();
					} catch (Exception e) {Util.trace(e);}
				}
			}

		}
		
		synchronized (this) {
			doStarted = false;
			doPaused = false;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "stacknn_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}

	
	/**
	 * Calculating output error. Derived classes should implement this method.
	 * This error is the opposite of gradient of minimized target function and the gradient of maximized target function.
	 * The real output can be null in some cases because the error may not be calculated by squared error function that needs real output. 
	 * @param outputLayer output layer.
	 * @param realOutput real output. It can be null.
	 * @param outputStack output stack. It can be null.
	 * @return output error.
	 */
	protected Content calcOutputError(ElementLayer outputLayer, Content realOutput, Stack outputStack) {
		return BackpropagatorAbstract.calcOutputErrorDefault(outputLayer, realOutput, outputStack);
	}

	
	/**
	 * Learning filters.
	 * @param learningRate learning rate.
	 * @param maxIteration maximum iteration.
	 */
	void learnFilters(double learningRate, int maxIteration) {
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_DEFAULT;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		for (Stack stack : stacks) {
			for (int i = 0; i < stack.size(); i++) {
				Content content = stack.get(i).getContent();
				if (content == null) continue;
				
				try {
					Filter filter = content.getFilter();
					BiasFilter biasFilter = filter != null ? new BiasFilter(filter, content.getBias()) : null;
					biasFilter = content.learnFilter(biasFilter, true, learningRate, maxIteration);
					
					if (biasFilter != null && biasFilter.filter != null) content.setFilter(biasFilter.filter);
					if (biasFilter != null && biasFilter.bias != null) content.setBias(biasFilter.bias);
				} catch (Throwable e) {Util.trace(e);}
			}
		}
	}
	
	
	/**
	 * Checking whether point values are normalized in rang [0, 1].
	 * @return whether point values are normalized in rang [0, 1].
	 */
	public boolean isNorm() {
		if (config.containsKey(Raster.NORM_FIELD))
			return config.getAsBoolean(Raster.NORM_FIELD);
		else
			return Raster.NORM_DEFAULT;
	}

	
	/**
	 * Getting default alpha.
	 * @return default alpha.
	 */
	int getDefaultAlpha() {
		if (config.containsKey(Image.ALPHA_FIELD))
			return config.getAsInt(Image.ALPHA_FIELD);
		else
			return Image.ALPHA_DEFAULT;
	}
	

//	/**
//	 * Checking whether only forwarding mode is set.
//	 * @return whether only forwarding mode is set.
//	 */
//	public boolean isOnlyForward() {
//		return onlyForward;
//	}
//	
//	
//	/**
//	 * Setting only forwarding mode.
//	 * @param onlyForward only forwarding mode.
//	 */
//	public void setOnlyForward(boolean onlyForward) {
//		this.onlyForward = onlyForward;
//	}
	
	
	@Override
	public void receivedInfo(NetworkInfoEvent evt) throws RemoteException {
		fireInfoEvent(evt);
	}

	
	@Override
	public void receivedDo(NetworkDoEvent evt) throws RemoteException {
		if (evt.getType() == NetworkDoEvent.Type.doing) {
			fireDoEvent(new NetworkDoEventImpl(this, NetworkDoEvent.Type.doing, "conv", 
				evt.getLearnResult(),
				evt.getProgressStep(), evt.getProgressTotalEstimated()));
		}
		else if (evt.getType() == NetworkDoEvent.Type.done) {
			fireDoEvent(new NetworkDoEventImpl(this, NetworkDoEvent.Type.done, "conv",
					evt.getLearnResult(),
					evt.getProgressStep(), evt.getProgressTotalEstimated()));
		}
	}

	
	@Override
	public void close() throws Exception {
		super.close();
		reset();
	}


}
