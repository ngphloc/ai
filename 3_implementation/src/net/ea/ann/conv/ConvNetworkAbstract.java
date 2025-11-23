/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;

import net.ea.ann.conv.filter.BiasFilter;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.conv.stack.StackNetworkAbstract;
import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.NetworkDoEvent;
import net.ea.ann.core.NetworkDoEvent.Type;
import net.ea.ann.core.NetworkDoEventImpl;
import net.ea.ann.core.NetworkInfoEvent;
import net.ea.ann.core.NetworkListener;
import net.ea.ann.core.NetworkStandardImpl;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Image;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Size;

/**
 * This class is an abstract implementation of convolutional network.
 * <br>
 * Output of the fully connected network (FCN) {@link #fullNetwork} is some feature/some class and output of the reserved fully connected network (RFCN) {@link #reversedFullNetwork} is (flat) unified content.
 * Note, the unified content is output of convolutional/deconvolutional network (main network) {@link #convLayers} and the dimension of feature is always the same to the dimension of FCN and RFCN.<br>
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
public abstract class ConvNetworkAbstract extends NetworkAbstract implements ConvNetwork, NetworkListener {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Name of only forwarding mode.<br/>
	 * If this parameter is true, the evaluation process is performed only forward in the convolutional network {@link #stacks}.
	 * In other words, weights are ignored in the convolutional network {@link #stacks} if this parameter is true so that only convolutional filters are focused.
	 * Therefore, learning task is only done if this parameter is false.
	 * In current version, this parameter is nearly set to be fixed as constant (true).
	 */
	public static final String ONLY_FORWARD_FIELD = StackNetworkAbstract.ONLY_FORWARD_FIELD;

	
	/**
	 * Name of only forwarding mode.
	 */
	public static final boolean ONLY_FORWARD_DEFAULT = StackNetworkAbstract.ONLY_FORWARD_DEFAULT;

	
	/**
	 * Name of learning filters field.
	 */
	public static final String LEARN_FILTERS_FIELD = StackNetworkAbstract.LEARN_FILTERS_FIELD;

	
	/**
	 * Name of learning filters field.
	 */
	public static final boolean LEARN_FILTERS_DEFAULT = StackNetworkAbstract.LEARN_FILTERS_DEFAULT;

	
	/**
	 * Name of extensive learning filters field.
	 */
	protected static final String LEARN_FILTERS_EXT_FIELD = StackNetworkAbstract.LEARN_FILTERS_EXT_FIELD;

	
	/**
	 * Name of extensive learning filter field.
	 */
	protected static final boolean LEARN_FILTERS_EXT_DEFAULT = StackNetworkAbstract.LEARN_FILTERS_EXT_DEFAULT;

	
	/**
	 * Name of learning weights field.
	 */
	public static final String LEARN_WEIGHTS_FIELD = StackNetworkAbstract.LEARN_WEIGHTS_FIELD;

	
	/**
	 * Name of learning weights field.
	 */
	public static final boolean LEARN_WEIGHTS_DEFAULT = StackNetworkAbstract.LEARN_WEIGHTS_DEFAULT;

	
	/**
	 * List of convolutional layers. This is the main convolutional network.
	 * The output of the the main convolutional network is unified as content called unified content.
	 * The input of the the main convolutional network is raster flattened as array of neuron values.
	 */
	protected List<ConvLayerSingle> convLayers = Util.newList(0);
	
	
	/**
	 * Fully connected network which converts output to feature.
	 * Input of the full network is flatten from the unified output of convolutional network (the main network).
	 * The output of fully connected network is always feature/class.
	 */
	protected NetworkStandardImpl fullNetwork = null;
	
	
	/**
	 * Reversed fully connected network.
	 * If the full network is not null, this reversed full network will convert feature to input of full network.
	 * If the full network is null, this reversed full network will convert feature to input of the convolutional network.
	 * The input of reversed fully connected network is always feature.
	 */
	protected NetworkStandardImpl reversedFullNetwork = null;

	
	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;
	

	/**
	 * Activation function reference, which is often activation function related to convolutional pixel like ReLU function.
	 */
	protected Function activateRef = null;

	
	/**
	 * Flag to indicate whether to pad zero when filering.
	 */
	protected boolean isPadZeroFilter = false;

	
	/**
	 * Unified output content.
	 */
	private ConvLayerSingle unifiedOutputContent = null;

	
	/**
	 * Constructor with neuron channel and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function, which is often activation function related to convolutional pixel like ReLU function.
	 * @param idRef ID reference.
	 */
	protected ConvNetworkAbstract(int neuronChannel, Function activateRef, Id idRef) {
		super(idRef);
		
		this.config.put(LEARN_MAX_ITERATION_FIELD, LEARN_MAX_ITERATION_DEFAULT);
		this.config.put(ONLY_FORWARD_FIELD, ONLY_FORWARD_DEFAULT);
		this.config.put(LEARN_FILTERS_FIELD, LEARN_FILTERS_DEFAULT);
		this.config.put(LEARN_FILTERS_EXT_FIELD, LEARN_FILTERS_EXT_DEFAULT);
		this.config.put(LEARN_WEIGHTS_FIELD, LEARN_WEIGHTS_DEFAULT);
		this.config.put(Raster.NORM_FIELD, Raster.NORM_DEFAULT);
		this.config.put(Image.ALPHA_FIELD, Image.ALPHA_DEFAULT);

		if (neuronChannel < 1)
			this.neuronChannel = neuronChannel = 1;
		else
			this.neuronChannel = neuronChannel;
		this.activateRef = activateRef == null ? (activateRef = Raster.toConvActivationRef(this.neuronChannel, isNorm())) : activateRef;
	}

	
	/**
	 * Default constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function, which is often activation function related to convolutional pixel like ReLU function.
	 */
	protected ConvNetworkAbstract(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null);
	}

	
	/**
	 * Resetting this network.
	 */
	public void reset() {
		convLayers.clear();
		fullNetwork = null;
		reversedFullNetwork = null;
		unifiedOutputContent = null;
	}
	
	
	/**
	 * Initialize with image/raster specification and filters.
	 * @param size layer size.
	 * @param filters specific filters.
	 * @param nFullHiddenOutputNeuron numbers of hidden neurons and output neurons of fully connected network.
	 * @param initReverse flag to indicate whether to initialize the reversed full network.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			Filter[] filters,
			int[] nFullHiddenOutputNeuron,
			boolean initReverse) {
		convLayers.clear();
		Size newSize = new Size(size.width, size.height, size.depth, size.time);

		if (filters == null || filters.length == 0) {
			convLayers.add(newLayer(newSize, null));
		}
		else {
			ConvLayerSingle layer = null;
			layer = addConvLayers(filters, newSize, layer);
			if (layer == null) return false;
			
			Filter.calcSize(newSize, filters[filters.length-1]);
			ConvLayerSingle lastLayer = newLayer(newSize, null);
			if (lastLayer != null) {
				convLayers.add(lastLayer);
				layer.setNextLayer(lastLayer);
				layer = lastLayer;
			}
		}
		
		if (nFullHiddenOutputNeuron == null || nFullHiddenOutputNeuron.length < 1) return true;

		return initializeFullNetwork(nFullHiddenOutputNeuron, initReverse);
	}
	
	
	/**
	 * Initialize with image/raster specification and filters.
	 * @param size layer size.
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
	 * Initialize with image/raster specification without fully connected network.
	 * @param size layer size.
	 * @param filters specific filters.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Size size,
			Filter[] filters) {
		return initialize(size, filters, null, false);
	}
	
	
	/**
	 * Initializing full network (FN).
	 * @param nFullHiddenOutputNeuron numbers of hidden neurons and output neurons of fully connected network.
	 * @param initReverse flag to indicate whether to initialize the reversed full network.
	 * @return true if initialization is successful.
	 */
	private boolean initializeFullNetwork(int[] nFullHiddenOutputNeuron, boolean initReverse) {
		if (nFullHiddenOutputNeuron == null || nFullHiddenOutputNeuron.length < 1) return false;
		
		ConvLayerSingle content = unifyOutputContent();
		if (content == null) return false;

		int rInputNeuron = 0, rOutputNeuron = 0;
		int[] rHiddenNeuron = null;
		fullNetwork = new NetworkStandardImpl(neuronChannel, activateRef);
		int nInputNeuron = content.getWidth() * content.getHeight() * content.getDepth() * content.getTime();
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
		} catch (RemoteException e) {Util.trace(e);}
		
		if (!initReverse) return true;
		
		if (rInputNeuron == 0 || rOutputNeuron == 0) return false;
		reversedFullNetwork = new NetworkStandardImpl(neuronChannel, activateRef);
		if (!reversedFullNetwork.initialize(rInputNeuron, rOutputNeuron, rHiddenNeuron)) return false;
		try {
			reversedFullNetwork.addListener(this);
		} catch (RemoteException e) {Util.trace(e);}
		
		return true;
	}

	
	/**
	 * Adding convolutional layers according to filters.
	 * @param filters array of filters
	 * @param size size of layer.
	 * @param prevLayer previous layer.
	 * @return current added layer.
	 */
	protected ConvLayerSingle addConvLayers(Filter[] filters, Size size, ConvLayerSingle prevLayer) {
		if (filters == null || filters.length == 0) return prevLayer;
		for (int i = 0; i < filters.length; i++) {
			if (i > 0) Filter.calcSize(size, filters[i-1]);
			ConvLayerSingle newLayer = newLayer(size, filters[i]);
			convLayers.add(newLayer);
			
			if (prevLayer != null) prevLayer.setNextLayer(newLayer);
			prevLayer = newLayer;
		}
		
		return prevLayer;
	}
	
	
	/**
	 * Creating new convolutional layer with size and filter.
	 * @param size layer size.
	 * @param filter specific filter.
	 * @return created layer.
	 */
	public abstract ConvLayerSingle newLayer(Size size, Filter filter);
	
	
	@Override
	public synchronized NeuronValue[] evaluateRaster(Raster inputRaster) throws RemoteException {
		if (inputRaster == null) return null;
		ConvLayerSingle inputLayer = convLayers.get(0);
		if (inputLayer == null) return null;
		
		NeuronValue[] input = inputRaster.toNeuronValues(inputLayer, isNorm());
		return evaluate(input);
	}


	/**
	 * Evaluate convolutional network by value input.
	 * @param input array of neuron values as input, which is content for the first convolutional layer.
	 * @return evaluated array of neuron values.
	 */
	public synchronized NeuronValue[] evaluate(NeuronValue[] input) {
		if (convLayers.size() == 0 || input == null) return null;
		
		ConvLayer2DAbstract inputLayer = (ConvLayer2DAbstract)convLayers.get(0);
		if (inputLayer == null) return null;
		input = inputLayer.setData(input);
		
		if (convLayers.size() > 1) {
			for (int i = 0; i < convLayers.size() -  1; i++) convLayers.get(i).forward();
		}
		
		//This code line is important for updating unified output content.
		unifyOutputContent();

		if (fullNetwork != null && fullNetwork.getInputLayer() != null) {
			Record record = new Record();
			record.input = convLayers.size() == 1 ? input : convertUnifiedContentToFullNetworkInput(null);
			try {
				return fullNetwork.evaluate(record);
			} catch (Throwable e) {Util.trace(e);}
			return null;
		}
		else if (convLayers.size() == 1 && reversedFullNetwork == null)
			return input;
		else {
			ConvLayerSingle content = null;
			try {
				content = getFeature();
			} catch (Throwable e) {Util.trace(e);}
			return (content != null ? content.getData() : null);
		}
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
	 * @param input content array as input. This is an content array and so, in this current version, only the first content is used for the first layer.
	 * @return evaluated array of neuron values.
	 */
	private NeuronValue[] evaluate(Content...input) {
		if (input == null || input.length == 0) return null;
		return evaluate(input[0].getData());
	}

	
	/**
	 * Converting unified content to full network input. This method, which is flattening method, should be overridden.
	 * @param unifiedContent unified content.
	 * @return converted unified content.
	 */
	NeuronValue[] convertUnifiedContentToFullNetworkInput(ConvLayerSingle unifiedContent) {
		if (unifiedContent == null) unifiedContent = getUnifiedOutputContent(false);
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
	 * @return converted full network.
	 */
	ConvLayerSingle convertFullNetworkInputToUnifiedContent(NeuronValue[] fnInput) {
		ConvLayerSingle unifiedContent = getUnifiedOutputContent(false);
		if (unifiedContent == null) return null;

		ConvLayerSingle content = null;
		if (fnInput == null || fullNetwork == null || fullNetwork.getNeuronChannel() == unifiedContent.getNeuronChannel()) {
			content = newLayer(new Size(unifiedContent.getWidth(), unifiedContent.getHeight(), unifiedContent.getDepth(), unifiedContent.getTime()), unifiedContent.getFilter());
		}
		else {
			fnInput = fnInput[0].aggregate(fnInput, unifiedContent.getNeuronChannel());
			content = newLayer(new Size(unifiedContent.getWidth(), unifiedContent.getHeight(), unifiedContent.getDepth(), unifiedContent.getTime()), unifiedContent.getFilter());
		}
		
		content.setData(fnInput);
		return content;
	}
	

	/**
	 * Converting reversed full network output to convolutional network input (maybe main network input). This method, which is aggregation method, should be overridden.
	 * This method is valid regardless of that the full network is null or not null but is is often used when the full network is null.
	 * If the full network is not null, the return value can be flattened one more time to become input of the full network.
	 * @param rfnOutput reversed full network output.
	 * @return convolutional network input converted from reversed full network output.
	 */
	NeuronValue[] convertReversedFullNetworkOutputToConvInput(NeuronValue[] rfnOutput) {
		if (rfnOutput == null || rfnOutput.length == 0 || convLayers.size() == 0) return rfnOutput;
		int neuronChannel = convLayers.get(0).getNeuronChannel();
		if (rfnOutput[0].length() == neuronChannel)
			return rfnOutput;
		else
			return rfnOutput[0].aggregate(rfnOutput, neuronChannel);
	}

	
	/**
	 * Unifying output content. This method should be overridden.
	 * @return unified output content is output of the last layer of the convolutional network which is unified into one content. 
	 */
	protected ConvLayerSingle unifyOutputContent() {
		if (convLayers.size() == 0)
			return (unifiedOutputContent = null);
		else
			return (unifiedOutputContent = convLayers.get(convLayers.size()-1));
	}

	
	/**
	 * Getting unified convolutional output layer.
	 * @param update flag to indicate whether to update unified content.
	 * @return unified output content is output of the last layer of the convolutional network which is unified into one content. 
	 */
	private ConvLayerSingle getUnifiedOutputContent(boolean update) {
		if (unifiedOutputContent == null || update) unifyOutputContent();
		return unifiedOutputContent;
	}

	
	/**
	 * Getting unified output content.
	 * @return unified output content is output of the last layer of the convolutional network which is unified into one content. 
	 */
	public ConvLayerSingle getUnifiedOutputContent() {
		return getUnifiedOutputContent(true);
	}

	
	/**
	 * Getting size of unified output content.
	 * @return unified output content size.
	 */
	public Size getUnifiedOutputContentSize() {
		ConvLayerSingle content = getUnifiedOutputContent();
		return content != null ? new Size(content.getWidth(), content.getHeight(), content.getDepth(), content.getTime()) : null;
	}

	
	@Override
	public ConvLayerSingle getFeature() throws RemoteException {
		if (fullNetwork == null) {
			if (reversedFullNetwork == null) return getUnifiedOutputContent(false);
			
			//Feature now is considered as a flattened 1-dimension vector.
			ConvLayerSingle feature = newLayer(new Size(reversedFullNetwork.getInputLayer().size(), 1, 1, 1), null);
			feature.setData(reversedFullNetwork.getInputLayer().getOutput());
			return feature;
		}
		else {
			//Feature now is considered as a flattened 1-dimension vector.
			ConvLayerSingle feature = newLayer(new Size(fullNetwork.getOutputLayer().size(), 1, 1, 1), null);
			feature.setData(fullNetwork.getOutputLayer().getOutput());
			return feature;
		}
	}


	/**
	 * Getting feature fit to neuron channel.
	 * @return feature fit to neuron channel.
	 */
	public ConvLayerSingle getFeatureFitChannel() {
		ConvLayerSingle content = null;
		try {
			content = getFeature();
		} catch (Throwable e) {Util.trace(e);}
		if (content == null || content.length() == 0 || content.getNeuronChannel() == this.neuronChannel) return content;
		
		NeuronValue[] data = new NeuronValue[content.length()];
		for (int i = 0; i < data.length; i++) {
			data[i] = content.get(i).getValue().resize(this.neuronChannel);
		}
		ConvLayerSingle newContent = newLayer(new Size(content.getWidth(), content.getHeight(), content.getDepth(), content.getTime()), content.getFilter());
		newContent.setData(data);
		return newContent;
	}

	
	/**
	 * Get size of feature.
	 * @return feature size.
	 */
	public Size getFeatureSize() {
		if (fullNetwork == null) {
			ConvLayerSingle feature = null;
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
		NeuronValue[] data = null;
		if (fullNetwork != null) {
			try {
				if (reversedFullNetwork != null) {
					data = reversedFullNetwork.evaluate(new Record(feature));
					data = convertFullNetworkInputToUnifiedContent(data).getData();
				}
				else {
					//In the case that the full network is not null but the reserved full network is null,
					//then the feature is considered as input of the full network and output of the full network may be classes in classification learning. 
					data = convertFullNetworkInputToUnifiedContent(feature).getData();;
				}
			} catch (Throwable e) {Util.trace(e);}
		}
		else {
			try {
				//In the case that the full network is null, the output of the reversed full network is considered is the first feature and
				//the output of the convolutional network (the entire network) is the final feature. note, the output of the reversed full network is input of the convolutional network.
				data = reversedFullNetwork != null ? reversedFullNetwork.evaluate(new Record(feature)) : feature;
				data = evaluate(convertReversedFullNetworkOutputToConvInput(data));
			} catch (Throwable e) {Util.trace(e);}
		}
		
		ConvLayerSingle content = getUnifiedOutputContent(false);
		return content != null ? content.createRaster(data, isNorm(), getDefaultAlpha()) : null;
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
	public NeuronValue[] learnOneByOne(Iterable<Record> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = paramGetLearningRate();
		return learnOneByOne(sample, learningRate, terminatedThreshold, maxIteration);
	}

	
	@Override
	public NeuronValue[] learn(Iterable<Record> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = paramGetLearningRate();
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
	public NeuronValue[] learnOneByOne(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		if (convLayers.size() < 1) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			Iterable<Record> subsample = resample(sample, iteration, maxIteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration+1);

			for (Record record : subsample) {
				if (record == null) continue;
				
				//Evaluating layers.
				try {
					evaluate(record);
				} catch (Throwable e) {Util.trace(e);}
				
				//If there is in only forwarding mode then continue.
				if (paramIsOnlyForward()) continue;

				ConvLayerSingle unifiedContent = getUnifiedOutputContent(false);
				
				//Main learning to learn RCN.
				if (fullNetwork != null) {
					error = fullNetwork.learn(convertUnifiedContentToFullNetworkInput(unifiedContent), record.output, lr, terminatedThreshold, 1);
				}
				
				//Learning RFCN.
				if (reversedFullNetwork != null) {
					NeuronValue[] rfnOutput = null;
					if (fullNetwork != null)
						rfnOutput = convertUnifiedContentToFullNetworkInput(unifiedContent);
					else if (record != null && record.input != null)
						rfnOutput = record.input[0].flatten(reversedFullNetwork.getNeuronChannel()); 
					reversedFullNetwork.learn(record.output, rfnOutput, lr, terminatedThreshold, 1);
				}

				//Be careful to learn filter because filter is only learned from large layer to small layer in this current version.
				if (config.getAsBoolean(LEARN_FILTERS_EXT_FIELD)) learnFiltersExt(lr, 1);
				
			} //End for
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "convnn_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else if (terminatedThreshold > 0 && config.getAsBoolean(LEARN_TERMINATE_ERROR_FIELD)) {
				double errorMean = NeuronValue.normMean(error);
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
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "convnn_backpropogate",
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
		
		if (convLayers.size() < 1) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			Iterable<Record> subsample = resample(sample, iteration, maxIteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration+1);

			List<Record> fnSample = Util.newList(0), rfnSample = Util.newList(0);
			for (Record record : subsample) {
				NeuronValue[] fnInput = null, rfnOutput = null;
				try {
					//Evaluating layers.
					evaluate(record);
					ConvLayerSingle unifiedContent = getUnifiedOutputContent(false);
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
			
			if (!paramIsOnlyForward()) {
				//Main learning to learn RCN.
				if (fullNetwork != null) error = fullNetwork.learn(fnSample, lr, terminatedThreshold, 1);
				
				//Learning RFCN.
				if (reversedFullNetwork != null) reversedFullNetwork.learn(rfnSample, lr, terminatedThreshold, 1);
				
				//Be careful to learn filter because filter is only learned from large layer to small layer in this current version.
				if (config.getAsBoolean(LEARN_FILTERS_EXT_FIELD)) learnFiltersExt(lr, 1);
			}
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "convnn_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else if (terminatedThreshold > 0 && config.getAsBoolean(LEARN_TERMINATE_ERROR_FIELD)) {
				double errorMean = NeuronValue.normMean(error);
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
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "convnn_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}

	
	/**
	 * Extensive learning filters.
	 * @param learningRate learning rate.
	 * @param maxIteration maximum iteration.
	 */
	void learnFiltersExt(double learningRate, int maxIteration) {
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		for (ConvLayerSingle convLayer : convLayers) {
			try {
				Filter filter = convLayer.getFilter();
				BiasFilter biasFilter = filter != null ? new BiasFilter(filter, convLayer.getBias()) : null;
				biasFilter = convLayer.learnFilter(biasFilter, true, learningRate, maxIteration);
				
				if (biasFilter != null && biasFilter.filter != null) convLayer.setFilter(biasFilter.filter);
				if (biasFilter != null && biasFilter.bias != null) convLayer.setBias(biasFilter.bias);
			} catch (Throwable e) {Util.trace(e);}
		}
	}

	
	/**
	 * Checking whether point values are normalized in rang [0, 1].
	 * @return whether point values are normalized in rang [0, 1].
	 */
	private boolean isNorm() {
		if (config.containsKey(Raster.NORM_FIELD))
			return config.getAsBoolean(Raster.NORM_FIELD);
		else
			return Raster.NORM_DEFAULT;
	}
	
	
	/**
	 * Getting default alpha.
	 * @return default alpha.
	 */
	private int getDefaultAlpha() {
		if (config.containsKey(Image.ALPHA_FIELD))
			return config.getAsInt(Image.ALPHA_FIELD);
		else
			return Image.ALPHA_DEFAULT;
	}

	
	/**
	 * Checking only forwarding mode.
	 * @return only forwarding mode.
	 */
	boolean paramIsOnlyForward() {
		if (config.containsKey(ONLY_FORWARD_FIELD))
			return config.getAsBoolean(ONLY_FORWARD_FIELD);
		else
			return ONLY_FORWARD_DEFAULT;
	}
	
	
	/**
	 * Setting only forwarding mode.
	 * @param onlyForward only forwarding mode.
	 * @return this network.
	 */
	ConvNetworkAbstract paramSetOnlyForward(boolean onlyForward) {
		config.put(ONLY_FORWARD_FIELD, onlyForward);
		return this;
	}

	
	/**
	 * Checking learning filters mode.
	 * @return learning filters mode.
	 */
	boolean paramIsLearnFilters() {
		if (config.containsKey(LEARN_FILTERS_FIELD))
			return config.getAsBoolean(LEARN_FILTERS_FIELD);
		else
			return LEARN_FILTERS_DEFAULT;
	}
	
	
	/**
	 * Setting learning filters mode.
	 * @param learnFilters learning filters mode.
	 * @return this network.
	 */
	ConvNetworkAbstract paramSetLearnFilters(boolean learnFilters) {
		config.put(LEARN_FILTERS_FIELD, learnFilters);
		return this;
	}

	
	/**
	 * Checking learning weights mode.
	 * @return learning weights mode.
	 */
	boolean paramIsLearnWeights() {
		if (config.containsKey(LEARN_WEIGHTS_FIELD))
			return config.getAsBoolean(LEARN_WEIGHTS_FIELD);
		else
			return LEARN_WEIGHTS_DEFAULT;
	}
	
	
	/**
	 * Setting learning weights mode.
	 * @param learnWeights learning weights mode.
	 * @return this network.
	 */
	ConvNetworkAbstract paramSetLearnWeights(boolean learnWeights) {
		config.put(LEARN_WEIGHTS_FIELD, learnWeights);
		return this;
	}

	
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
