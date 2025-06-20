/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.classifier;

import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

import net.ea.ann.conv.Content;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.conv.filter.FilterAssoc;
import net.ea.ann.conv.stack.StackNetworkImpl;
import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.NetworkStandard;
import net.ea.ann.core.NetworkStandardImpl;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.generator.GeneratorStandard;
import net.ea.ann.core.generator.GeneratorWeighted;
import net.ea.ann.core.generator.Trainer;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.RasterProperty;
import net.ea.ann.raster.RasterProperty.Label;
import net.ea.ann.raster.RasterWrapperProperty;
import net.ea.ann.raster.Size;
import net.ea.ann.raster.SizeZoom;

/**
 * This class is default implementation of classifier.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ClassifierImpl extends StackNetworkImpl implements Classifier {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
//	/**
//	 * Neuron channel of classifier nut.
//	 */
//	private static final int FULL_NETWORK_NEURON_CHANNEL_DEFAULT = 1;
	
	
	/**
	 * Field of the number elements of a combination.
	 */
	private static final String COMB_NUMBER_FIELD = GeneratorWeighted.COMB_NUMBER_FIELD;
	
	
	/**
	 * Default value for the field of the number elements of a combination.
	 */
	private static final int COMB_NUMBER_DEFAULT = GeneratorWeighted.COMB_NUMBER_DEFAULT;

	
	/**
	 * Name of zoom-out field.
	 */
	private final static String ZOOMOUT_FIELD = "classifier_zoomout";

	
	/**
	 * Default value of zoom-out field.
	 */
	public final static int ZOOMOUT_DEFAULT = NetworkAbstract.ZOOMOUT_DEFAULT;

	
	/**
	 * Name of getting feature field.
	 */
	private final static String GET_FEATURE_FIELD = "classifier_get_feature";

	
	/**
	 * Default value of getting field.
	 */
	private final static boolean GET_FEATURE_DEFAULT = false;

	
	/**
	 * Name of simplest field.
	 */
	private final static String SIMPLEST_FIELD = "classifier_simplest";

	
	/**
	 * Default value of simplest field.
	 */
	private final static boolean SIMPLEST_DEFAULT = false;

	
	/**
	 * This class represents specific classification task.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public static class ClassifierNut extends GeneratorWeighted<Trainer> {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Constructor with neuron channel, activation function, and identifier reference.
		 * @param neuronChannel neuron channel.
		 * @param activateRef activation function.
		 * @param idRef identifier reference.
		 */
		public ClassifierNut(int neuronChannel, Function activateRef, Id idRef) {
			super(neuronChannel, activateRef, idRef);
		}

		/**
		 * Constructor with neuron channel and activation function.
		 * @param neuronChannel neuron channel.
		 * @param activateRef activation function.
		 */
		public ClassifierNut(int neuronChannel, Function activateRef) {
			this(neuronChannel, activateRef, null);
		}

		
		/**
		 * Constructor with neuron channel.
		 * @param neuronChannel neuron channel.
		 */
		public ClassifierNut(int neuronChannel) {
			this(neuronChannel, null, null);
		}

	}
	
	
	/**
	 * Map of classes.
	 */
	protected Map<Integer, Label> classMap = Util.newMap(0);

	
	/**
	 * Constructor with neuron channel, activation functions, and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function, which is often activation function related to weights like sigmod function.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 * @param idRef ID reference.
	 */
	protected ClassifierImpl(int neuronChannel, Function activateRef, Function contentActivateRef, Id idRef) {
		super(neuronChannel, activateRef, contentActivateRef, idRef);
//		this.fullNetworkNeuronChannel = FULL_NETWORK_NEURON_CHANNEL_DEFAULT;
		
		this.config.put(COMB_NUMBER_FIELD, COMB_NUMBER_DEFAULT);
		this.config.put(ZOOMOUT_FIELD, ZOOMOUT_DEFAULT);
		this.config.put(GET_FEATURE_FIELD, GET_FEATURE_DEFAULT);
		this.config.put(SIMPLEST_FIELD, SIMPLEST_DEFAULT);
		this.config.put(HIDDEN_LAYER_MIN_FILED, HIDDEN_LAYER_MIN_DEFAULT);
		
		GeneratorStandard.fillConfig(this.config);
	}

	
	/**
	 * Constructor with neuron channel and activation functions.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function, which is often activation function related to weights like sigmod function.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 */
	protected ClassifierImpl(int neuronChannel, Function activateRef, Function contentActivateRef) {
		this(neuronChannel, activateRef, contentActivateRef, null);
	}


	@Override
	public void reset() {
		super.reset();
		clearClassifierInfo();
	}


	/**
	 * Clearing classification information.
	 */
	private void clearClassifierInfo() {
		classMap.clear();
	}
	
	
	/**
	 * Creating full network.
	 * @return full network.
	 */
	private ClassifierNut createFullNetwork() {
		ClassifierNut nut = new ClassifierNut(fullNetworkNeuronChannel, activateRef, idRef);
		nut.setParent(this);
		try {
			nut.getConfig().putAll(config);
		} catch (Throwable e) {Util.trace(e);}
		nut.paramSetCombNumber(getCombNumber());
		return nut;
	}
	
	
	/**
	 * Getting full network.
	 * @return full network.
	 */
	private ClassifierNut getFullNetwork() {
		return (ClassifierNut)fullNetwork;
	}
	
	
	/**
	 * Creating reversed full network.
	 * @return reversed full network.
	 */
	private NetworkStandardImpl createReversedFullNetwork() {
		GeneratorStandard<Trainer> generator = new GeneratorStandard<Trainer>(fullNetworkNeuronChannel, activateRef, idRef);
		generator.setParent(this);
		try {
			generator.getConfig().putAll(config);
		} catch (Throwable e) {Util.trace(e);}
		return generator;
	}
	
	
	@Override
	protected boolean initializeFullNetwork(int[] nFullHiddenOutputNeuron, boolean initReverse) {
		if (nFullHiddenOutputNeuron == null || nFullHiddenOutputNeuron.length < 1) return false;
		
		Content content = unifyOutputContent();
		if (content == null) return false;

		fullNetwork = createFullNetwork();
		int nInputNeuron = getFullNetworkNeuronChannelRatio() * content.getWidth() * content.getHeight() * content.getDepth() * content.getTime();
		if (nFullHiddenOutputNeuron.length == 1) {
			if (!fullNetwork.initialize(nInputNeuron, nFullHiddenOutputNeuron[0])) return false;
		}
		else {
			int length = nFullHiddenOutputNeuron.length;
			int[] nHiddenNeuron = Arrays.copyOf(nFullHiddenOutputNeuron, length-1);
			if (!fullNetwork.initialize(nInputNeuron, nFullHiddenOutputNeuron[length-1], nHiddenNeuron)) return false;
		}
		try {
			fullNetwork.addListener(this);
		} catch (RemoteException e) {Util.trace(e);}
		
		if (!initReverse) return true;
		
		int rInputNeuron = fullNetwork.getOutputLayer().size();
		int rOutputNeuron = fullNetwork.getInputLayer().size();
		if (rInputNeuron == 0 || rOutputNeuron == 0) return false;
		int[] rHiddenNeuron = null;
		LayerStandard[] hiddenLayers = fullNetwork.getHiddenLayers();
		if (hiddenLayers != null && hiddenLayers.length > 0) {
			rHiddenNeuron = new int[hiddenLayers.length];
			for (int i = 0; i < hiddenLayers.length; i++) rHiddenNeuron[i] = hiddenLayers[hiddenLayers.length-1-i].size();
		}
		reversedFullNetwork = createReversedFullNetwork();
		if (!reversedFullNetwork.initialize(rInputNeuron, rOutputNeuron, rHiddenNeuron)) return false;
		try {
			reversedFullNetwork.addListener(this);
		} catch (Throwable e) {Util.trace(e);}
		
		return true;
	}

	
	@Override
	public synchronized NeuronValue[] evaluate(NeuronValue[]... input) {
		return super.evaluate(input);
	}


	@Override
	public NeuronValue[] learnOne(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		List<Raster> rasters = RasterAssoc.toInputRasters(sample);
		if (rasters.size() == 0) return super.learnOne(sample, learningRate, terminatedThreshold, maxIteration);
		List<Record> newSample = prelearn(rasters);
		if (newSample.size() == 0)
			return super.learnOne(sample, learningRate, terminatedThreshold, maxIteration);
		else
			return super.learnOne(newSample, learningRate, terminatedThreshold, maxIteration);
	}


	@Override
	public NeuronValue[] learn(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		List<Raster> rasters = RasterAssoc.toInputRasters(sample);
		if (rasters.size() == 0) return super.learn(sample, learningRate, terminatedThreshold, maxIteration);
		List<Record> newSample = prelearn(rasters);
		if (newSample.size() == 0)
			return super.learn(sample, learningRate, terminatedThreshold, maxIteration);
		else
			return super.learn(newSample, learningRate, terminatedThreshold, maxIteration);
	}


	@Override
	public NeuronValue[] learnRasterOne(Iterable<Raster> sample) throws RemoteException {
		return learnOne(RasterAssoc.toInputSample(sample));
	}


	@Override
	public NeuronValue[] learnRaster(Iterable<Raster> sample) throws RemoteException {
		return learn(RasterAssoc.toInputSample(sample));
	}


	/**
	 * Pre-processing for learning.
	 * @param sample
	 * @return new sample.
	 */
	private List<Record> prelearn(Iterable<Raster> sample) {
		clearClassifierInfo();
		
		List<Label> labels = Util.newList(0);
		List<Raster> train = Util.newList(0);
		for (Raster raster : sample) {
			if (raster == null) continue;
			RasterProperty rp = raster.getProperty();
			int labelId = rp.getLabelId();
			if (labelId < 0) continue;
			
			train.add(raster);
			boolean found = false;
			for (Label label : labels) {
				if (label.labelId == labelId) {
					found = true;
					break;
				}
			}
			if (!found) labels.add(new Label(labelId, rp.getLabelName()));
		}
		if (labels.size() == 0 || train.size() == 0) return Util.newList(0);
		
		Label.sort(labels, true);
		for (int classNumber = 0; classNumber < labels.size(); classNumber++) {
			classMap.put(classNumber, labels.get(classNumber));
		}

		Size size = RasterAssoc.getAverageSize(train);
		Filter[][] filterArrays = getFilterArrays(size, getDim(train));
		if (!initialize(size, filterArrays)) {
			reset();
			return Util.newList(0);
		}
		
		Content content = unifyOutputContent();
		if (content == null) return Util.newList(0);
		int[] nHiddenOutput = null;
		if (isSimplest())
			nHiddenOutput = new int[] {labels.size()};
		else {
			int nInput = getFullNetworkNeuronChannelRatio() * content.getWidth() * content.getHeight() * content.getDepth() * content.getTime();
			int[] nHidden = NetworkStandard.constructHiddenNeuronNumbers(nInput, labels.size(), getHiddenLayerMin());
			nHiddenOutput = Arrays.copyOf(nHidden, nHidden.length + 1);
		}
		nHiddenOutput[nHiddenOutput.length-1] = labels.size();
		if (!initializeFullNetwork(nHiddenOutput, false)) return Util.newList(0);

		List<Record> newsample = Util.newList(0);
		for (Raster raster : train) {
			int label = raster.getProperty().getLabelId();
			int classIndex = classOf(label);
			if (classIndex < 0) continue;
			
			NeuronValue[] output = getFullNetwork().createOutputByClass(classIndex);
			if (output == null) continue;
			Record record = new Record(raster);
			record.output = output;
			newsample.add(record);
		}
		return newsample;
	}
	
	
	@Override
	public List<Raster> classify(Iterable<Raster> sample) throws RemoteException {
		List<Raster> results = Util.newList(0);
		for (Raster raster : sample) {
			if (raster == null) continue;
			try {
				evaluateRaster(raster);
			} catch (Throwable e) {Util.trace(e);}
			
			int maxClass = getFullNetwork().extractClass();
			if (maxClass < 0) continue;
			Label label = labelOf(maxClass);
			if (label == null) continue;
			
			RasterProperty rp = raster.getProperty().shallowDuplicate();
			rp.setLabel(new Label(label));
			RasterWrapperProperty rw = new RasterWrapperProperty(raster);
			rw.setProperty(rp);
			results.add(rw);
		}
		return results;
	}


	/**
	 * Getting class number of label.
	 * @param label specified label.
	 * @return class number of label.
	 */
	private int classOf(int label) {
		Set<Integer> classNumbers = classMap.keySet();
		for (int classNumber : classNumbers) {
			Label labelObject = classMap.get(classNumber);
			if (labelObject.labelId == label) return classNumber;
		}
		return -1;
	}
	
	
	/**
	 * Getting label of class number.
	 * @param classNumber class number.
	 * @return label of class number.
	 */
	private Label labelOf(int classNumber) {
		return classMap.containsKey(classNumber) ? classMap.get(classNumber) : null;
	}


	/**
	 * Getting filter arrays.
	 * @param size size of content.
	 * @param dim dimension.
	 * @return filter arrays.
	 */
	protected Filter[][] getFilterArrays(Size size, int dim) {
		int zoomOutOne = getZoomOut();
		List<Filter[]> filterArrays = Util.newList(0);
		if (zoomOutOne > 1) {
			SizeZoom zoomOut = SizeZoom.zoom(zoomOutOne, dim>1?zoomOutOne:1, dim>2?zoomOutOne:1, 1);
			SizeZoom sizeZ = RasterAssoc.calcFitSize(
				new SizeZoom(size.width, size.height, size.depth, size.time, zoomOut.widthZoom, zoomOut.heightZoom, zoomOut.depthZoom, zoomOut.timeZoom),
				Size.unit());
			Filter[] filters = Filter.calcZoomFilters(sizeZ, getFilterFactory(), true);
			filterArrays.add(filters);
		}
		
		if (dim == 2 && isGetFeature()) {
			Filter[][] featureArrays = isNorm() ? FilterAssoc.createNormFeatureExtractor2D(newStack(Size.unit())) : FilterAssoc.createFeatureExtractor2D(newStack(Size.unit()));
			for (Filter[] filters : featureArrays) filterArrays.add(filters);
		}
		
		return filterArrays.size() > 0 ? filterArrays.toArray(new Filter[][] {}) : null;
	}
	
	
	/**
	 * Getting dimension of sample.
	 * @param sample specified sample.
	 * @return dimension of sample.
	 */
	private static int getDim(Iterable<Raster> sample) {
		for (Raster raster : sample) {
			if (raster != null) return new RasterAssoc(raster).getDim();
		}
		return 0;
	}

	
	/**
	 * Getting zooming out ratio.
	 * @return zooming out ratio.
	 */
	private int getZoomOut() {
		int zoomOut = ZOOMOUT_DEFAULT;
		if (config.containsKey(ZOOMOUT_FIELD)) zoomOut = config.getAsInt(ZOOMOUT_FIELD);
		return zoomOut < 1 ? ZOOMOUT_DEFAULT : zoomOut;
	}

	
	/**
	 * Getting the number elements of a combination.
	 * @return the number elements of a combination.
	 */
	private int getCombNumber() {
		int combNumber = config.getAsInt(COMB_NUMBER_FIELD);
		return combNumber < 1 ? COMB_NUMBER_DEFAULT : combNumber;
	}

	
	/**
	 * Checking whether to get feature.
	 * @return whether to get feature..
	 */
	private boolean isGetFeature() {
		return config.getAsBoolean(GET_FEATURE_FIELD);
	}

	
	/**
	 * Checking whether this classifier is simplest (only 2 layers).
	 * @return whether this classifier is simplest (only 2 layers).
	 */
	private boolean isSimplest() {
		return config.getAsBoolean(SIMPLEST_FIELD);
	}

	
	/**
	 * Getting minimum number of hidden layers.
	 * @return minimum number of hidden layers.
	 */
	private int getHiddenLayerMin() {
		int hiddenMin = config.getAsInt(HIDDEN_LAYER_MIN_FILED);
		return hiddenMin < HIDDEN_LAYER_MIN_DEFAULT ? HIDDEN_LAYER_MIN_DEFAULT : hiddenMin;
	}

	
	/**
	 * Creating classifier with neuron channel, activation functions, and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function, which is often activation function related to weights like sigmod function.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 * @param idRef ID reference.
	 * @return classifier.
	 */
	public static ClassifierImpl create(int neuronChannel, Function activateRef, Function contentActivateRef, Id idRef) {
		return new ClassifierImpl(neuronChannel, activateRef, contentActivateRef, idRef);
	}


	/**
	 * Creating classifier with neuron channel and activation functions.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function, which is often activation function related to weights like sigmod function.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 * @return classifier.
	 */
	public static ClassifierImpl create(int neuronChannel, Function activateRef, Function contentActivateRef) {
		return create(neuronChannel, activateRef, contentActivateRef, null);
	}


	/**
	 * Creating classifier with neuron channel and norm flag.
	 * @param neuronChannel specified neuron channel.
	 * @param isNorm norm flag.
	 * @return classifier.
	 */
	public static ClassifierImpl create(int neuronChannel, boolean isNorm) {
		Function activateRef = Raster.toActivationRef(neuronChannel, isNorm);
		Function contentActivateRef = Raster.toConvActivationRef(neuronChannel, isNorm);
		return create(neuronChannel, activateRef, contentActivateRef, null);
	}
	
	
}
