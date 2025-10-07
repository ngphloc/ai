/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

import net.ea.ann.core.value.NeuronValue;

/**
 * This class is abstract implementation of standard neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class NetworkStandardAbstract extends NetworkAbstract implements NetworkStandard, TextParsable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Input layer.
	 */
	protected LayerStandard inputLayer = null;

	
	/**
	 * Memory layer.
	 */
	protected List<LayerStandard> hiddenLayers = Util.newList(0);

	
	/**
	 * Output layer.
	 */
	protected LayerStandard outputLayer = null;
	

	/**
	 * Memory layer.
	 */
	protected LayerStandard memoryLayer = null;
	
	
	/**
	 * Constructor with ID reference.
	 * @param idRef ID reference.
	 */
	protected NetworkStandardAbstract(Id idRef) {
		super(idRef);
	}

	
	/**
	 * Default constructor.
	 */
	protected NetworkStandardAbstract() {
		this(null);
	}
	
	
	/**
	 * Initialize with number of neurons.
	 * @param nInputNeuron number of input neurons.
	 * @param nOutputNeuron number of output neurons.
	 * @param nHiddenNeuron number of hidden neurons as well as number of hidden layers.
	 * @param nMemoryNeuron number of memory neurons.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int nInputNeuron, int nOutputNeuron, int[] nHiddenNeuron, int nMemoryNeuron) {
		nInputNeuron = nInputNeuron < 1 ? 1 : nInputNeuron;
		nOutputNeuron = nOutputNeuron < 1 ? 1 : nOutputNeuron;
		nMemoryNeuron = nMemoryNeuron < 0 ? 0 : nMemoryNeuron;
		
		this.inputLayer = newLayer(nInputNeuron, null, null);
		
		if (this.hiddenLayers != null) this.hiddenLayers.clear();
		if (nHiddenNeuron != null && nHiddenNeuron.length > 0) {
			this.hiddenLayers = Util.newList(nHiddenNeuron.length);
			for (int i = 0; i < nHiddenNeuron.length; i++) {
				LayerStandard prevHiddenLayer = i == 0 ? this.inputLayer : this.hiddenLayers.get(i - 1);
				LayerStandard hiddenLayer = newLayer(nHiddenNeuron[i] < 1 ? 1 : nHiddenNeuron[i], prevHiddenLayer, null);
				this.hiddenLayers.add(hiddenLayer);
			}
		}
		
		LayerStandard preOutputLayer = this.hiddenLayers.size() > 0 ? this.hiddenLayers.get(this.hiddenLayers.size() - 1) : this.inputLayer;
		this.outputLayer = newLayer(nOutputNeuron, preOutputLayer, null);
		
		this.memoryLayer = null;
		if (nMemoryNeuron > 0 && nHiddenNeuron != null && nHiddenNeuron.length > 0) {
			this.memoryLayer = newLayer(nMemoryNeuron, null, null);
			this.outputLayer.setRiboutLayer(this.memoryLayer); //this.outputLayer.setRiboutLayer(this.hiddenLayers.get(this.hiddenLayers.size() - 1))
			this.hiddenLayers.get(0).setRibinLayer(this.memoryLayer);
		}
		
		return true;
	}

	
	/**
	 * Initialize with number of neurons.
	 * @param nInputNeuron number of input neurons.
	 * @param nOutputNeuron number of output neurons.
	 * @param nHiddenNeuron number of hidden neurons.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int nInputNeuron, int nOutputNeuron, int[] nHiddenNeuron) {
		return initialize(nInputNeuron, nOutputNeuron, nHiddenNeuron, 0);
	}
	
	
	/**
	 * Initialize with number of neurons.
	 * @param nInputNeuron number of input neurons.
	 * @param nOutputNeuron number of output neurons.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int nInputNeuron, int nOutputNeuron) {
		return initialize(nInputNeuron, nOutputNeuron, null, 0);
	}

	
	/**
	 * Reseting network.
	 */
	public void reset() {
		inputLayer = null;
		hiddenLayers.clear();
		outputLayer = null;
		memoryLayer = null;
	}
	
	
	/**
	 * Create layer.
	 * @return created layer.
	 */
	protected abstract LayerStandard newLayer();
	
	
	/**
	 * Creating new layer. This method can be called from derived classes.
	 * @param nNeuron number of neurons.
	 * @param prevLayer previous layer.
	 * @param nextLayer next layer.
	 * @return new layer.
	 */
	private LayerStandard newLayer(int nNeuron, LayerStandard prevLayer, LayerStandard nextLayer) {
		LayerStandard layer = newLayer();
		nNeuron = nNeuron < 0 ? 0 : nNeuron;
		for (int i = 0; i < nNeuron; i++) {
			layer.add(layer.newNeuron());
		}
		
		if (prevLayer != null) prevLayer.setNextLayer(layer);
		if (nextLayer != null) layer.setNextLayer(nextLayer);

		return layer;
	}

	
	/**
	 * Getting type of specified layer.
	 * @param layer specified layer.
	 * @return type of specified layer.
	 */
	public LayerType typeOf(LayerStandard layer) {
		if (layer == null) return LayerType.unknown;
		
		if (inputLayer != null && layer == inputLayer) return LayerType.input;
		if (outputLayer != null && layer == outputLayer) return LayerType.output;
		for (LayerStandard hiddenLayer : hiddenLayers) {
			if (layer == hiddenLayer) return LayerType.hidden;
		}
		if (memoryLayer != null && layer == memoryLayer) return LayerType.memory;
		
		List<LayerStandard> backbone = getBackbone();
		for (LayerStandard l : backbone) {
			List<LayerStandard> ribin = getRibinbone(l);
			if (findLayer(ribin, layer) >= 0) return LayerType.ribin;
			List<LayerStandard> ribout = getRiboutbone(l);
			if (findLayer(ribout, layer) >= 0) return LayerType.ribout;
		}

		return LayerType.unknown;
	}
	
	
	/**
	 * Getting input layer.
	 * @return input layer.
	 */
	public LayerStandard getInputLayer() {
		return inputLayer;
	}

	
	/**
	 * Getting hidden layers.
	 * @return array of hidden layers.
	 */
	public LayerStandard[] getHiddenLayers() {
		return hiddenLayers.toArray(new LayerStandard[] {});
	}

	
	/**
	 * Getting index of hidden layer.
	 * @param layer hidden layer.
	 * @return index of hidden layer.
	 */
	public int hiddenIndexOf(LayerStandard layer) {
		if (layer == null) return -1;
		
		for (int i = 0; i < hiddenLayers.size(); i++) {
			LayerStandard hiddenLayer = hiddenLayers.get(i);
			if (layer == hiddenLayer) return i;
		}
		
		return -1;
	}
	
	
	/**
	 * Getting output layer.
	 * @return output layer.
	 */
	public LayerStandard getOutputLayer() {
		return outputLayer;
	}

	
	/**
	 * Getting memory layer.
	 * @return memory layer.
	 */
	public LayerStandard getMemoryLayer() {
		return memoryLayer;
	}

	
	/**
	 * Getting backbone which is chain of main layers.
	 * @return backbone which is chain of main layers.
	 */
	public List<LayerStandard> getBackbone() {
		List<LayerStandard> backbone = Util.newList(2);
		if (inputLayer == null || outputLayer == null)
			return backbone;
		
		backbone.add(inputLayer);
		if (hiddenLayers.size() > 0) backbone.addAll(hiddenLayers);
		backbone.add(outputLayer);
		
		return backbone;
	}
	
	
	/**
	 * Getting list of input rib bones.
	 * @return list of input rib bones.
	 */
	public List<List<LayerStandard>> getRibinbones() {
		List<List<LayerStandard>> ribbones = Util.newList(0);
		
		List<LayerStandard> backbone = getBackbone();
		for (LayerStandard layer : backbone) {
			List<LayerStandard> ribbone = getRibinbone(layer);
			if (ribbone.size() > 0) ribbones.add(ribbone);
		}
		
		return ribbones;
	}
	
	
	/**
	 * Getting input rib bone of specified layer.
	 * @param layer specified layer.
	 * @return input rib bone of specified layer.
	 */
	private List<LayerStandard> getRibinbone(LayerStandard layer) {
		List<LayerStandard> ribbone = Util.newList(0);
		if (layer == null) return ribbone;
		LayerStandard ribLayer = layer.getRibinLayer();
		if (ribLayer == null || ribLayer == memoryLayer) return ribbone;
		List<LayerStandard> backbone = getBackbone();
		
		ribbone.add(0, layer);
		while (ribLayer != null) {
			ribbone.add(0, ribLayer);
			
			if (backbone.contains(ribLayer)) //rib-in layer turns back to its network backbone.
				break;
			else if (ribLayer instanceof LayerStandardAbstract) {
				NetworkStandardAbstract otherNetwork = ((LayerStandardAbstract)ribLayer).getNetwork();
				if (otherNetwork != null && otherNetwork != this) {
					List<LayerStandard> otherBackbone = getBackbone();
					if (otherBackbone.contains(ribLayer)) break; //rib-in layer comes into other network backbone.
				}
			}
			
			ribLayer = ribLayer.getPrevLayer();
			LayerStandard prevLayer = ribLayer.getPrevLayer();
			if (prevLayer == null || prevLayer.getRiboutLayer() == ribLayer) //Stop when rib-in layer meets rib-out layer.
				break;
			else
				ribLayer = prevLayer;
		}
		
		return ribbone;
	}
	
	
	/**
	 * Getting list of output rib bones.
	 * @return list of output rib bones.
	 */
	public List<List<LayerStandard>> getRiboutbones() {
		List<List<LayerStandard>> ribbones = Util.newList(0);
		
		List<LayerStandard> backbone = getBackbone();
		for (LayerStandard layer : backbone) {
			List<LayerStandard> ribbone = getRiboutbone(layer);
			if (ribbone.size() > 0) ribbones.add(ribbone);
		}
		
		return ribbones;
	}

	
	/**
	 * Getting output rib bone of specified layer.
	 * @param layer specified layer.
	 * @return output rib bone of specified layer.
	 */
	private List<LayerStandard> getRiboutbone(LayerStandard layer) {
		List<LayerStandard> ribbone = Util.newList(0);
		if (layer == null) return ribbone;
		LayerStandard ribLayer = layer.getRiboutLayer();
		if (ribLayer == null || ribLayer == memoryLayer) return ribbone;
		List<LayerStandard> backbone = getBackbone();
		
		ribbone.add(layer);
		while (ribLayer != null) {
			ribbone.add(ribLayer);
			
			if (backbone.contains(ribLayer)) //rib-out layer turns back to its network backbone.
				break;
			else if (ribLayer instanceof LayerStandardAbstract) {
				NetworkStandardAbstract otherNetwork = ((LayerStandardAbstract)ribLayer).getNetwork();
				if (otherNetwork != null && otherNetwork != this) {
					List<LayerStandard> otherBackbone = getBackbone();
					if (otherBackbone.contains(ribLayer)) break; //rib-out layer comes into other network backbone.
				}
			}
			
			LayerStandard nextLayer = ribLayer.getNextLayer();
			if (nextLayer == null || nextLayer.getRibinLayer() == ribLayer) //Stop when rib-in layer meets rib-out layer.
				break;
			else
				ribLayer = nextLayer;
		}
		
		return ribbone;
	}


	/**
	 * Getting list of short input rib bones.
	 * @return list of short input rib bones.
	 */
	public List<List<LayerStandard>> getShortRibinbones() {
		List<List<LayerStandard>> ribbones = Util.newList(0);
		
		List<LayerStandard> backbone = getBackbone();
		for (LayerStandard layer : backbone) {
			List<LayerStandard> ribbone = getShortRibinbone(layer);
			if (ribbone.size() > 0) ribbones.add(ribbone);
		}
		
		return ribbones;
	}

	
	/**
	 * Getting short input rib bone of specified layer.
	 * @param layer specified layer.
	 * @return short input rib bone of specified layer.
	 */
	private List<LayerStandard> getShortRibinbone(LayerStandard layer) {
		List<LayerStandard> ribbone = Util.newList(0);
		if (layer == null) return ribbone;
		LayerStandard ribLayer = layer.getRibinLayer();
		if (ribLayer == null || ribLayer == memoryLayer) return ribbone;
		
		ribbone.add(0, layer);
		ribbone.add(0, ribLayer); //Only one rib-in layer which is often a layer on backbone of other network.
		return ribbone;
	}

	
	/**
	 * Getting list of short output rib bones.
	 * @return list of short output rib bones.
	 */
	public List<List<LayerStandard>> getShortRiboutbones() {
		List<List<LayerStandard>> ribbones = Util.newList(0);
		
		List<LayerStandard> backbone = getBackbone();
		for (LayerStandard layer : backbone) {
			List<LayerStandard> ribbone = getShortRiboutbone(layer);
			if (ribbone.size() > 0) ribbones.add(ribbone);
		}
		
		return ribbones;
	}

	
	/**
	 * Getting short output rib bone of specified layer.
	 * @param layer specified layer.
	 * @return short output rib bone of specified layer.
	 */
	private List<LayerStandard> getShortRiboutbone(LayerStandard layer) {
		List<LayerStandard> ribbone = Util.newList(0);
		if (layer == null) return ribbone;
		LayerStandard ribLayer = layer.getRiboutLayer();
		if (ribLayer == null || ribLayer == memoryLayer) return ribbone;
		
		ribbone.add(layer);
		ribbone.add(ribLayer); //Only one rib-out layer which is often a layer on backbone of other network.
		return ribbone;
	}

	
	/**
	 * Getting all layers.
	 * @return all layers.
	 */
	public List<LayerStandard> getAllLayers() {
		Set<LayerStandard> all = Util.newSet(0);
		if (inputLayer != null) all.add(inputLayer);
		all.addAll(hiddenLayers);
		if (outputLayer != null) all.add(outputLayer);
		if (memoryLayer != null) all.add(memoryLayer);
		
		List<List<LayerStandard>> bones = getRibinbones();
		for (List<LayerStandard> bone : bones) all.addAll(bone);
		bones = getRiboutbones();
		for (List<LayerStandard> bone : bones) all.addAll(bone);

		List<LayerStandard> list = Util.newList(all.size());
		list.addAll(all);
		return list;
	}
	
	
	/**
	 * Getting non-empty layers.
	 * @return list of non-empty layers.
	 */
	private List<LayerStandard> getNonemptyLayers() {
		List<LayerStandard> all = getAllLayers();
		List<LayerStandard> nonempty = Util.newList(0);
		for (LayerStandard layer : all) {
			if (layer.size() > 0) nonempty.add(layer);
		}
		return nonempty;
	}

	
	/**
	 * Finding layer by specified identifier.
	 * @param layerId specified identifier.
	 * @return found layer.
	 */
	public LayerStandard findLayer(int layerId) {
		List<LayerStandard> all = getAllLayers();
		for (LayerStandard layer : all) {
			if (layer != null && layer.id() == layerId) return layer;
		}
		
		return null;
	}
	
	
	/**
	 * Finding specified layer in specified bone.
	 * @param bone specified bone.
	 * @param layer specified layer.
	 * @return index of specified layer in specified bone.
	 */
	private static int findLayer(List<LayerStandard> bone, LayerStandard layer) {
		if (layer == null || bone.size() == 0) return -1;
		for (int i = 0; i < bone.size(); i++) {
			if (bone.get(i) == layer) return i;
		}
		
		return -1;
	}
	

	/**
	 * Finding neuron by specified identifier.
	 * @param neuronId specified identifier.
	 * @return found neuron.
	 */
	public NeuronStandard findNeuron(int neuronId) {
		List<LayerStandard> layers = getNonemptyLayers();
		for (LayerStandard layer : layers) {
			int index = layer.indexOf(neuronId);
			if (index >= 0) return layer.get(index);
		}
		
		return null;
	}
	
	
	@Override
	public synchronized NeuronValue[] evaluate(Record inputRecord) throws RemoteException {
		return evaluate(inputRecord, true);
	}


	/**
	 * Evaluating entire network.
	 * @param inputRecord input record.
	 * @param resetMemory resetting memory flag.
	 * @return array as output of output layer.
	 */
	private NeuronValue[] evaluate(Record inputRecord, boolean resetMemory) {
		if (inputRecord == null) return null;
		List<LayerStandard> backbone = getBackbone();
		if (backbone.size() == 0) return null;
		
		if (memoryLayer != null && resetMemory) {
			NeuronValue zero = memoryLayer.newNeuronValue().zero();
			for (int j = 0; j < memoryLayer.size(); j++) {
				NeuronStandard neuron = memoryLayer.get(j);
				neuron.setInput(zero); neuron.setOutput(zero);
			}
		}

		boolean updateMemory = false;
		for (int i = 0; i < backbone.size(); i++) {
			LayerStandard layer = backbone.get(i);
			List<LayerStandard> ribinbone = getRibinbone(layer);
			if (ribinbone != null && ribinbone.size() > 1 && inputRecord.ribinInput != null) {
				int index = backbone.indexOf(ribinbone.get(ribinbone.size()-1));
				if (inputRecord.ribinInput.containsKey(index)) {
					evaluate(ribinbone, inputRecord.ribinInput.get(index));
					updateMemory = true;
				}
			}
			
			if (inputRecord.input != null || backbone.get(0).hasSomePrevLayers()) {
				backbone.get(i).evaluate(inputRecord.input);
				
				List<LayerStandard> riboutbone = getRiboutbone(layer);
				for (int j = 1; j < riboutbone.size(); j++) riboutbone.get(j).evaluate();
				updateMemory = true;
			}
		}
		
		if (memoryLayer != null && updateMemory) {
			for (int j = 0; j < memoryLayer.size(); j++) memoryLayer.get(j).evaluate();
		}
		
		return backbone.get(backbone.size() - 1).getOutput();
	}

	
	/**
	 * Evaluating bone with specified input.
	 * @param bone list of layers including input layer.
	 * @param input specified input.
	 * @return evaluated output.
	 */
	static NeuronValue[] evaluate(List<LayerStandard> bone, NeuronValue[] input) {
		if (bone.size() == 0) return null;
		for (int i = 0; i < bone.size(); i++) bone.get(i).evaluate(input);
		return bone.get(bone.size()-1).getOutput();
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
	 * Learning neural network by back propagate algorithm, one-by-one record over sample.
	 * @param sample learning sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	protected abstract NeuronValue[] learnOneByOne(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration);
	
	
	/**
	 * Learning neural network by back propagate algorithm.
	 * @param sample learning sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	protected abstract NeuronValue[] learn(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration);

	
	/**
	 * Calculate error of output neuron. Derived classes should implement this method.
	 * This error is the opposite of gradient of minimized target function and the gradient of maximized target function.  
	 * The real output can be null in some cases because the error may not be calculated by squared error function that needs real output.  
	 * @param outputNeuron output neuron. Output value of this neuron is retrieved by method {@link NeuronStandard#getOutput()}.
	 * @param realOutput real output. It can be null because this method is flexible.
	 * @param outputLayer output layer. It can be null because this method is flexible.
	 * @param outputNeuronIndex index of output neuron. It can be -1 because this method is flexible. This is optional parameter.
	 * @param realOutputs real outputs. It can be null because this method is flexible. This is optional parameter. 
	 * @param params option parameter array which can be null. 
	 * @return error or loss of the output neuron.
	 */
	protected abstract NeuronValue calcOutputError(NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer, int outputNeuronIndex, NeuronValue[] realOutputs, Object...params);

		
	/**
	 * Verbalize a list of layers.
	 * @param layers list of layers.
	 * @param tab tab text.
	 * @return verbalized text.
	 */
	private static String toText(List<LayerStandard> layers, String tab) {
		StringBuffer buffer = new StringBuffer();
		for (int i = 0; i < layers.size(); i++) {
			if (i > 0) buffer.append("\n");

			String layerText = LayerStandardImpl.toText(layers.get(i), null);
			layerText = layerText.replaceAll("l##", "" + (i+1));
			buffer.append(layerText);
		}
		
		String text = buffer.toString();
		if (tab != null && !tab.isEmpty()) {
			text = tab + text; text = text.replaceAll("\n", "\n" + tab);
		}
		return text;
	}
	
	
	/**
	 * Verbalize network.
	 * @param network specific network.
	 * @param tab tab text.
	 * @return verbalized text.
	 */
	protected static String toText(NetworkStandardAbstract network, String tab) {
		StringBuffer buffer = new StringBuffer();
		String internalTab = "    ";
		
		List<LayerStandard> backbone = network.getBackbone();
		if (backbone.size() > 0) {
			buffer.append("BACKBONE:\n");
			buffer.append(toText(backbone, internalTab));
		}
		
		LayerStandard memory = network.getMemoryLayer();
		if (memory != null) {
			buffer.append("MEMORY:\n");
			buffer.append(toText(Arrays.asList(memory), internalTab));
		}
		
		List<List<LayerStandard>> ribinBones = network.getRibinbones();
		for (List<LayerStandard> ribinBone : ribinBones) {
			if (ribinBone.size() > 0) {
				buffer.append("RIBIN BONE:\n");
				buffer.append(toText(ribinBone, internalTab));
			}
		}
		
		List<List<LayerStandard>> riboutBones = network.getRiboutbones();
		for (List<LayerStandard> riboutBone : riboutBones) {
			if (riboutBone.size() > 0) {
				buffer.append("RIBOUT BONE:\n");
				buffer.append(toText(riboutBone, internalTab));
			}
		}

		return buffer.toString();
	}


	@Override
	public String toText() {
		try {
			return toText(this, null);
		}
		catch (Throwable e) {}
		
		return super.toString();
	}

	
	@Override
	public void close() throws Exception {
		super.close();
		reset();
	}


}


