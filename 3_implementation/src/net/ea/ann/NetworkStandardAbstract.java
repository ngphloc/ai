/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann;

import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;

/**
 * This class is abstract implementation of standard neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class NetworkStandardAbstract extends NetworkAbstract implements NetworkStandard {


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
	 */
	public NetworkStandardAbstract(Id idRef) {
		super(idRef);
	}

	
	/**
	 * Default constructor.
	 */
	public NetworkStandardAbstract() {
		this(null);
	}
	
	
	/**
	 * Initialize with number of neurons.
	 * @param nInputNeuron number of input neurons.
	 * @param nOutputNeuron number of output neurons.
	 * @param nHiddenNeuron number of hidden neurons as well as number of hidden layers.
	 * @param nMemoryNeuron number of memory neurons.
	 */
	public void initialize(int nInputNeuron, int nOutputNeuron, int[] nHiddenNeuron, int nMemoryNeuron) {
		nInputNeuron = nInputNeuron < 1 ? 1 : nInputNeuron;
		nOutputNeuron = nOutputNeuron < 1 ? 1 : nOutputNeuron;
		nMemoryNeuron = nMemoryNeuron < 0 ? 0 : nMemoryNeuron;
		
		this.inputLayer = newLayer(nInputNeuron, null, null);
		
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
		
		if (nMemoryNeuron > 0 && nHiddenNeuron != null && nHiddenNeuron.length > 0) {
			this.memoryLayer = newLayer(nMemoryNeuron, null, null);
			this.outputLayer.setRiboutLayer(this.memoryLayer); //this.outputLayer.setRiboutLayer(this.hiddenLayers.get(this.hiddenLayers.size() - 1))
			this.memoryLayer.setRibinLayer(this.hiddenLayers.get(0));
		}
	}

	
	/**
	 * Initialize with number of neurons.
	 * @param nInputNeuron number of input neurons.
	 * @param nOutputNeuron number of output neurons.
	 * @param nHiddenNeuron number of hidden neurons.
	 */
	public void initialize(int nInputNeuron, int nOutputNeuron, int[] nHiddenNeuron) {
		initialize(nInputNeuron, nOutputNeuron, nHiddenNeuron, 0);
	}
	
	
	/**
	 * Initialize with number of neurons.
	 * @param nInputNeuron number of input neurons.
	 * @param nOutputNeuron number of output neurons.
	 */
	public void initialize(int nInputNeuron, int nOutputNeuron) {
		initialize(nInputNeuron, nOutputNeuron, null, 0);
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
	public List<LayerStandard> getRibinbone(LayerStandard layer) {
		List<LayerStandard> ribbone = Util.newList(0);
		if (layer == null) return ribbone;
		LayerStandard ribLayer = layer.getRibinLayer();
		if (ribLayer == null || ribLayer == memoryLayer) return ribbone;
		
		ribbone.add(0, layer);
		while (ribLayer != null) {
			ribbone.add(0, ribLayer);
			ribLayer = ribLayer.getPrevLayer();
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
	public List<LayerStandard> getRiboutbone(LayerStandard layer) {
		List<LayerStandard> ribbone = Util.newList(0);
		if (layer == null) return ribbone;
		LayerStandard ribLayer = layer.getRiboutLayer();
		if (ribLayer == null || ribLayer == memoryLayer) return ribbone;
		
		ribbone.add(layer);
		while (ribLayer != null) {
			ribbone.add(ribLayer);
			LayerStandard nextLayer = ribLayer.getNextLayer();
			if (nextLayer == null || nextLayer.getRibinLayer() != ribLayer)
				ribLayer = nextLayer;
			else
				ribLayer = null;
		}
		
		return ribbone;
	}


	/**
	 * Finding layer by specified identifier.
	 * @param layerId specified identifier.
	 * @return found layer.
	 */
	public LayerStandard findLayer(int layerId) {
		List<LayerStandard> all = Util.newList(0);
		if (inputLayer != null) all.add(inputLayer);
		all.addAll(hiddenLayers);
		if (outputLayer != null) all.add(outputLayer);
		if (memoryLayer != null) all.add(memoryLayer);
		
		List<List<LayerStandard>> bones = getRibinbones();
		for (List<LayerStandard> bone : bones) all.addAll(bone);
		bones = getRiboutbones();
		for (List<LayerStandard> bone : bones) all.addAll(bone);
		
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
	protected static int findLayer(List<LayerStandard> bone, LayerStandard layer) {
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
	public Neuron findNeuron(int neuronId) {
		List<LayerStandard> layers = getNonemptyLayers();
		for (LayerStandard layer : layers) {
			int index = layer.indexOf(neuronId);
			if (index >= 0) return layer.get(index);
		}
		
		return null;
	}
	
	
	/**
	 * Getting non-empty layers.
	 * @return list of non-empty layers.
	 */
	private List<LayerStandard> getNonemptyLayers() {
		List<LayerStandard> nonempty = Util.newList(0);
		
		List<LayerStandard> all = Util.newList(0);
		if (inputLayer != null) all.add(inputLayer);
		all.addAll(hiddenLayers);
		if (outputLayer != null) all.add(outputLayer);
		if (memoryLayer != null) all.add(memoryLayer);
		
		for (LayerStandard layer : all) {
			if (layer == null || layer.size() == 0) continue;
			
			nonempty.add(layer);
			
			LayerStandard ribLayer = layer.getRibinLayer();
			while (ribLayer != null && ribLayer != memoryLayer) {
				if (ribLayer.size() > 0) nonempty.add(ribLayer);
				ribLayer = ribLayer.getPrevLayer();
			}
			
			ribLayer = layer.getRiboutLayer();
			while (ribLayer != null && ribLayer != memoryLayer) {
				if (ribLayer.size() > 0) nonempty.add(ribLayer);
				ribLayer = ribLayer.getNextLayer();
			}
		}
		
		return nonempty;
	}

	
	@Override
	public synchronized NeuronValue[] eval(Record inputRecord, boolean refresh) throws RemoteException {
		if (inputRecord == null) return null;
		List<LayerStandard> backbone = getBackbone();
		if (backbone.size() == 0) return null;
		
		if (refresh) {
			List<LayerStandard> nonempty = getNonemptyLayers();
			for (LayerStandard layer : nonempty) {
				for (int j = 0; j < layer.size(); j++) {
					Neuron neuron = layer.get(j);
					neuron.setInput(layer.newNeuronValue());
					neuron.setOutput(layer.newNeuronValue());
				}
			}
		}
		
		for (int i = 0; i < backbone.size(); i++) {
			LayerStandard layer = backbone.get(i);
			List<LayerStandard> ribinbone = getRibinbone(layer);
			if (ribinbone != null && ribinbone.size() > 1 && inputRecord.ribInput != null) {
				int id = ribinbone.get(0).id();
				if (inputRecord.ribInput.containsKey(id))
					eval(ribinbone, inputRecord.ribInput.get(id), refresh);
			}
			
			if (inputRecord.input != null) {
				NeuronValue[] output = eval(backbone, i, inputRecord.input, refresh);
				
				List<LayerStandard> riboutbone = getRiboutbone(layer);
				if (riboutbone != null && riboutbone.size() > 1)
					eval(riboutbone, output, refresh);
			}
		}
		
		if (memoryLayer != null) {
			for (int j = 0; j < memoryLayer.size(); j++) eval(memoryLayer.get(j), refresh);
		}
		
		return backbone.get(backbone.size() - 1).getOutput();
	}


	/**
	 * Evaluating bone with specified input.
	 * @param bone list of layers including input layer.
	 * @param input specified input.
	 * @return evaluated output.
	 */
	protected static NeuronValue[] eval(List<LayerStandard> bone, NeuronValue[] input) {
		if (bone == null || bone.size() == 0) return null;
		
		for (LayerStandard layer : bone) {
			for (int j = 0; j < layer.size(); j++) {
				Neuron neuron = layer.get(j);
				neuron.setInput(layer.newNeuronValue());
				neuron.setOutput(layer.newNeuronValue());
			}
		}

		LayerStandard layer0 = bone.get(0);
		input = NeuronValue.adjustArray(input, layer0.size(), layer0);
		for (int j = 0; j < layer0.size(); j++) {
			Neuron neuron = layer0.get(j);
			neuron.setInput(input[j]);
			neuron.setOutput(input[j]);
		}
		
		for (int i = 1;  i < bone.size(); i++) {
			LayerStandard layer = bone.get(i);
			for (int j = 0; j < layer.size(); j++) layer.get(j).eval();
		}
		
		return bone.get(bone.size() - 1).getOutput();
	}
	
	
	/**
	 * Evaluating bone with specified input.
	 * @param bone list of layers including input layer.
	 * @param input specified input.
	 * @param refresh refresh mode.
	 * @return evaluated output.
	 */
	private NeuronValue[] eval(List<LayerStandard> bone, NeuronValue[] input, boolean refresh) {
		if (bone.size() == 0) return null;
		for (int i = 0; i < bone.size(); i++) {
			eval(bone, i, input, refresh);
		}
		return bone.get(bone.size() - 1).getOutput();
	}
	
	
	/**
	 * Evaluating a layer of the bone with specified input.
	 * @param bone list of layers including input layer.
	 * @param index index.
	 * @param input specified input.
	 * @param refresh refresh mode.
	 * @return evaluated output.
	 */
	private NeuronValue[] eval(List<LayerStandard> bone, int index, NeuronValue[] input, boolean refresh) {
		LayerStandard layer = bone.get(index);
		input = NeuronValue.adjustArray(input, layer.size(), layer);
		
		if (index == 0) {
			for (int j = 0; j < layer.size(); j++) {
				Neuron neuron = layer.get(j);
				neuron.setInput(input[j]);
				neuron.setOutput(input[j]);
			}
		}
		else {
			for (int j = 0; j < layer.size(); j++) eval(layer.get(j), refresh);
		}
		
		return layer.getOutput();
	}

	
	/**
	 * Evaluating neuron.
	 * @param neuron specified neuron.
	 * @param refresh refresh mode. If true, neuron output is not re-evaluated because it was refreshed with initial values. If false, neuron output is re-evaluated.
	 * @return evaluated output.
	 */
	private NeuronValue eval(Neuron neuron, boolean refresh) {
		if (!refresh) return neuron.eval();
		
		List<WeightedNeuron> sources = Util.newList(0);
		sources.addAll(Arrays.asList(neuron.getPrevNeurons()));
		if (neuron.getLayer() != null && neuron.getLayer().getRibinLayer() != memoryLayer)
			sources.addAll(Arrays.asList(neuron.getRibinNeurons()));
		sources.addAll(Arrays.asList(neuron.getPrevNeuronsImplicit()));
		
		if (sources.size() == 0) {
			NeuronValue out = neuron.getInput();
			neuron.setOutput(out);
			return out;
		}
		
		NeuronValue in = neuron.getBias();
		for (WeightedNeuron source : sources) {
			in = in.add(source.neuron.getOutput().multiply(source.weight.value));
		}
		
		neuron.setInput(in);
		NeuronValue out = neuron.getActivateRef().eval(in);
		neuron.setOutput(out);
		return out;
	}
	
	
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
	public String toString() {
		try {
			return toText(this, null);
		}
		catch (Throwable e) {}
		
		return super.toString();
	}


	/**
	 * Calculating errors.
	 * @param bone list of layers including input layer.
	 * @param output output of output layer. 
	 * @return list of errors.
	 */
	@SuppressWarnings("unused")
	@Deprecated
	private static List<NeuronValue[]> bpCalcErrors(List<LayerStandard> bone, NeuronValue[] output) {
		List<NeuronValue[]> errors = Util.newList(0);
		if (bone.size() < 2) return errors;
		
		for (int i = bone.size() - 1; i >= 1; i--) {
			LayerStandard layer = bone.get(i);
			LayerStandard nextLayer = i < bone.size() - 1 ? bone.get(i + 1) : null;
			NeuronValue[] error = NeuronValue.makeArray(layer.size(), layer);
			errors.add(0, error);
			
			for (int j = 0; j < layer.size(); j++) {
				Neuron neuron = layer.get(j);
				NeuronValue out = neuron.getOutput();
				NeuronValue derivative = neuron.getActivateRef().derivative(out);
				
				if (i == bone.size() - 1)
					error[j] = output[j].subtract(out).multiplyDerivative(derivative);
				else {
					NeuronValue rsum = layer.newNeuronValue();
					NeuronValue[] nextError = errors.get(1);
					WeightedNeuron[] targets = neuron.getNextNeurons(nextLayer);
					for (WeightedNeuron target : targets) {
						int index = nextLayer.indexOf(target.neuron);
						rsum = rsum.add(nextError[index].multiply(target.weight.value));
					}
					error[j] = rsum.multiplyDerivative(derivative);
				}
			}
		}
		
		return errors;
	}

	
	/**
	 * Updating weights and biases.
	 * @param bone list of layers including input layer.
	 * @param errors list of errors. Error list excludes input error and so it is 1 less than backbone. 
	 * @param learningRate learning rate.
	 */
	@Deprecated
	private static void bpUpdateWeightsBiases(List<LayerStandard> bone, List<NeuronValue[]> errors, double learningRate) {
		if (bone.size() < 2) return;
		
		for (int i = 0; i < bone.size() - 1; i++) {
			LayerStandard layer = bone.get(i);
			LayerStandard nextLayer = bone.get(i + 1);
			NeuronValue[] error = i > 0 ? errors.get(i - 1) : null;
			NeuronValue[] nextError = errors.get(i);
			
			for (int j = 0; j < layer.size(); j++) {
				Neuron neuron = layer.get(j);
				NeuronValue out = neuron.getOutput();
				
				WeightedNeuron[] targets = neuron.getNextNeurons(nextLayer);
				for (WeightedNeuron target : targets) {
					Weight nw = target.weight;
					int index = nextLayer.indexOf(target.neuron);
					NeuronValue delta = nextError[index].multiply(out).multiply(learningRate);
					nw.value = nw.value.add(delta);
				}
				
				if (i > 0) {
					NeuronValue delta = error[j].multiply(learningRate);
					neuron.setBias(neuron.getBias().add(delta));
				}
			}
			
			if (i == bone.size() - 1) {
				for (int j = 0; j < nextLayer.size(); j++) {
					Neuron neuron = nextLayer.get(j);
					NeuronValue delta = nextError[j].multiply(learningRate);
					neuron.setBias(neuron.getBias().add(delta));
				}
			}
		}
	}
	
	
	/**
	 * Update weights and biases of the triple including center layer.
	 * @param centerLayer specified center layer.
	 * @param bone list of layers including input layer.
	 * @param errors list of errors. Error list excludes input error and so it is 1 less than bone. 
	 * @param learningRate learning rate.
	 */
	@SuppressWarnings("unused")
	@Deprecated
	private static void bpUpdateWeightsBiasesAttachedTriple(LayerStandard centerLayer, List<LayerStandard> bone, List<NeuronValue[]> errors, double learningRate) {
		if (bone.size() < 2) return;

		LayerStandard prevLayer = centerLayer.getPrevLayer();
		if (prevLayer == null) return;
		LayerStandard nextLayer = centerLayer.getNextLayer();
		if (nextLayer == null) return;
		
		int nextErrorIndex = findLayer(bone, nextLayer) - 1;
		if (nextErrorIndex < 0) return;
		
		//Evaluating center neurons.
		for (int j = 0; j < centerLayer.size(); j++) centerLayer.get(j).eval();
		
		//Updating errors of center layer.
		NeuronValue[] centerError = NeuronValue.makeArray(centerLayer.size(), centerLayer);
		NeuronValue[] nextError = errors.get(nextErrorIndex);
		for (int j = 0; j < centerLayer.size(); j++) {
			Neuron centerNeuron = centerLayer.get(j);
			NeuronValue out = centerNeuron.getOutput();
			NeuronValue derivative = centerNeuron.getActivateRef().derivative(out);

			NeuronValue rsum = centerLayer.newNeuronValue();;
			WeightedNeuron[] targets = centerNeuron.getNextNeurons(nextLayer);
			for (WeightedNeuron target : targets) {
				int index = nextLayer.indexOf(target.neuron);
				rsum = rsum.add(nextError[index].multiply(target.weight.value));
			}
			centerError[j] = rsum.multiplyDerivative(derivative);
		}
		
		List<LayerStandard> newBackbone = Util.newList(3);
		newBackbone.add(prevLayer);
		newBackbone.add(centerLayer);
		newBackbone.add(nextLayer);
		List<NeuronValue[]> newErrors = Util.newList(2);
		newErrors.add(centerError);
		newErrors.add(nextError);
		
		bpUpdateWeightsBiases(newBackbone, newErrors, learningRate);
	}


}


