/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann;

import java.rmi.NoSuchObjectException;
import java.rmi.Remote;
import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;
import java.util.Arrays;
import java.util.List;

/**
 * This class is abstract implementation of neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class NetworkAbstract implements Network {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Maximum iteration of learning neural network.
	 */
	public final static String LEARN_MAX_ITERATION_FIELD = "learn_max_iteration";
	
	
	/**
	 * Terminated threshold of learning neural network.
	 */
	public final static String LEARN_TERMINATED_THRESHOLD_FIELD = "learn_terminated_threshold";

	
	/**
	 * Learning rate.
	 */
	public final static String LEARN_RATE_FIELD = "learn_rate";

	
	/**
	 * Layer type.
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	public static enum LayerType {
		
		/**
		 * Input layer.
		 */
		input,
		
		/**
		 * Hidden layer.
		 */
		hidden,
		
		/**
		 * Output layer.
		 */
		output,
		
		/**
		 * Memory layer.
		 */
		memory,
		
		/**
		 * Input rib layer.
		 */
		ribin,
		
		/**
		 * Memory layer.
		 */
		ribout,
		
		/**
		 * Unknown layer.
		 */
		unknown,
		
	}
	
	
    /**
	 * Internal identifier.
	 */
	protected Id idRef = new Id();
	
	
	/**
	 * Activation function reference.
	 */
	protected Function activateRef = null;

	
	/**
	 * Input layer.
	 */
	protected Layer inputLayer = null;

	
	/**
	 * Memory layer.
	 */
	protected List<Layer> hiddenLayers = Util.newList(0);

	
	/**
	 * Output layer.
	 */
	protected Layer outputLayer = null;
	

	/**
	 * Memory layer.
	 */
	protected Layer memoryLayer = null;
	
	
	/**
	 * Holding a list of listeners.
	 */
    protected transient NetworkListenerList listenerList = new NetworkListenerList();

    
    /**
     * Flag to indicate whether algorithm learning process was started.
     */
    protected volatile boolean doStarted = false;
    
    
    /**
     * Flag to indicate whether algorithm learning process was paused.
     */
    protected volatile boolean doPaused = false;

    
	/**
	 * Configuration.
	 */
	protected NetworkConfig config = new NetworkConfig();
	
	
	/**
	 * Flag to indicate whether this hidden Markov model was exported.
	 */
	protected boolean exported = false;

	
	/**
	 * Default constructor.
	 */
	protected NetworkAbstract() {
		config.put(LEARN_MAX_ITERATION_FIELD, LEARN_MAX_ITERATION_DEFAULT);
		config.put(LEARN_TERMINATED_THRESHOLD_FIELD, LEARN_TERMINATED_THRESHOLD_DEFAULT);
		config.put(LEARN_RATE_FIELD, LEARN_RATE_DEFAULT);
	}
	
	
	/**
	 * Constructor with number of neurons.
	 * @param nInputNeuron number of input neurons.
	 * @param nOutputNeuron number of output neurons.
	 * @param nHiddenLayer number of hidden layers.
	 * @param nHiddenNeuron number of hidden neurons.
	 * @param nMemoryNeuron number of memory neurons.
	 */
	public NetworkAbstract(int nInputNeuron, int nOutputNeuron, int nHiddenLayer, int nHiddenNeuron, int nMemoryNeuron) {
		this();
		this.activateRef = newFunction();
		
		nInputNeuron = nInputNeuron < 1 ? 1 : nInputNeuron;
		nOutputNeuron = nOutputNeuron < 1 ? 1 : nOutputNeuron;
		nHiddenLayer = nHiddenLayer < 0 ? 0 : nHiddenLayer;
		if (nHiddenLayer == 0) nHiddenNeuron = 0;
		if (nHiddenNeuron == 0) nHiddenLayer = 0;
		nHiddenNeuron = nHiddenNeuron < 0 ? 0 : nHiddenNeuron;
		nMemoryNeuron = nMemoryNeuron < 0 ? 0 : nMemoryNeuron;
		
		this.inputLayer = newLayer(nInputNeuron, null, null);
		
		if (nHiddenNeuron > 0) {
			this.hiddenLayers = Util.newList(nHiddenLayer);
			for (int l = 0; l < nHiddenLayer; l++) {
				Layer prevHiddenLayer = l == 0 ? this.inputLayer : this.hiddenLayers.get(l - 1);
				Layer hiddenLayer = newLayer(nHiddenNeuron, prevHiddenLayer, null);
				this.hiddenLayers.add(hiddenLayer);
			}
		}
		
		Layer preOutputLayer = this.hiddenLayers.size() > 0 ? this.hiddenLayers.get(this.hiddenLayers.size() - 1) : this.inputLayer;
		this.outputLayer = newLayer(nOutputNeuron, preOutputLayer, null);
		
		if (nMemoryNeuron > 0 && nHiddenNeuron > 0) {
			this.memoryLayer = newLayer(nMemoryNeuron, null, null);
			this.outputLayer.setRiboutLayer(this.memoryLayer); //this.outputLayer.setRiboutLayer(this.hiddenLayers.get(this.hiddenLayers.size() - 1))
			this.memoryLayer.setRibinLayer(this.hiddenLayers.get(0));
		}
	}

	
	/**
	 * Constructor with number of neurons.
	 * @param nInputNeuron number of input neurons.
	 * @param nOutputNeuron number of output neurons.
	 * @param nHiddenLayer number of hidden layers.
	 * @param nHiddenNeuron number of hidden neurons.
	 */
	public NetworkAbstract(int nInputNeuron, int nOutputNeuron, int nHiddenLayer, int nHiddenNeuron) {
		this(nInputNeuron, nOutputNeuron, nHiddenLayer, nHiddenNeuron, 0);
	}
	
	
	/**
	 * Constructor with number of neurons.
	 * @param nInputNeuron number of input neurons.
	 * @param nOutputNeuron number of output neurons.
	 */
	public NetworkAbstract(int nInputNeuron, int nOutputNeuron) {
		this(nInputNeuron, nOutputNeuron, 0, 0, 0);
	}

	
	/**
	 * Create logistic function.
	 * @return logistic function.
	 */
	protected abstract Function newFunction();
	
	
	/**
	 * Create layer.
	 * @return created layer.
	 */
	protected abstract Layer newLayer();
	
	
	/**
	 * Creating new layer. This method can be called from derived classes.
	 * @param nNeuron number of neurons.
	 * @param prevLayer previous layer.
	 * @param nextLayer next layer.
	 * @return new layer.
	 */
	private Layer newLayer(int nNeuron, Layer prevLayer, Layer nextLayer) {
		Layer layer = newLayer();
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
	public LayerType typeOf(Layer layer) {
		if (layer == null) return LayerType.unknown;
		
		if (inputLayer != null && layer == inputLayer) return LayerType.input;
		if (outputLayer != null && layer == outputLayer) return LayerType.output;
		for (Layer hiddenLayer : hiddenLayers) {
			if (layer == hiddenLayer) return LayerType.hidden;
		}
		if (memoryLayer != null && layer == memoryLayer) return LayerType.memory;
		
		List<Layer> backbone = getBackbone();
		for (Layer l : backbone) {
			List<Layer> ribin = getRibinbone(l);
			if (findLayer(ribin, layer) >= 0) return LayerType.ribin;
			List<Layer> ribout = getRiboutbone(l);
			if (findLayer(ribout, layer) >= 0) return LayerType.ribout;
		}

		return LayerType.unknown;
	}
	
	
	/**
	 * Getting input layer.
	 * @return input layer.
	 */
	public Layer getInputLayer() {
		return inputLayer;
	}

	
	/**
	 * Getting hidden layers.
	 * @return array of hidden layers.
	 */
	public Layer[] getHiddenLayers() {
		return hiddenLayers.toArray(new Layer[] {});
	}

	
	/**
	 * Getting index of hidden layer.
	 * @param layer hidden layer.
	 * @return index of hidden layer.
	 */
	public int hiddenIndexOf(Layer layer) {
		if (layer == null) return -1;
		
		for (int i = 0; i < hiddenLayers.size(); i++) {
			Layer hiddenLayer = hiddenLayers.get(i);
			if (layer == hiddenLayer) return i;
		}
		
		return -1;
	}
	
	
	/**
	 * Getting output layer.
	 * @return output layer.
	 */
	public Layer getOutputLayer() {
		return outputLayer;
	}

	
	/**
	 * Getting memory layer.
	 * @return memory layer.
	 */
	public Layer getMemoryLayer() {
		return memoryLayer;
	}

	
	/**
	 * Getting backbone which is chain of main layers.
	 * @return backbone which is chain of main layers.
	 */
	public List<Layer> getBackbone() {
		List<Layer> backbone = Util.newList(2);
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
	public List<List<Layer>> getRibinbones() {
		List<List<Layer>> ribbones = Util.newList(0);
		
		List<Layer> backbone = getBackbone();
		for (Layer layer : backbone) {
			List<Layer> ribbone = getRibinbone(layer);
			if (ribbone.size() > 0) ribbones.add(ribbone);
		}
		
		return ribbones;
	}
	
	
	/**
	 * Getting input rib bone of specified layer.
	 * @param layer specified layer.
	 * @return input rib bone of specified layer.
	 */
	public List<Layer> getRibinbone(Layer layer) {
		List<Layer> ribbone = Util.newList(0);
		if (layer == null) return ribbone;
		Layer ribLayer = layer.getRibinLayer();
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
	public List<List<Layer>> getRiboutbones() {
		List<List<Layer>> ribbones = Util.newList(0);
		
		List<Layer> backbone = getBackbone();
		for (Layer layer : backbone) {
			List<Layer> ribbone = getRiboutbone(layer);
			if (ribbone.size() > 0) ribbones.add(ribbone);
		}
		
		return ribbones;
	}

	
	/**
	 * Getting output rib bone of specified layer.
	 * @param layer specified layer.
	 * @return output rib bone of specified layer.
	 */
	public List<Layer> getRiboutbone(Layer layer) {
		List<Layer> ribbone = Util.newList(0);
		if (layer == null) return ribbone;
		Layer ribLayer = layer.getRiboutLayer();
		if (ribLayer == null || ribLayer == memoryLayer) return ribbone;
		
		ribbone.add(layer);
		while (ribLayer != null) {
			ribbone.add(ribLayer);
			Layer nextLayer = ribLayer.getNextLayer();
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
	public Layer findLayer(int layerId) {
		List<Layer> all = Util.newList(0);
		if (inputLayer != null) all.add(inputLayer);
		all.addAll(hiddenLayers);
		if (outputLayer != null) all.add(outputLayer);
		if (memoryLayer != null) all.add(memoryLayer);
		
		List<List<Layer>> bones = getRibinbones();
		for (List<Layer> bone : bones) all.addAll(bone);
		bones = getRiboutbones();
		for (List<Layer> bone : bones) all.addAll(bone);
		
		for (Layer layer : all) {
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
	protected static int findLayer(List<Layer> bone, Layer layer) {
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
		List<Layer> layers = getNonemptyLayers();
		for (Layer layer : layers) {
			int index = layer.indexOf(neuronId);
			if (index >= 0) return layer.get(index);
		}
		
		return null;
	}
	
	
	/**
	 * Getting non-empty layers.
	 * @return list of non-empty layers.
	 */
	private List<Layer> getNonemptyLayers() {
		List<Layer> nonempty = Util.newList(0);
		
		List<Layer> all = Util.newList(0);
		if (inputLayer != null) all.add(inputLayer);
		all.addAll(hiddenLayers);
		if (outputLayer != null) all.add(outputLayer);
		if (memoryLayer != null) all.add(memoryLayer);
		
		for (Layer layer : all) {
			if (layer == null || layer.size() == 0) continue;
			
			nonempty.add(layer);
			
			Layer ribLayer = layer.getRibinLayer();
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
	public synchronized Value[] eval(Record inputRecord, boolean refresh) throws RemoteException {
		if (inputRecord == null) return null;
		List<Layer> backbone = getBackbone();
		if (backbone.size() == 0) return null;
		
		if (refresh) {
			List<Layer> nonempty = getNonemptyLayers();
			for (Layer layer : nonempty) {
				for (int j = 0; j < layer.size(); j++) {
					Neuron neuron = layer.get(j);
					neuron.setInput(layer.newValue());
					neuron.setOutput(layer.newValue());
				}
			}
		}
		
		for (int i = 0; i < backbone.size(); i++) {
			Layer layer = backbone.get(i);
			List<Layer> ribinbone = getRibinbone(layer);
			if (ribinbone != null && ribinbone.size() > 1 && inputRecord.ribInput != null) {
				int id = ribinbone.get(0).id();
				if (inputRecord.ribInput.containsKey(id))
					eval(ribinbone, inputRecord.ribInput.get(id), refresh);
			}
			
			if (inputRecord.input != null) {
				Value[] output = eval(backbone, i, inputRecord.input, refresh);
				
				List<Layer> riboutbone = getRiboutbone(layer);
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
	protected static Value[] eval(List<Layer> bone, Value[] input) {
		if (bone == null || bone.size() == 0) return null;
		
		for (Layer layer : bone) {
			for (int j = 0; j < layer.size(); j++) {
				Neuron neuron = layer.get(j);
				neuron.setInput(layer.newValue());
				neuron.setOutput(layer.newValue());
			}
		}

		Layer layer0 = bone.get(0);
		input = Value.adjustArray(input, layer0.size(), layer0);
		for (int j = 0; j < layer0.size(); j++) {
			Neuron neuron = layer0.get(j);
			neuron.setInput(input[j]);
			neuron.setOutput(input[j]);
		}
		
		for (int i = 1;  i < bone.size(); i++) {
			Layer layer = bone.get(i);
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
	private Value[] eval(List<Layer> bone, Value[] input, boolean refresh) {
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
	private Value[] eval(List<Layer> bone, int index, Value[] input, boolean refresh) {
		Layer layer = bone.get(index);
		input = Value.adjustArray(input, layer.size(), layer);
		
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
	private Value eval(Neuron neuron, boolean refresh) {
		if (!refresh) return neuron.eval();
		
		List<WeightedNeuron> sources = Util.newList(0);
		sources.addAll(Arrays.asList(neuron.getPrevNeurons()));
		if (neuron.getLayer() != null && neuron.getLayer().getRibinLayer() != memoryLayer)
			sources.addAll(Arrays.asList(neuron.getRibinNeurons()));
		sources.addAll(Arrays.asList(neuron.getPrevNeuronsImplicit()));
		
		if (sources.size() == 0) {
			Value out = neuron.getInput();
			neuron.setOutput(out);
			return out;
		}
		
		Value in = neuron.getBias();
		for (WeightedNeuron source : sources) {
			in = in.add(source.weight.value.multiply(source.neuron.getOutput()));
		}
		
		neuron.setInput(in);
		Value out = neuron.getActivateRef().eval(in);
		neuron.setOutput(out);
		return out;
	}
	
	
	/**
	 * Learning the neural network.
	 * @param sample sample for learning.
	 * @return learned error.
	 */
	public Value[] learn(Record[] sample) {
		try {
			return learn(Arrays.asList(sample));
		} catch (Throwable e) {Util.trace(e);}
		return null;
	}
	
	
	@Override
	public void addListener(NetworkListener listener) throws RemoteException {
		synchronized (listenerList) {
			listenerList.add(NetworkListener.class, listener);
		}
	}


	@Override
	public void removeListener(NetworkListener listener) throws RemoteException {
		synchronized (listenerList) {
			listenerList.remove(NetworkListener.class, listener);
		}
	}
	
	
	/**
	 * Getting an array of listeners.
	 * @return array of listeners.
	 */
	protected NetworkListener[] getListeners() {
		if (listenerList == null) return new NetworkListener[] {};
		synchronized (listenerList) {
			return listenerList.getListeners(NetworkListener.class);
		}

	}
	
	
	/**
	 * Firing information event.
	 * @param evt information event.
	 */
	protected void fireInfoEvent(NetworkInfoEvent evt) {
		if (listenerList == null) return;
		
		NetworkListener[] listeners = getListeners();
		for (NetworkListener listener : listeners) {
			try {
				listener.receivedInfo(evt);
			}
			catch (Throwable e) { 
				Util.trace(e);
			}
		}
	}

	
	/**
	 * Firing learning event.
	 * @param evt learning event.
	 */
	protected void fireDoEvent(NetworkDoEvent evt) {
		if (listenerList == null) return;
		
		NetworkListener[] listeners = getListeners();
		for (NetworkListener listener : listeners) {
			try {
				listener.receivedDo(evt);
			}
			catch (Throwable e) {
				Util.trace(e);
			}
		}
	}


	@Override
	public boolean doPause() throws RemoteException {
		if (!isDoRunning()) return false;
		
		doPaused  = true;
		
		try {
			wait();
		} 
		catch (Throwable e) {
			Util.trace(e);
		}
		
		return true;
	}


	@Override
	public boolean doResume() throws RemoteException {
		if (!isDoPaused()) return false;
		
		doPaused = false;
		notifyAll();
		
		return true;
	}


	@Override
	public boolean doStop() throws RemoteException {
		if (!isDoStarted()) return false;
		
		doStarted = false;
		
		if (doPaused) {
			doPaused = false;
			notifyAll();
		}
		
		try {
			wait();
		} 
		catch (Throwable e) {
			Util.trace(e);
		}
		
		return true;
	}


	@Override
	public boolean isDoStarted() throws RemoteException {
		return doStarted;
	}


	@Override
	public boolean isDoPaused() throws RemoteException {
		return doStarted && doPaused;
	}


	@Override
	public boolean isDoRunning() throws RemoteException {
		return doStarted && !doPaused;
	}

	
	@Override
	public NetworkConfig getConfig() throws RemoteException {
		return config;
	}


	@Override
	public void setConfig(NetworkConfig config) throws RemoteException {
		if (config != null) this.config.putAll(config);
	}


	@Override
	public synchronized Remote export(int serverPort) throws RemoteException {
		if (exported) return null;
		
		Remote stub = null;
		try {
			stub = UnicastRemoteObject.exportObject(this, serverPort);
		}
		catch (Exception e) {
			try {
				if (stub != null) UnicastRemoteObject.unexportObject(this, true);
			}
			catch (Exception e2) {}
			stub = null;
		}
	
		exported = stub != null;
		return stub;
	}


	@Override
	public synchronized void unexport() throws RemoteException {
		if (!exported) return;

		try {
        	UnicastRemoteObject.unexportObject(this, true);
			exported = false;
		}
		catch (NoSuchObjectException e) {
			exported = false;
			Util.trace(e);
		}
		catch (Throwable e) {
			Util.trace(e);
		}
	}

	
	@Override
	public void close() throws Exception {
		try {
			unexport();
		}
		catch (Throwable e) {
			Util.trace(e);
		}
	}


	/**
	 * Verbalize a list of layers.
	 * @param layers list of layers.
	 * @param tab tab text.
	 * @return verbalized text.
	 */
	private static String toText(List<Layer> layers, String tab) {
		StringBuffer buffer = new StringBuffer();
		for (int i = 0; i < layers.size(); i++) {
			if (i > 0) buffer.append("\n");

			String layerText = LayerImpl.toText(layers.get(i), null);
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
	protected static String toText(NetworkAbstract network, String tab) {
		StringBuffer buffer = new StringBuffer();
		String internalTab = "    ";
		
		List<Layer> backbone = network.getBackbone();
		if (backbone.size() > 0) {
			buffer.append("BACKBONE:\n");
			buffer.append(toText(backbone, internalTab));
		}
		
		Layer memory = network.getMemoryLayer();
		if (memory != null) {
			buffer.append("MEMORY:\n");
			buffer.append(toText(Arrays.asList(memory), internalTab));
		}
		
		List<List<Layer>> ribinBones = network.getRibinbones();
		for (List<Layer> ribinBone : ribinBones) {
			if (ribinBone.size() > 0) {
				buffer.append("RIBIN BONE:\n");
				buffer.append(toText(ribinBone, internalTab));
			}
		}
		
		List<List<Layer>> riboutBones = network.getRiboutbones();
		for (List<Layer> riboutBone : riboutBones) {
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


//	public synchronized double[] learn(double learningRate, double errorThreshold, int maxIteration, Collection<double[]> mainSample, List<Collection<double[]>> auxSamples) throws RemoteException {
//		if (mainSample == null || mainSample.size() == 0) return null;
//		errorThreshold = errorThreshold < 0 ? 0 : errorThreshold;
//		maxIteration = maxIteration < 0 ? 0 : maxIteration;
//		
//		List<Layer> backbone = getBackbone();
//		if (backbone.size() < 2) return null;
//		List<List<Layer>> ribbones = getRibinbones();
//		int nRib = 0;
//		if (ribbones != null && ribbones.size() > 0 && auxSamples != null && auxSamples.size() > 0)
//			nRib = Math.min(ribbones.size(), auxSamples.size());
//		
//		double[] error = null;
//		int iteration = 0;
//		doStarted = true;
//		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
//			for (int i = 0; i < nRib; i++) {
//				List<Layer> ribbon = ribbones.get(i);
//				Collection<double[]> auxSample = auxSamples.get(i);
//				if (ribbon != null && ribbon.size() >= 2 && auxSample != null && auxSample.size() > 0)
//					learn(auxSample, ribbon, null, learningRate);
//			}
//			
//			error = learn(mainSample, backbone, memoryLayer, learningRate);
//
//			iteration ++;
//
//			if (error == null || error.length == 0)
//				doStarted = false;
//			else {
//				double errorMean = 0;
//				for (double r : error) errorMean += Math.abs(r);
//				errorMean = errorMean / error.length;
//				
//				if (errorMean < errorThreshold) doStarted = false; 
//			}
//			
//			synchronized (this) {
//				while (doPaused) {
//					notifyAll();
//					try {
//						wait();
//					} catch (Exception e) {LogUtil.trace(e);}
//				}
//			}
//
//		}
//		
//		synchronized (this) {
//			doStarted = false;
//			doPaused = false;
//			
//			notifyAll();
//		}
//		
//		return error;
//	}
//
//	
//	/**
//	 * Learning the backbone as neural network.
//	 * @param sample sample includes input and output.
//	 * @param bone list of layers including input layer.
//	 * @param memoryLayer memory layer.
//	 * @param learningRate learning rate.
//	 * @return output error.
//	 * @throws RemoteException if any error raises.
//	 */
//	private static double[] learn(Collection<double[]> sample, List<Layer> bone, Layer memoryLayer, double learningRate) {
//		List<Layer> layers = Util.newList(bone.size());
//		layers.addAll(bone);
//		if (memoryLayer != null) layers.add(memoryLayer);
//		for (Layer layer : layers) {
//			for (int j = 0; j < layer.size(); j++) {
//				Neuron neuron = layer.get(j);
//				neuron.setInput(0);
//				neuron.setOutput(0);
//				
//				if (layer.getRibinLayer() == null && layer.getRiboutLayer() == null)
//					neuron.setBias(0);
//			}
//		}
//		
//		int nInput = bone.get(0).size();
//		int nOutput = bone.get(bone.size() - 1).size();
//		List<double[]> errors = Util.newList(0);
//		for (double[] record : sample) {
//			if (record == null || nInput > record.length) continue;
//			double[] input = Arrays.copyOfRange(record, 0, nInput);
//			double[] output = Arrays.copyOfRange(record, nInput, nInput + nOutput);
//			
//			//Calculating outputs.
//			eval(bone, input);
//			
//			//Calculating errors.
//			errors = calcErrors(bone, output);
//			
//			//Updating weights and biases.
//			updateWeightsBiases(bone, errors, learningRate);
//			
//			//Updating weights and biases related to memory layer.
//			if (memoryLayer != null && memoryLayer.size() > 0)
//				updateWeightsBiasesTriple(memoryLayer, bone, errors, learningRate);
//		}
//		
//		return errors.size() > 0 ? errors.get(errors.size() - 1) : null;
//	}
	
	
}
