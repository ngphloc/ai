/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import java.awt.Dimension;
import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;

import net.ea.ann.conv.filter.DeconvFilter;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.NetworkDoEvent;
import net.ea.ann.core.NetworkDoEventImpl;
import net.ea.ann.core.NetworkInfoEvent;
import net.ea.ann.core.NetworkListener;
import net.ea.ann.core.NetworkStandardImpl;
import net.ea.ann.core.NeuronValue;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.NetworkDoEvent.Type;

/**
 * This class is an abstract implementation of convolutional layer.
 * 
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
	 * List of convolutional layers.
	 */
	protected List<ConvLayer> convLayers = Util.newList(0);
	
	
	/**
	 * Fully connected network.
	 */
	protected NetworkStandardImpl fullNetwork = null;
	
	
	/**
	 * Neuron channel.
	 */
	protected int neuronChannel = 1;
	
	
	/**
	 * Constructor with neuron channel and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param idRef ID reference.
	 */
	protected ConvNetworkAbstract(int neuronChannel, Id idRef) {
		super(idRef);
		
		this.neuronChannel = neuronChannel;
		
		this.config.put(Raster.SOURCE_IMAGE_TYPE_FIELD, Raster.SOURCE_IMAGE_TYPE_DEFAULT);
		this.config.put(Raster.NORM_FIELD, Raster.NORM_DEFAULT);
		this.config.put(Raster.SOURCE_RESIZE_FIELD, Raster.SOURCE_RESIZE_DEFAULT);
		this.config.put(Raster.ALPHA_FIELD, Raster.ALPHA_DEFAULT);
	}

	
	/**
	 * Default constructor with neuron channel.
	 * @param neuronChannel neuron channel or depth.
	 */
	protected ConvNetworkAbstract(int neuronChannel) {
		this(neuronChannel, null);
	}

	
//	/**
//	 * Resetting data structures for initialization.
//	 */
//	protected void reset() {
//		convLayers.clear();
//		fullNetwork = null;
//		neuronChannel = 1;
//	}

	
	/**
	 * Initialize with image/raster specification and filters.
	 * @param width raster width.
	 * @param height raster height.
	 * @param filters specific filters.
	 * @param nFullHiddenOutputNeuron numbers of hidden neurons and output neurons of fully connected network.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int width, int height,
			Filter[] filters,
			int[] nFullHiddenOutputNeuron) {
//		reset();
		
		if (width <= 0 || height <= 0) return false;
		if (filters == null || filters.length == 0) return false;
		
		Dimension size = new Dimension(width, height);
		ConvLayer layer = null;
		layer = addConvLayers(filters, size, layer);
		if (layer == null) return false;
		
		Filter lastFilter = filters[filters.length-1];
		if (lastFilter instanceof DeconvFilter) {
			size.width *= lastFilter.slideWidth();
			size.height *= lastFilter.slideHeight();
		}
		else {
			size.width /= lastFilter.slideWidth();
			size.height /= lastFilter.slideHeight();
		}
		ConvLayer lastLayer = newLayer(size.width, size.height, null);
		if (lastLayer != null) {
			convLayers.add(lastLayer);
			layer.setNextLayer(lastLayer);
			layer = lastLayer;
		}
		
		if (nFullHiddenOutputNeuron == null || nFullHiddenOutputNeuron.length < 1) return true;
		
		fullNetwork = new NetworkStandardImpl(neuronChannel, Raster.toActivationRef(neuronChannel, isNorm()));
		int nInputNeuron = layer.getWidth() * layer.getHeight();
		if (nFullHiddenOutputNeuron.length == 1)
			fullNetwork.initialize(nInputNeuron, nFullHiddenOutputNeuron[0]);
		else {
			int length = nFullHiddenOutputNeuron.length;
			int[] nHiddenNeuron = Arrays.copyOf(nFullHiddenOutputNeuron, length-1);
			fullNetwork.initialize(nInputNeuron, nFullHiddenOutputNeuron[length-1], nHiddenNeuron);
		}

		try {
			fullNetwork.addListener(this);
		} catch (RemoteException e) {
			Util.trace(e);
		}
		
		return true;
	}
	
	
	/**
	 * Initialize with image/raster specification without fully connected network.
	 * @param width raster width.
	 * @param height raster height.
	 * @param filters specific filters.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int width, int height,
			Filter[] filters) {
		return initialize(width, height,
			filters,
			null);
	}
	
	
	/**
	 * Adding convolutional layers according to filters.
	 * @param filters array of filters
	 * @param size size of raster.
	 * @param prevLayer previous layer.
	 * @return current added layer.
	 */
	protected ConvLayer addConvLayers(Filter[] filters, Dimension size, ConvLayer prevLayer) {
		if (filters == null || filters.length == 0) return prevLayer;
		
		for (int i = 0; i < filters.length; i++) {
			Filter filter = filters[i];
			if (i > 0) {
				if (filter instanceof DeconvFilter) {
					size.width *= filter.slideWidth();
					size.height *= filter.slideHeight();
				}
				else {
					size.width /= filter.slideWidth();
					size.height /= filter.slideHeight();
				}
			}
			ConvLayer newLayer = newLayer(size.width, size.height, filter);
			if (newLayer == null) continue;
			
			convLayers.add(newLayer);
			if (prevLayer != null) prevLayer.setNextLayer(newLayer);
			prevLayer = newLayer;
		}
		
		return prevLayer;
	}
	
	
	/**
	 * Creating new convolutional layer with width, height, and filter.
	 * @param width raster width.
	 * @param height raster height.
	 * @param filter specific filter.
	 * @return created layer.
	 */
	public abstract ConvLayer newLayer(int width, int height, Filter filter);
	
	
	@Override
	public synchronized NeuronValue[] evaluateByRaster(Raster inputRaster) throws RemoteException {
		if (inputRaster == null) return null;
		ConvLayer inputLayer = convLayers.get(0);
		if (inputLayer == null) return null;
		
		NeuronValue[] input = inputRaster.convertFromRasterToNeuronValues(neuronChannel, inputLayer.getWidth(), inputLayer.getHeight(),
				getSourceImageType(), isSourceRasterResize(), isNorm());
		return evaluate(input);
	}


	/**
	 * Evaluate convolutional network by value input.
	 * @param input array of neuron values as input.
	 * @return evaluated array of neuron values.
	 */
	public NeuronValue[] evaluate(NeuronValue[] input) {
		if (convLayers.size() == 0 || input == null) return null;
		
		ConvLayer inputLayer = convLayers.get(0);
		if (inputLayer == null) return null;
		input = NeuronValue.adjustArray(input, inputLayer.size(), inputLayer);
		for (int i = 0; i < input.length; i++) {
			inputLayer.getNeurons()[i].setValue(input[i]);
		}
		if (convLayers.size() == 1) return inputLayer.getData();
		
		for (int i = 0; i < convLayers.size() -  1; i++) convLayers.get(i).forward();
		if (fullNetwork == null) return convLayers.get(convLayers.size()-1).getData(); 
		
		input = convLayers.get(convLayers.size()-1).getData();
		LayerStandard layer0 = fullNetwork.getInputLayer();
		if (layer0 == null) return input;
		
		Record record = new Record();
		record.input = input;
		try {
			return fullNetwork.evaluate(record, true);
		} catch (Throwable e) {
			Util.trace(e);
		}
		
		return null;
	}
	

	@Override
	public NeuronValue[] learn(Iterable<Record> sample) throws RemoteException {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		if (convLayers.size() == 0 || sample == null) return null;
		ConvLayer convInputLayer = getConvInputLayer();
		if (convInputLayer == null) return null;
		ConvLayer convOutputLayer = getConvOutputLayer();
		if (convOutputLayer == null) return null;
		
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		double learningRate = config.getAsReal(LEARN_RATE_FIELD);
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		int maxIteration = 0;
		for (Record record : sample) {
			if (record != null) maxIteration++; 
		}
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		for (Record record : sample) {
			if (!doStarted) break;
			
			if (record == null || record.output == null) continue;
			
			try {
				NeuronValue[] evaluated = null;
				if (record.input != null)
					evaluated = evaluate(record.input);
				else if ((record.undefinedInput != null) && (record.undefinedInput instanceof Raster))
					evaluated = evaluateByRaster((Raster)record.undefinedInput);
				
				if (evaluated == null) continue;
			} catch (Throwable e) {Util.trace(e);}
			
			if (fullNetwork != null) {
				try {
					Record bpRecord = new Record();
					bpRecord.input = convOutputLayer.getData();
					bpRecord.output = record.output;
					error = fullNetwork.bpLearn(sample, learningRate, terminatedThreshold, 1);
				} catch (Throwable e) {Util.trace(e);}
			}
				
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "conv_learn",
					"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			synchronized (this) {
				while (doPaused) {
					notifyAll();
					try {
						wait();
					} catch (Exception e) {Util.trace(e);}
				}
			}
				
		} //End for
		
		synchronized (this) {
			doStarted = false;
			doPaused = false;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "vae_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}


	/**
	 * Getting convolutional input layer.
	 * @return convolutional input layer.
	 */
	public ConvLayer getConvInputLayer() {
		if (convLayers.size() == 0)
			return null;
		else
			return convLayers.get(0);
	}
	
	
	/**
	 * Getting convolutional output layer.
	 * @return convolutional output layer.
	 */
	public ConvLayer getConvOutputLayer() {
		if (convLayers.size() == 0)
			return null;
		else
			return convLayers.get(convLayers.size() - 1);
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

	
	/**
	 * Getting source image type.
	 * @return source image type.
	 */
	private int getSourceImageType() {
		if (config.containsKey(Raster.SOURCE_IMAGE_TYPE_FIELD))
			return config.getAsInt(Raster.SOURCE_IMAGE_TYPE_FIELD);
		else
			return Raster.SOURCE_IMAGE_TYPE_DEFAULT;
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
	 * Getting whether source raster is resized.
	 * @return whether source raster is resized.
	 */
	private boolean isSourceRasterResize() {
		if (config.containsKey(Raster.SOURCE_RESIZE_FIELD))
			return config.getAsBoolean(Raster.SOURCE_RESIZE_FIELD);
		else
			return Raster.SOURCE_RESIZE_DEFAULT;
	}
	
	
}
