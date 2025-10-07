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
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkDoEventImpl;
import net.ea.ann.core.NetworkStandard;
import net.ea.ann.core.NetworkStandardImpl;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.NetworkDoEvent.Type;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Raster2D;
import net.ea.ann.raster.Size;

/**
 * This class represents the extended convolutional stack network which supports to convert feature to raster.
 * In other words, this class supports to reverse the main convolutional network {@link StackNetworkAbstract#stacks}.
 * Be careful to use this stack network because converting feature to raster consumes a lot of memory.
 * @author Loc Nguyen
 * @version 1.0
 */
public class StackNetworkRaster extends StackNetworkImpl {

	
	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * The neural network to convert unified content to raster.
	 * This network is the reverse of the main convolutional network {@link StackNetworkAbstract#stacks}.
	 */
	protected NetworkStandardImpl unifiedContentToRaster = null;
	
	
	/**
	 * Constructor with neuron channel, activation functions, and ID reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function, which is often activation function related to weights like sigmod function.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 * @param idRef ID reference.
	 */
	public StackNetworkRaster(int neuronChannel, Function activateRef, Function contentActivateRef, Id idRef) {
		super(neuronChannel, activateRef, contentActivateRef, idRef);
		this.config.put(Raster2D.LEARN_FIELD, Raster2D.LEARN_DEFAULT);
	}

	
	/**
	 * Constructor with neuron channel and activation functions.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function, which is often activation function related to weights like sigmod function.
	 * @param contentActivateRef activation function of content, which is often activation function relate to convolutional pixel like ReLU function.
	 */
	public StackNetworkRaster(int neuronChannel, Function activateRef, Function contentActivateRef) {
		this(neuronChannel, activateRef, contentActivateRef, null);
	}

	
	@Override
	public void reset() {
		super.reset();
		unifiedContentToRaster = null;
	}
	
	
	@Override
	public boolean initialize(Size size, Filter[] filters,
			int[] nFullHiddenOutputNeuron,
			boolean initReverse) {
		if(!super.initialize(size, filters, nFullHiddenOutputNeuron, initReverse)) return false;
		return initializeUnifiedContentToRasterNetwork();
	}

	
	@Override
	public boolean initialize(Size size,
			Filter[][] filterArrays,
			int[] nFullHiddenOutputNeuron,
			boolean initReverse) {
		if (!super.initialize(size, filterArrays, nFullHiddenOutputNeuron, initReverse)) return false;
		return initializeUnifiedContentToRasterNetwork();
	}
	
	
	/**
	 * Initializing unified content-to-raster network.
	 * @return true if initialization is successful.
	 */
	private boolean initializeUnifiedContentToRasterNetwork() {
		if (!config.getAsBoolean(Raster2D.LEARN_FIELD)) {
			unifiedContentToRaster = null;
			return true;
		}
		
		Content inputContent = getOriginalContent();
		if (inputContent == null) return false;
		Content unifiedContent = getUnifiedOutputContent();
		if (unifiedContent == null) return false;
		
		unifiedContentToRaster = new NetworkStandardImpl(neuronChannel, contentActivateRef);
		boolean initialized = unifiedContentToRaster.initialize(
			unifiedContent.length(),
			inputContent.length(),
			NetworkStandard.constructHiddenNeuronNumbers(unifiedContent.length(), inputContent.length(), getHiddenLayerMin()));
		try {
			if (initialized) unifiedContentToRaster.addListener(this);
		} catch (Throwable e) {Util.trace(e);}
		return initialized;
	}
	
	
	/**
	 * Getting original content which is often the input content.
	 * @return original content which is often the first content of the input stack of the convolutional network which is unified into one content.
	 * This method can be improved in the next version. 
	 */
	private Content getOriginalContent() {
		if (stacks.size() == 0)
			return null;
		else if (stacks.get(0).size() == 0)
			return null;
		else
			return stacks.get(0).get(0).getContent();
	}

	
	@Override
	public Raster createRaster(NeuronValue[] feature) {
		if (unifiedContentToRaster == null || stacks.size() == 0) return super.createRaster(feature);
		NeuronValue[] contentData = convertFeatureToUnifiedContentData(feature);
		if (contentData == null) return null;
		
		NeuronValue[] rasterData = null;
		try {
			rasterData = unifiedContentToRaster.evaluate(new Record(contentData));
			return getOriginalContent().createRaster(rasterData, isNorm(), getDefaultAlpha());
		} catch (Throwable e) {Util.trace(e);}
		
		return null;
	}

	
	@Override
	public NeuronValue[] learnOneByOne(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		NeuronValue[] error0 = super.learnOneByOne(sample, learningRate, terminatedThreshold, maxIteration);

		if (unifiedContentToRaster == null || stacks.size() == 0) return error0;
		try {
			if (isDoStarted()) return error0;
		} catch (Throwable e) {Util.trace(e);}
		
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
				
				//Learning the reverse of the main convolutional neural network.
				try {
					error = null;
					Record newRecord = new Record();
					newRecord.input = getUnifiedOutputContent().getData();
					newRecord.output = getOriginalContent().getData();
					error = unifiedContentToRaster.learn(newRecord.input, newRecord.output, lr, terminatedThreshold, 1);
				} catch (Throwable e) {Util.trace(e);}
			} //End for
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "stacknn_raster_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else {

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
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "stacknn_raster_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}

		return error0;
	}

	
	@Override
	public NeuronValue[] learn(Iterable<Record> sample, double learningRate, double terminatedThreshold,
			int maxIteration) {
		NeuronValue[] error = super.learn(sample, learningRate, terminatedThreshold, maxIteration);
		
		if (unifiedContentToRaster == null || stacks.size() == 0) return error;
		try {
			if (isDoStarted()) return error;
		} catch (Throwable e) {Util.trace(e);}
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			Iterable<Record> subsample = resample(sample, iteration, maxIteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration+1);

			List<Record> newSample = Util.newList(0);
			for (Record record : subsample) {
				if (record == null) continue;
				
				//Evaluating layers.
				try {
					evaluate(record);
				} catch (Throwable e) {Util.trace(e);}
				
				Record newRecord = new Record();
				newRecord.input = getUnifiedOutputContent().getData();
				newRecord.output = getOriginalContent().getData();
				newSample.add(newRecord);
			} //End for
			
			//Learning the reverse of the main convolutional neural network.
			unifiedContentToRaster.learn(newSample, lr, terminatedThreshold, 1);
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "stacknn_raster_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else {

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
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "stacknn_raster_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}

		return error;
	}

	
	/**
	 * Getting minimum number of hidden layers.
	 * @return minimum number of hidden layers.
	 */
	private int getHiddenLayerMin() {
		if (!config.containsKey(HIDDEN_LAYER_MIN_FILED)) return HIDDEN_LAYER_MIN_DEFAULT;
		int hiddenMin = config.getAsInt(HIDDEN_LAYER_MIN_FILED);
		return hiddenMin < HIDDEN_LAYER_MIN_DEFAULT ? HIDDEN_LAYER_MIN_DEFAULT : hiddenMin;
	}


}
