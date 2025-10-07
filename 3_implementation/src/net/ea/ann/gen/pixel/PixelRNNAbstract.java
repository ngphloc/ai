/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen.pixel;

import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.NetworkDoEvent.Type;
import net.ea.ann.core.NetworkDoEventImpl;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.gen.ConvGenModelAbstract;
import net.ea.ann.raster.Cube;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Size;
import net.ea.ann.rnn.RecurrentNetwork;
import net.ea.ann.rnn.RecurrentNetworkAbstract;
import net.ea.ann.rnn.RecurrentNetworkImpl;
import net.ea.ann.rnn.lstm.LongShortTermMemoryImpl;

/**
 * This class is an abstract implementation of generative pixel recurrent neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class PixelRNNAbstract extends ConvGenModelAbstract implements PixelRNN {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Field of randomizing Z data.
	 */
	protected final static String RANDOM_ZDATA_FIELD = "pixrnn_random_zdata";
	
	
	/**
	 * Default value for field of randomizing Z data.
	 */
	protected final static boolean RANDOM_ZDATA_DEFAULT = false;

	
	/**
	 * Recurrent neural network.
	 */
	protected RecurrentNetworkImpl rnn = null;
	
	
	/**
	 * Z pixel.
	 */
	protected int zPixel = 1;
	
	
	/**
	 * Means of every inputs of recurrent neural network.
	 */
	protected List<NeuronValue[]> rnnInputMeans = Util.newList(0);

	
	/**
	 * Constructor with neuron channel, raster channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel which is often larger than or equal to neuron channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 */
	protected PixelRNNAbstract(int neuronChannel, int rasterChannel, Size size, Id idRef) {
		super(neuronChannel, rasterChannel, size, idRef);
		this.config.put(RANDOM_ZDATA_FIELD, RANDOM_ZDATA_DEFAULT);
		RecurrentNetworkAbstract.fillConfig(this.config);
	}

	
	/**
	 * Constructor with neuron channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 */
	protected PixelRNNAbstract(int neuronChannel, Size size, Id idRef) {
		this(neuronChannel, neuronChannel, size, idRef);
	}
	
	
	/**
	 * Constructor with neuron channel and size.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 */
	protected PixelRNNAbstract(int neuronChannel, Size size) {
		this(neuronChannel, neuronChannel, size, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	protected PixelRNNAbstract(int neuronChannel) {
		this(neuronChannel, neuronChannel, Size.unit(), null);
	}


	/**
	 * Creating recurrent neural network. Derived class can override this method.
	 * @return recurrent neural network.
	 */
	protected RecurrentNetworkImpl createRNN() {
		Function auxActivateRef = Raster.toReLUActivationRef(neuronChannel, isNorm());
		return new LongShortTermMemoryImpl(neuronChannel, activateRef, auxActivateRef, idRef);
	}
	
	
	@Override
	public void reset() throws RemoteException {
		super.reset();
		rnn = null;
	}

	
	/**
	 * Update configuration of recurrent neural network.
	 */
	private void updateRNNConfig() {
		if (rnn == null) return;
		try {
			rnn.getConfig().putAll(this.getConfig());
		} catch (Throwable e) {Util.trace(e);}
	}
	
	
	/**
	 * Updating means of every inputs of recurrent neural network.
	 * @param rnnSample specified sample.
	 */
	private void updateRNNInputMeans(List<List<Record>> rnnSample) {
		if (rnnSample.size() == 0 || isRandomZData()) return;
		for (List<Record> recordList : rnnSample) {
			for (int i = 0; i < recordList.size(); i++) {
				Record record = recordList.get(i);
				if (i < rnnInputMeans.size()) {
					NeuronValue[] sum = NeuronValue.add(rnnInputMeans.get(i), record.input);
					rnnInputMeans.set(i, sum);
				}
				else 
					rnnInputMeans.add(record.input);
			}
		}
		double r = 1.0 / (double)rnnSample.size();
		for (int i = 0; i < rnnInputMeans.size(); i++) {
			NeuronValue[] mean = NeuronValue.multiply(rnnInputMeans.get(i), r);
			rnnInputMeans.set(i, mean);
		}
	}
	
	
	@Override
	public boolean initialize(int xDim, int zDim, int[] nHiddenNeuronDecode, Size volume) {
		int neuronPerPixel = getNeuronPerPixel(); 
		zPixel = zDim / neuronPerPixel;
		zPixel = zPixel < 1 ? 1 : zPixel;
		
		rnn = createRNN();
		updateRNNConfig();
		if (!rnn.initialize(neuronPerPixel, neuronPerPixel, null, volume)) return false;
		zPixel = zPixel < rnn.length() ? zPixel : rnn.length();
		rnnInputMeans.clear();
		return true;
	}


	@Override
	protected NeuronValue[] learnOneByOne(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		if (rnn == null || rnn.length() < 1) return null;
		updateRNNConfig();
		rnnInputMeans.clear();
		
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
				
				NeuronValue[] input = null;
				if (record.input == null) {
					if (conv != null) {
						try {
							//Learning convolutional network.
							if (ConvGenModelAbstract.hasLearning(conv)) conv.learnOneByOne(Arrays.asList(record), lr, terminatedThreshold, 1);
							conv.evaluate(record);
							input = conv.getFeatureFitChannel().getData();
						} catch (Throwable e) {Util.trace(e);}
						if (input == null) continue;
						input = convertFeatureToX(input);
					}
					else if (record.getRasterInput() != null) {
						input = record.getRasterInput().toNeuronValues(rasterChannel, new Size(width, height, depth, time), isNorm());
						if (input == null) continue;
						input = convertFeatureToX(input);
					}
				}
				else
					input = record.input;
				
				List<Record> rnnRecord = convertXToRNNInputRecord(input);
				if (rnnRecord.size() > 0 ) {
					List<List<Record>> rnnSample = Arrays.asList(rnnRecord);
					error = rnn.learnOneByOne(rnnSample, lr, terminatedThreshold, 1);
					updateRNNInputMeans(rnnSample);
				}
			}
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "pixrnn_backpropogate",
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
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "pixrnn_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}

	
	@Override
	protected NeuronValue[] learn(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		if (rnn == null || rnn.length() < 1) return null;
		updateRNNConfig();
		rnnInputMeans.clear();
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			Iterable<Record> subsample = resample(sample, iteration, maxIteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration+1);

			//Learning convolutional network.
			try {
				if (conv != null && ConvGenModelAbstract.hasLearning(conv))
					conv.learn(subsample, lr, terminatedThreshold, 1);
			} catch (Throwable e) {Util.trace(e);}

			List<List<Record>> rnnSample = Util.newList(0);
			for (Record record : subsample) {
				if (record == null) continue;
				
				NeuronValue[] input = null;
				if (record.input == null) {
					if (conv != null) {
						try {
							conv.evaluate(record);
							input = conv.getFeatureFitChannel().getData();
						} catch (Throwable e) {Util.trace(e);}
						if (input == null) continue;
						input = convertFeatureToX(input);
					}
					else if (record.getRasterInput() != null) {
						input = record.getRasterInput().toNeuronValues(rasterChannel, new Size(width, height, depth, time), isNorm());
						if (input == null) continue;
						input = convertFeatureToX(input);
					}
				}
				else
					input = record.input;
				
				List<Record> rnnRecord = convertXToRNNInputRecord(input);
				if (rnnRecord.size() > 0 ) rnnSample.add(rnnRecord);
			}
			
			if (rnnSample.size() > 0 ) {
				error = rnn.learn(rnnSample, lr, terminatedThreshold, 1);
				updateRNNInputMeans(rnnSample);
			}

			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "pixrnn_backpropogate",
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
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "pixrnn_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}


	@Override
	public G generate() throws RemoteException {
		if (rnn == null || rnn.length() < 1) return null;
		
		int neuronPerPixel = getNeuronPerPixel();
		List<NeuronValue[]> rnnInput = Util.newList(zPixel);
		List<NeuronValue> zInput = Util.newList(zPixel*neuronPerPixel);
		
		//Setting z data.
		if (rnnInputMeans.size() == 0 || zPixel > rnnInputMeans.size() || isRandomZData()) {
			//Randomizing z data.
			NeuronValue zero = rnn.get(0).getInputLayer().newNeuronValue().zero();
			Random rnd = new Random();
			for (int i = 0; i < zPixel; i++) {
				NeuronValue[] values = new NeuronValue[neuronPerPixel];
				for (int j = 0; j < neuronPerPixel; j++) {
					NeuronValue value = zero.valueOf(Util.randomGaussian(rnd)); 
					values[j] = value;
					zInput.add(value);
				}
				rnnInput.add(values);
			}
		}
		else {
			//Setting z data from means.
			for (int i = 0; i < zPixel; i++) {
				NeuronValue[] values = rnnInputMeans.get(i);
				if (values.length != neuronPerPixel) throw new RuntimeException("Wrong number of neurons per pixels.");
				for (NeuronValue value : values) zInput.add(value);
				rnnInput.add(values);
			}
		}
		//Evaluating based on z data to generate x data.
		try {
			rnn.evaluate(rnnInput);
		} catch (Throwable e) {Util.trace(e);}
		
		G g = new G();
		g.z = zInput.toArray(new NeuronValue[] {});
		g.xgen = convertRNNOutputToX();
		return g;
	}


	@Override
	public G generateBest() throws RemoteException {
		return generate();
	}


	@Override
	protected NeuronValue[] generateByZ(NeuronValue...dataZ) {
		try {
			rnn.evaluate(dataZ);
		} catch (Throwable e) {}
		
		return convertRNNOutputToX();
	}

	
	/*
	 * This method will be improved in the next version. Please pay attention to method generate() with only modifying RNN input.
	 */
	@Override
	public G recover(NeuronValue[] dataX, Cube region, boolean random, boolean calcError) throws RemoteException {
		return super.recover(dataX, region, random, calcError);
	}


	/**
	 * Getting number of neurons per pixel.
	 * @return number of neurons per pixel.
	 */
	private int getNeuronPerPixel() {
		return rasterChannel / neuronChannel;
	}
	
	
	/**
	 * Converting X data to recurrent neural network data.
	 * @param dataX X data.
	 * @return recurrent neural network data.
	 */
	private List<NeuronValue[]> convertXToRNNInput(NeuronValue[] dataX) {
		List<NeuronValue[]> rnnInput = Util.newList(0);
		if (rnn == null || rnn.length() < 1) return rnnInput;
		
		int neuronPerPixel = getNeuronPerPixel();
		if (neuronPerPixel != rnn.get(0).getInputLayer().size()) throw new RuntimeException("Converting X data to RNN input causes error.");
		for (int i = 0; i < dataX.length; i += neuronPerPixel) {
			NeuronValue[] values = Arrays.copyOfRange(dataX, i, i+neuronPerPixel);
			rnnInput.add(values);
			if (rnnInput.size() >= rnn.length()) break;
		}
		return rnnInput;
	}

	
	/**
	 * Converting X data to recurrent neural network record.
	 * @param dataX X data.
	 * @return recurrent neural network record.
	 */
	private List<Record> convertXToRNNInputRecord(NeuronValue[] dataX) {
		List<NeuronValue[]> rnnInput = convertXToRNNInput(dataX);
		List<Record> rnnRecordInput = Util.newList(rnnInput.size());
		for (NeuronValue[] values : rnnInput) {
			Record record = new Record(values, values);
			rnnRecordInput.add(record);
		}
		return rnnRecordInput;
	}

	
	/**
	 * Converting RNN output to X data;
	 * @return X data.
	 */
	private NeuronValue[] convertRNNOutputToX() {
		int neuronPerPixel = getNeuronPerPixel();
		if (neuronPerPixel != rnn.get(0).getOutputLayer().size()) throw new RuntimeException("Converting RNN output to X data causes error.");
		NeuronValue[] dataX = new NeuronValue[neuronPerPixel*rnn.length()];
		for (int i = 0; i < rnn.length(); i++) {
			LayerStandard outputLayer = rnn.get(i).getOutputLayer();
			for (int j = 0; j < neuronPerPixel; j++) {
				int index = i*neuronPerPixel + j;
				dataX[index] = RecurrentNetwork.verifyNonvector(outputLayer.get(j).getOutput());
			}
		}
		return dataX;
	}


	/**
	 * Checking whether randomizing Z data.
	 * @return whether randomizing Z data.
	 */
	private boolean isRandomZData() {
		return config.getAsBoolean(RANDOM_ZDATA_FIELD);
	}
	
	
}
