/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen.vae;

import java.rmi.RemoteException;
import java.util.Arrays;
import java.util.List;

import net.ea.ann.conv.filter.Filter;
import net.ea.ann.conv.stack.StackNetworkInitializer;
import net.ea.ann.core.Id;
import net.ea.ann.core.LayerStandard;
import net.ea.ann.core.NetworkDoEvent.Type;
import net.ea.ann.core.NetworkDoEventImpl;
import net.ea.ann.core.NetworkStandardImpl;
import net.ea.ann.core.NeuronStandard;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.generator.GeneratorStandard;
import net.ea.ann.core.generator.Trainer;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.gen.ConvGenModelAbstract;
import net.ea.ann.gen.gan.AdversarialNetwork;
import net.ea.ann.raster.Size;

/**
 * This class implements the Adversarial Variational Autoencoders (AVA).
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class AVA extends ConvVAEImpl {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Internal adversarial network.
	 */
	protected AdversarialNetwork decodeAdv = null;

	
	/**
	 * Constructor with neuron channel, raster channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 */
	public AVA(int neuronChannel, int rasterChannel, Size size, Id idRef) {
		super(neuronChannel, rasterChannel, size, idRef);
	}

	
	/**
	 * Constructor with neuron channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 */
	public AVA(int neuronChannel, Size size, Id idRef) {
		this(neuronChannel, neuronChannel, size, idRef);
	}
	
	
	/**
	 * Constructor with neuron channel and size.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 */
	public AVA(int neuronChannel, Size size) {
		this(neuronChannel, neuronChannel, size, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel or depth.
	 */
	public AVA(int neuronChannel) {
		this(neuronChannel, neuronChannel, new Size(1, 1, 1, 1), null);
	}

	
	/**
	 * Creating adversarial network.
	 * @return adversarial network.
	 */
	protected AdversarialNetwork createAdversarialNetwork() {
		return new AdversarialNetwork(neuronChannel, activateRef, idRef);
	}
	
	
	@Override
	protected NetworkStandardImpl createDecoder() {
		GeneratorStandard<?> generator = new GeneratorStandard<Trainer>(neuronChannel, activateRef, idRef) {
			
			/**
			 * Serial version UID for serializable class. 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			protected NeuronValue calcOutputError2(NeuronStandard outputNeuron, NeuronValue realOutput, LayerStandard outputLayer, int outputNeuronIndex, NeuronValue[] realOutputs, Object...params) {
				NeuronValue error = super.calcOutputError2(outputNeuron, realOutput, outputLayer, outputNeuronIndex, realOutputs, params);
				NeuronValue errorAdv = AdversarialNetwork.calcDecodedErrorAdv(outputNeuron, decodeAdv);
				return error.add(errorAdv);
			}
			
		};
		generator.setParent(this);
		return generator;
	}


	/**
	 * Initialize with X dimension and Z dimension as well as other specifications.
	 * @param xDim X dimension.
	 * @param zDim Z dimension where z is random data to generate data X.
	 * @param nHiddenNeuronEncode number of encoded hidden neurons.
	 * @param nHiddenNeuronDecode number of decoded hidden neurons.
	 * @param nHiddenNeuronAdversarial number of adversarial hidden neurons.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int xDim, int zDim, int[] nHiddenNeuronEncode, int[] nHiddenNeuronDecode, int[] nHiddenNeuronAdversarial) {
		if (xDim <= 0 || zDim <= 0) return false;
		
		this.encoder = createEncoder();
		if(!this.encoder.initialize(xDim, isVarXDiagonal() ? 2*zDim : zDim*(zDim+1), nHiddenNeuronEncode)) return false;
		
		LayerStandard encodeLayer = this.encoder.getOutputLayer();
		this.muX = new NeuronStandard[zDim];
		for (int i = 0; i < zDim; i++) {
			this.muX[i] = encodeLayer.get(i);
		}
		
		this.varX = new NeuronStandard[zDim][];
		NeuronValue zero = encodeLayer.newNeuronValue().zero();
		NeuronStandard zeroNeuron = encodeLayer.newNeuron();
		zeroNeuron.setInput(zero);
		zeroNeuron.setOutput(zero);
		for (int i = 0; i < zDim; i++) {
			this.varX[i] = new NeuronStandard[zDim];
			for (int j = 0; j < zDim; j++) {
				if (isVarXDiagonal())
					this.varX[i][j] = i == j ? encodeLayer.get(zDim + i) : zeroNeuron;
				else
					this.varX[i][j] = encodeLayer.get(zDim + i*zDim + j);
			}
		}

		this.decoder = createDecoder();
		if(!this.decoder.initialize(zDim, xDim, nHiddenNeuronDecode)) return false;
		
		if (nHiddenNeuronAdversarial == null || nHiddenNeuronAdversarial.length == 0) return true;
		
		this.decodeAdv = createAdversarialNetwork();
		if (!this.decodeAdv.initialize(xDim, 1, nHiddenNeuronAdversarial)) return false;

		return true;
	}

	
	@Override
	public boolean initialize(int xDim, int zDim, int[] nHiddenNeuronEncode, int[] nHiddenNeuronDecode) {
		return this.initialize(xDim, zDim, nHiddenNeuronEncode, nHiddenNeuronDecode, nHiddenNeuronEncode);
	}


	@Override
	public boolean initialize(int zDim, int[] nHiddenNeuronEncode, int[] nHiddenNeuronDecode,
			Filter[] convFilters, Filter[] deconvFilters) {
		int xDim = 0;
		
		xDim = width*height*depth*time;
		if (convFilters != null && convFilters.length > 0) {
			conv = createConvNetwork();
			if (conv == null)
				return false;
			else if (!new StackNetworkInitializer(conv).initialize(new Size(width, height, depth, time), convFilters))
				return false;
			
			try {
				Size size = conv.getFeatureSize();
				xDim = size.width*size.height*size.depth * time;
			} catch (Throwable e) {Util.trace(e);}
		}
		int ratio = rasterChannel / neuronChannel;
		ratio = ratio < 1 ? 1 : ratio; 
		xDim = xDim * ratio;

		if(!this.initialize(xDim, zDim, nHiddenNeuronEncode, nHiddenNeuronDecode))
			return false;

		if (deconvFilters != null && deconvFilters.length > 0) {
			Size deconvSize = new Size(width, height, depth, time);
			if (conv != null) {
				try {
					deconvSize = conv.getUnifiedOutputContentSize();
				} catch (Throwable e) {Util.trace(e);}
			}
			
			deconv = createDeconvNetwork();
			if (deconv == null)
				return false;
			else if (!new StackNetworkInitializer(deconv).initialize(deconvSize, deconvFilters))
				return false;
		}
		
		return true;
	}

	
	@Override
	public void reset() throws RemoteException {
		super.reset();
		decodeAdv = null;
	}


	@Override
	protected NeuronValue[] learnOne(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		if (encoder == null || encoder.getBackbone().size() < 2) return null;
		if (decoder == null || decoder.getBackbone().size() < 2) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			Iterable<Record> subsample = resample(sample, iteration, maxIteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration);

			for (Record record : subsample) {
				if (record == null) continue;
				
				NeuronValue[] input = null;
				if (record.input == null) {
					if (conv != null) {
						try {
							//Learning convolutional network.
							if (ConvGenModelAbstract.hasLearning(conv)) conv.learnOne(Arrays.asList(record), lr, terminatedThreshold, 1);
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
				
				//Learning encoder.
				try {
					encoder.learn(input, lr, terminatedThreshold, 1);
				} catch (Throwable e) {Util.trace(e);}
				
				NeuronValue[] dataZ = randomizeDataZ(learnRnd);
				
				//Learning decoder.
				try {
					error = decoder.learn(dataZ, input, lr, terminatedThreshold, 1);
				} catch (Throwable e) {Util.trace(e);}
				
				//Learning adversarial network.
				if (decodeAdv != null) {
					try {
						Record newRecord = new Record(input);
						decodeAdv.setPrevOutput(decodeAdv.evaluate(newRecord));
						
						//Getting generated X
						NeuronValue[] generatedX = decoder.evaluate(new Record(dataZ));
						
						//Learning adversarial network.
						decodeAdv.learn(generatedX, lr, terminatedThreshold, 1);
					} catch (Throwable e) {Util.trace(e);}
					
					try {
						decodeAdv.setPrevOutput(null);
					} catch (Throwable e) {Util.trace(e);}
				}
				
				//It is unnecessary to learn the deconvolutional encoding network because the deconvolutional encoding network has neither full network nor reversed full network. 
			}
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "ava_backpropogate",
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
		
		//Fix something
		adjustVarX();

		synchronized (this) {
			doStarted = false;
			doPaused = false;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "ava_backpropogate",
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
		
		if (encoder == null || encoder.getBackbone().size() < 2) return null;
		if (decoder == null || decoder.getBackbone().size() < 2) return null;
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_MAX;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			Iterable<Record> subsample = resample(sample, iteration, maxIteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration);

			//Learning convolutional network.
			try {
				if (conv != null && ConvGenModelAbstract.hasLearning(conv))
					conv.learn(subsample, lr, terminatedThreshold, 1);
			} catch (Throwable e) {Util.trace(e);}

			List<Record> encodeSample = Util.newList(0);
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
				
				encodeSample.add(new Record(input, null));
			}
			
			if (encodeSample.size() == 0) break;
			//Learning encoder.
			encoder.learn(encodeSample, lr, terminatedThreshold, 1);
			
			//Learning decoder.
			List<Record> decodeSample = Util.newList(encodeSample.size());
			for (Record encodeRecord : encodeSample) decodeSample.add(new Record(randomizeDataZ(learnRnd), encodeRecord.input));
			error = decoder.learn(decodeSample, lr, terminatedThreshold, 1);
			
			//Learning adversarial network.
			if (decodeAdv != null) {
				List<Record> decodeAdvSample = Util.newList(decodeSample.size());
				int n = 0;
				for (Record decodeAdvRecord : decodeSample) {
					if(decodeAdv.evaluateSetPrevOutputAccum(new Record(decodeAdvRecord.output))) n++;
					
					try {
						NeuronValue[] generatedX = decoder.evaluate(new Record(decodeAdvRecord.input));
						decodeAdvSample.add(new Record(generatedX, null));
					} catch (Throwable e) {Util.trace(e);}
				}
				NeuronValue[] prevOutput = decodeAdv.getPrevOutput();
				if (prevOutput != null && n > 0) {
					for (int i = 0; i < prevOutput.length; i++) prevOutput[i] = prevOutput[i].divide(n);
					decodeAdv.setPrevOutput(prevOutput);
				}
				//Learning adversarial network.
				decodeAdv.learn(decodeAdvSample, lr, terminatedThreshold, 1);
				decodeAdv.setPrevOutput(null);
			}
			
			//It is unnecessary to learn the deconvolutional encoding network because the deconvolutional encoding network has neither full network nor reversed full network. 

			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "ava_backpropogate",
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
		
		//Fix something
		adjustVarX();

		synchronized (this) {
			doStarted = false;
			doPaused = false;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "ava_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}

	
	/**
	 * Creating AVA with neuron channel, raster channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 * @return Adversarial Variational Autoencoders (AVA).
	 */
	public static AVA create(int neuronChannel, int rasterChannel, Size size, Id idRef) {
		size.width = size.width < 1 ? 1 : size.width;
		size.height = size.height < 1 ? 1 : size.height;
		size.depth = size.depth < 1 ? 1 : size.depth;
		size.time = size.time < 1 ? 1 : size.time;
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		rasterChannel = rasterChannel < neuronChannel ? neuronChannel : rasterChannel;
		return new AVA(neuronChannel, rasterChannel, size, idRef);
	}

	
	/**
	 * Creating AVA with neuron channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 * @return Adversarial Variational Autoencoders (AVA).
	 */
	public static AVA create(int neuronChannel, Size size, Id idRef) {
		return create(neuronChannel, neuronChannel, size, idRef);
	}

	
	/**
	 * Creating AVA with neuron channel and size.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 * @return Adversarial Variational Autoencoders (AVA).
	 */
	public static AVA create(int neuronChannel, Size size) {
		return create(neuronChannel, neuronChannel, size, null);
	}

	
	/**
	 * Creating AVA with neuron channel and raster channel.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @return Adversarial Variational Autoencoders (AVA).
	 */
	public static AVA create(int neuronChannel, int rasterChannel) {
		return create(neuronChannel, rasterChannel, Size.unit(), null);
	}

	
	/**
	 * Creating AVA with neuron channel.
	 * @param neuronChannel neuron channel.
	 * @return Adversarial Variational Autoencoders (AVA).
	 */
	public static AVA create(int neuronChannel) {
		return create(neuronChannel, neuronChannel, Size.unit(), null);
	}

	
}
