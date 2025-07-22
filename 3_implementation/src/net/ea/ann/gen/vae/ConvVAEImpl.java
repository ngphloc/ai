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

import net.ea.ann.conv.Content;
import net.ea.ann.conv.ConvSupporter;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.conv.filter.FilterFactory;
import net.ea.ann.conv.stack.StackNetworkAbstract;
import net.ea.ann.conv.stack.StackNetworkAssoc;
import net.ea.ann.core.Id;
import net.ea.ann.core.NetworkDoEvent.Type;
import net.ea.ann.core.NetworkDoEventImpl;
import net.ea.ann.core.NetworkStandard;
import net.ea.ann.core.Record;
import net.ea.ann.core.Util;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueCreator;
import net.ea.ann.gen.ConvGenModelAbstract;
import net.ea.ann.gen.ConvGenSetting;
import net.ea.ann.gen.FeatureGetter;
import net.ea.ann.gen.FeatureToX;
import net.ea.ann.gen.RasterUtility;
import net.ea.ann.raster.Cube;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.Size;

/**
 * This class represent an default implementation of convolutional Variational Autoencoders.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ConvVAEImpl extends VAEImpl implements ConvVAE, FeatureToX, FeatureGetter, ConvSupporter, RasterUtility {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Raster channel which is often larger than or equal to neuron channel.
	 */
	protected int rasterChannel = 1;
	
	
	/**
	 * Layer width.
	 */
	protected int width = 1;
	
	
	/**
	 * Layer height.
	 */
	protected int height = 1;

	
	/**
	 * Layer depth.
	 */
	protected int depth = 1;

	
	/**
	 * Layer time.
	 */
	protected int time = 1;

	
	/**
	 * Thick-stack property. In thick-stack mode (true), every stack in convolutional network should have more than one element layer.
	 */
	protected boolean thickStack = ConvGenSetting.THICK_STACK_DEFAULT;

	
	/**
	 * Convolutional network which can have fully connected network (FCN) or reversed fully connected network (RFCN) or both even.
	 */
	protected StackNetworkAbstract conv = null;

	
	/**
	 * Deconvolutional network which have neither fully connected network (FCN) nor reversed fully connected network (RFCN).
	 * Therefore, the deconvolutional network has only main convolutional network.
	 */
	protected StackNetworkAbstract deconv = null;
	
	
	/**
	 * Constructor with neuron channel, raster channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel which is often larger than or equal to neuron channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 */
	public ConvVAEImpl(int neuronChannel, int rasterChannel, Size size, Id idRef) {
		super(neuronChannel, null, idRef);
		this.rasterChannel = rasterChannel = ConvGenModelAbstract.fixRasterChannel(this.neuronChannel, rasterChannel);
		
		this.width = size.width;
		this.height = size.height;
		this.depth = size.depth;
		this.time = size.time;
		
		this.config.put(ConvGenModelAbstract.CONV_CLASSIFIER_FIELD, ConvGenModelAbstract.CONV_CLASSIFIER_DEFAULT);
	}

	
	/**
	 * Constructor with neuron channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 */
	public ConvVAEImpl(int neuronChannel, Size size, Id idRef) {
		this(neuronChannel, neuronChannel, size, idRef);
	}

	
	/**
	 * Constructor with neuron channel and size.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 */
	public ConvVAEImpl(int neuronChannel, Size size) {
		this(neuronChannel, neuronChannel, size, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public ConvVAEImpl(int neuronChannel) {
		this(neuronChannel, neuronChannel, Size.unit(), null);
	}

	
	@Override
	public void setSetting(ConvGenSetting setting) throws RemoteException {
		if (setting == null) return;
		
		this.width = setting.width;
		this.height = setting.height;
		this.depth = setting.depth;
		this.time = setting.time;
		this.thickStack = setting.thickStack;
	}
	

	@Override
	public ConvGenSetting getSetting() throws RemoteException {
		ConvGenSetting setting = new ConvGenSetting();
		setting.width = this.width;
		setting.height = this.height;
		setting.depth = this.depth;
		setting.time = this.time;
		setting.thickStack = this.thickStack;
		
		return setting;
	}
	
	
	/**
	 * Initialize with Z dimension as well as other specifications.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronEncode number of encoded hidden neurons.
	 * @param nHiddenNeuronDecode number of decoded hidden neurons.
	 * @param convFilterArrays arrays of convolutional filters. Filters in the same array have the same size.
	 * @param deconvFilterArrays deconvolutional filters. Filters in the same array have the same size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronEncode, int[] nHiddenNeuronDecode,
			Filter[][] convFilterArrays, Filter[][] deconvFilterArrays) {
		int xDim = 0;
		
		xDim = width*height*depth*time;
		if (convFilterArrays != null && convFilterArrays.length > 0) {
			conv = createConvNetwork();
			if (conv == null)
				return false;
			else if (thickStack) {
				if (!conv.initialize(new Size(width, height, depth, time), convFilterArrays)) return false;
			}
			else if (convFilterArrays.length == 1) {
				if (!conv.initialize(new Size(width, height, depth, time), convFilterArrays[0])) return false;
			}
			else {
				if (!conv.initialize(new Size(width, height, depth, time), convFilterArrays)) return false;
			}
			
			try {
				Size size = conv.getFeatureSize();
				xDim = size.width * size.height * size.depth * size.time;
			} catch (Throwable e) {Util.trace(e);}
		}
		else
			conv = null;
		
		int ratio = rasterChannel / neuronChannel;
		ratio = ratio < 1 ? 1 : ratio; 
		xDim = xDim * ratio;
		if(!super.initialize(xDim, zDim, nHiddenNeuronEncode, nHiddenNeuronDecode)) return false;

		if (deconvFilterArrays != null && deconvFilterArrays.length > 0) {
			Size deconvSize = new Size(width, height, depth, time);
			if (conv != null) {
				try {
					deconvSize = conv.getUnifiedOutputContentSize();
				} catch (Throwable e) {Util.trace(e);}
			}
			
			deconv = createDeconvNetwork();
			if (deconv == null)
				return false;
			else if (thickStack) {
				if (!deconv.initialize(deconvSize, deconvFilterArrays)) return false;
			}
			else if (deconvFilterArrays.length == 1) {
				if (!deconv.initialize(deconvSize, deconvFilterArrays[0])) return false;
			}
			else {
				if (!deconv.initialize(deconvSize, deconvFilterArrays)) return false;
			}
		}
		else
			deconv = null;
		
		return true;
	}

	
	/**
	 * Initialize with Z dimension as well as other specifications.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronEncode number of encoded hidden neurons.
	 * @param convFilterArrays arrays of convolutional filters. Filters in the same array have the same size.
	 * @param deconvFilterArrays deconvolutional filters. Filters in the same array have the same size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronEncode,
			Filter[][] convFilterArrays, Filter[][] deconvFilterArrays) {
		return initialize(zDim,
			nHiddenNeuronEncode,
			nHiddenNeuronEncode != null && nHiddenNeuronEncode.length > 0? reverse(nHiddenNeuronEncode) : null,
			convFilterArrays, deconvFilterArrays);
	}
	
	
	/**
	 * Initialize with Z dimension as well as other specifications.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronEncode number of encoded hidden neurons.
	 * @param convFilterArrays arrays of convolutional filters. Filters in the same array have the same size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronEncode,
			Filter[][] convFilterArrays) {
		return initialize(zDim, nHiddenNeuronEncode, convFilterArrays, (Filter[][])null);
	}

	
	@Override
	public boolean initialize(int zDim,
			Filter[][] convFilterArrays, Filter[][] deconvFilterArrays) throws RemoteException {
		int xDimTemp = Filter.calcLengthSimply(width*height*depth*time, convFilterArrays);
		if (xDimTemp == 0) return false;
		int[] nHiddenNeuronEncode = NetworkStandard.constructHiddenNeuronNumbers(xDimTemp, zDim, getHiddenLayerMin());
		return initialize(zDim, nHiddenNeuronEncode, convFilterArrays, deconvFilterArrays);
	}


	/**
	 * Initialize with Z dimension as well as other specifications.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronEncode number of encoded hidden neurons.
	 * @param nHiddenNeuronDecode number of decoded hidden neurons.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters deconvolutional filters.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronEncode, int[] nHiddenNeuronDecode,
			Filter[] convFilters, Filter[] deconvFilters) {
		Filter[][] convFilterArrays = null, deconvFilterArrays = null;
		if (convFilters != null && convFilters.length > 0) convFilterArrays = new Filter[][] {convFilters};
		if (deconvFilters != null && deconvFilters.length > 0) deconvFilterArrays = new Filter[][] {deconvFilters};
		return initialize(zDim, nHiddenNeuronEncode, nHiddenNeuronDecode, convFilterArrays, deconvFilterArrays);
	}
	
	
	/**
	 * Initialize with Z dimension as well as other specifications.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronEncode number of encoded hidden neurons.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters deconvolutional filters.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronEncode, 
			Filter[] convFilters, Filter[] deconvFilters) {
		return initialize(zDim,
			nHiddenNeuronEncode,
			nHiddenNeuronEncode != null && nHiddenNeuronEncode.length > 0? reverse(nHiddenNeuronEncode) : null,
			convFilters, deconvFilters);
	}

	
	/**
	 * Initialize with Z dimension as well as other specifications.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronEncode number of encoded hidden neurons.
	 * @param convFilters convolutional filters.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronEncode, 
			Filter[] convFilters) {
		return initialize(zDim, nHiddenNeuronEncode, convFilters, (Filter[])null);
	}

	
	/**
	 * Initialize with Z dimension and number of encoding hidden neuron.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronEncode number of encoded hidden neurons.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronEncode) {
		return initialize(zDim, nHiddenNeuronEncode, (Filter[])null);
	}


	@Override
	public boolean initialize(int zDim,
			Filter[] convFilters, Filter[] deconvFilters) throws RemoteException {
		int xDimTemp = Filter.calcLength(width*height*depth*time, convFilters);
		if (xDimTemp == 0) return false;
		int[] nHiddenNeuronEncode = NetworkStandard.constructHiddenNeuronNumbers(xDimTemp, zDim, getHiddenLayerMin());
		return initialize(zDim, nHiddenNeuronEncode, convFilters, deconvFilters);
	}
	
	
	/**
	 * Initialize with Z dimension and convolutional filters.
	 * @param zDim Z dimension
	 * @param convFilters convolutional filters.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim,
			Filter[] convFilters) {
		try {
			return initialize(zDim, convFilters, (Filter[])null);
		}
		catch (RemoteException e) {Util.trace(e);}
		
		return false;
	}

	
	/**
	 * Initialize with Z dimension with zooming out ratio.
	 * @param zDim Z dimension
	 * @param zoomOutRatio zooming out ratio.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim,
			int zoomOutRatio) {
		try {
			Filter[] convFilters = null;
			Filter[] deconvFilters = null;
			if (zoomOutRatio > 1) {
				FilterFactory factory = getFilterFactory();
				convFilters = new Filter[] {factory.zoomOut(zoomOutRatio, zoomOutRatio, zoomOutRatio)};
				deconvFilters = new Filter[] {factory.zoomIn(zoomOutRatio, zoomOutRatio, zoomOutRatio)};
			}
			
			return initialize(zDim, convFilters, deconvFilters);
		}
		catch (RemoteException e) {Util.trace(e);}
		
		return false;
	}

	
	/**
	 * Initialize with Z dimension.
	 * @param zDim Z dimension
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim) {
		return initialize(zDim, 1);
	}

	
	@Override
	public void reset() throws RemoteException {
		super.reset();
		conv = null;
		deconv = null;
	}


	/**
	 * Creating convolutional neural network.
	 * @return convolutional neural network.
	 */
	protected StackNetworkAbstract createConvNetwork() {
		return ConvGenModelAbstract.defaultConvNetwork(this, isNorm(), idRef);
	}
	
	
	/**
	 * Creating deconvolutional neural network.
	 * @return deconvolutional neural network.
	 */
	protected StackNetworkAbstract createDeconvNetwork() {
		return createConvNetwork();
	}


	@Override
	public NeuronValue[] convertFeatureToX(NeuronValue[] feature) {
		if (feature == null || feature.length == 0 || rasterChannel == neuronChannel)
			return feature;
		else
			return NeuronValue.flattenByChannel(feature, neuronChannel);
	}
	
	
	@Override
	public NeuronValue[] convertXToFeature(NeuronValue[] dataX) {
		if (dataX == null || dataX.length == 0 || rasterChannel == neuronChannel)
			return dataX;
		else
			return NeuronValue.aggregateByChannel(dataX, rasterChannel);
	}

	
	@Override
	public NeuronValueCreator getConvNeuronValueCreator() {
		return createConvNetwork().newStack(Size.unit());
	}

	
	@Override
	public FilterFactory getFilterFactory() {
		return createConvNetwork().getFilterFactory();
	}
	
	
	@Override
	public NeuronValue[] learnRasterOne(Iterable<Raster> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = config.getAsReal(LEARN_RATE_FIELD);
		return learnRasterOne(sample, learningRate, terminatedThreshold, maxIteration);
	}


	@Override
	public NeuronValue[] learnRaster(Iterable<Raster> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = config.getAsReal(LEARN_RATE_FIELD);
		return learnRaster(sample, learningRate, terminatedThreshold, maxIteration);
	}


	@Override
	protected NeuronValue[] learnOne(Iterable<Record> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		try {
			if (isDoStarted()) return null;
		} catch (Throwable e) {Util.trace(e);}
		
		if (encoder == null || encoder.getBackbone().size() < 2) return null;
		if (decoder == null || decoder.getBackbone().size() < 2) return null;
		
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
				
				//Learning decoder.
				try {
					error = decoder.learn(randomizeDataZ(learnRnd), input, lr, terminatedThreshold, 1);
				} catch (Throwable e) {Util.trace(e);}
				
				//It is unnecessary to learn the deconvolutional encoding network because the deconvolutional encoding network has neither full network nor reversed full network. 
			}
			
			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "convvae_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0 || (iteration >= maxIteration && maxIteration == 1))
				doStarted = false;
			else if (terminatedThreshold > 0 && config.isBooleanValue(LEARN_TERMINATE_ERROR_FIELD)) {
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
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "convvae_backpropogate",
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
		
		maxIteration = maxIteration >= 0 ? maxIteration :  LEARN_MAX_ITERATION_DEFAULT;
		terminatedThreshold = Double.isNaN(terminatedThreshold) || terminatedThreshold < 0 ? LEARN_TERMINATED_THRESHOLD_DEFAULT : terminatedThreshold;
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? LEARN_RATE_DEFAULT : learningRate;
		
		NeuronValue[] error = null;
		int iteration = 0;
		doStarted = true;
		while (doStarted && (maxIteration <= 0 || iteration < maxIteration)) {
			sample = resample(sample, iteration); //Re-sampling.
			double lr = calcLearningRate(learningRate, iteration);

			//Learning convolutional network.
			try {
				if (conv != null && ConvGenModelAbstract.hasLearning(conv))
					conv.learn(sample, lr, terminatedThreshold, 1);
			} catch (Throwable e) {Util.trace(e);}

			List<Record> encodeSample = Util.newList(0);
			for (Record record : sample) {
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
			
			List<Record> decodeSample = Util.newList(0);
			for (Record record : encodeSample) decodeSample.add(new Record(randomizeDataZ(learnRnd), record.input));
			//Learning decoder.
			error = decoder.learn(decodeSample, lr, terminatedThreshold, 1);
			
			//It is unnecessary to learn the deconvolutional encoding network because the deconvolutional encoding network has neither full network nor reversed full network. 

			iteration ++;
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.doing, "convvae_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));

			if (error == null || error.length == 0 || iteration >= maxIteration)
				doStarted = false;
			else if (terminatedThreshold > 0 && config.isBooleanValue(LEARN_TERMINATE_ERROR_FIELD)) {
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
			
			fireDoEvent(new NetworkDoEventImpl(this, Type.done, "convvae_backpropogate",
				"At final iteration " + iteration + "\nThe learned result is:\n" + this, iteration, maxIteration));
			
			notifyAll();
		}
		
		return error;
	}

	
	/**
	 * Learning neural network by back propagate algorithm with image specifications.
	 * @param sample learning sample. There is only inputs in the sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	private NeuronValue[] learnRasterOne(Iterable<Raster> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		return learnOne(RasterAssoc.toInputSample(sample), learningRate, terminatedThreshold, maxIteration);
	}

	
	/**
	 * Learning neural network by back propagate algorithm with image specifications.
	 * @param sample learning sample. There is only inputs in the sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	private NeuronValue[] learnRaster(Iterable<Raster> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		return learn(RasterAssoc.toInputSample(sample), learningRate, terminatedThreshold, maxIteration);
	}

	
	/**
	 * Converting X data to raster.
	 * @param dataX X data.
	 * @return raster converted from X data.
	 */
	private Raster convertXDataToRaster(NeuronValue[] dataX) {
		return ConvGenModelAbstract.convertXDataToRaster(dataX, conv, deconv, this, this);
	}
	
	
	@Override
	public synchronized G generateRaster() throws RemoteException {
		try {
			G g = generate();
			if (g == null || g.xgen == null) return null;
			g.xgenUndefined = convertXDataToRaster(g.xgen);
			return g;
		}
		catch (Throwable e) {Util.trace(e);}
		
		return null;
	}
	
	
	@Override
	public synchronized G generateRasterBest() throws RemoteException {
		try {
			G g = generateBest();
			if (g == null || g.xgen == null) return null;
			g.xgenUndefined = convertXDataToRaster(g.xgen);
			return g;
		}
		catch (Throwable e) {Util.trace(e);}
		
		return null;
	}


	@Override
	public G generateRaster(NeuronValue...dataZ) throws RemoteException {
		try {
			NeuronValue[] genX = generateByZ(dataZ);
			if (genX == null) return null;
			
			G g = new G();
			g.z = dataZ;
			g.xgen = genX;
			g.xgenUndefined = convertXDataToRaster(g.xgen);
			return g;
		}
		catch (Throwable e) {Util.trace(e);}
		
		return null;
	}


	@Override
	public synchronized G recover(NeuronValue[] dataX, Cube region, boolean random, boolean calcError) throws RemoteException {
		return ConvGenModelAbstract.recover(this, this, dataX, region, random, calcError);
	}


	@Override
	public G recoverRaster(Raster raster, Cube region, boolean random, boolean calcError) throws RemoteException {
		if (raster == null) return null;
		
		NeuronValue[] dataX = raster.toNeuronValues(neuronChannel, new Size(width, height, depth, time), isNorm());
		G g = recover(dataX, region, random, calcError);
		if (g == null) return null;
		
		if (g.xgen != null) {
			g.xgenUndefined = convertXDataToRaster(g.xgen);
		}
		return g;
	}


	@Override
	public G reproduceRaster(Raster raster, Cube region, boolean random, boolean calcError) throws RemoteException {
		return ConvGenModelAbstract.reproduceRaster(this, raster, region, random, calcError);
	}


	@Override
	public Size getFeatureSize() {
		if (conv == null) return new Size(width, height, depth, time);
		
		Size size = null;
		try {
			size = conv.getFeatureSize();
		} catch (Throwable e) {Util.trace(e);}
		
		return size != null ? size : new Size(width, height, depth, time);
	}
	
	
	@Override
	public Content getFeature() throws RemoteException {
		return conv != null ? conv.getFeatureFitChannel() : null;
	}

	
	@Override
	public Content getFeature(Raster raster) throws RemoteException {
		return conv != null ? new StackNetworkAssoc(conv).getFeature(raster) : null;
	}

	
	@Override
	public Raster createRaster(NeuronValue[] values) {
		return RasterAssoc.createRaster(convertXToFeature(values), rasterChannel, new Size(width, height, depth, time),
			isNorm(), getDefaultAlpha());
	}
	
	
	@Override
	public int getRasterChannel() throws RemoteException {
		return rasterChannel;
	}


	/**
	 * Creating convolutional VAE with neuron channel, raster channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 * @return Convolutional Variational Autoencoders.
	 */
	public static ConvVAEImpl create(int neuronChannel, int rasterChannel, Size size, Id idRef) {
		size.width = size.width < 1 ? 1 : size.width;
		size.height = size.height < 1 ? 1 : size.height;
		size.depth = size.depth < 1 ? 1 : size.depth;
		size.time = size.time < 1 ? 1 : size.time;
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		rasterChannel = rasterChannel < neuronChannel ? neuronChannel : rasterChannel;
		return new ConvVAEImpl(neuronChannel, rasterChannel, size, idRef);
	}

	
	/**
	 * Creating convolutional VAE with neuron channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 * @return Convolutional Variational Autoencoders.
	 */
	public static ConvVAEImpl create(int neuronChannel, Size size, Id idRef) {
		return create(neuronChannel, neuronChannel, size, idRef);
	}
	
	
	/**
	 * Creating convolutional VAE with neuron channel and size.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 * @return Convolutional Variational Autoencoders.
	 */
	public static ConvVAEImpl create(int neuronChannel, Size size) {
		return create(neuronChannel, neuronChannel, size, null);
	}

	
	/**
	 * Creating convolutional VAE with neuron channel and raster channel.
	 * @param neuronChannel neuron channel.
	 * @param rasterChannel raster channel.
	 * @return Convolutional Variational Autoencoders.
	 */
	public static ConvVAEImpl create(int neuronChannel, int rasterChannel) {
		return create(neuronChannel, rasterChannel, Size.unit(), null);
	}

	
	/**
	 * Creating convolutional VAE with neuron channel.
	 * @param neuronChannel neuron channel.
	 * @return Convolutional Variational Autoencoders.
	 */
	public static ConvVAEImpl create(int neuronChannel) {
		return create(neuronChannel, neuronChannel, Size.unit(), null);
	}

	
}
