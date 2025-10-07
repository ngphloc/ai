/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen;

import java.rmi.RemoteException;
import java.util.Arrays;

import net.ea.ann.classifier.Classifier;
import net.ea.ann.classifier.StackClassifier;
import net.ea.ann.conv.Content;
import net.ea.ann.conv.ConvSupporter;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.conv.filter.FilterFactory;
import net.ea.ann.conv.stack.StackNetworkAbstract;
import net.ea.ann.conv.stack.StackNetworkAssoc;
import net.ea.ann.conv.stack.StackNetworkImpl;
import net.ea.ann.conv.stack.StackNetworkInitializer;
import net.ea.ann.core.Id;
import net.ea.ann.core.Network;
import net.ea.ann.core.NetworkConfig;
import net.ea.ann.core.NetworkStandard;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueCreator;
import net.ea.ann.raster.Cube;
import net.ea.ann.raster.Image;
import net.ea.ann.raster.Point;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Raster2D;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.Size;

/**
 * This class is the abstract implementation of convolutional generative model.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class ConvGenModelAbstract extends GenModelAbstract implements ConvGenModel, FeatureToX, FeatureGetter, ConvSupporter, RasterUtility {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Classifier field.
	 */
	public final static String CONV_CLASSIFIER_FIELD = "convgm_classifier";
	
	
	/**
	 * Default value for classifier field.
	 */
	public final static boolean CONV_CLASSIFIER_DEFAULT = false;

	
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
	 * @param rasterChannel raster channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 */
	protected ConvGenModelAbstract(int neuronChannel, int rasterChannel, Size size, Id idRef) {
		super(neuronChannel, null, idRef);
		this.rasterChannel = rasterChannel = RasterAssoc.fixRasterChannel(this.neuronChannel, rasterChannel);

		this.width = size.width;
		this.height = size.height;
		this.depth = size.depth;
		this.time = size.time;
	}
	

	/**
	 * Constructor with neuron channel, size, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 * @param idRef identifier reference.
	 */
	protected ConvGenModelAbstract(int neuronChannel, Size size, Id idRef) {
		this(neuronChannel, neuronChannel, size, idRef);
	}
	
	
	/**
	 * Constructor with neuron channel and size.
	 * @param neuronChannel neuron channel.
	 * @param size layer size.
	 */
	protected ConvGenModelAbstract(int neuronChannel, Size size) {
		this(neuronChannel, neuronChannel, size, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	protected ConvGenModelAbstract(int neuronChannel) {
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


	@Override
	public void reset() throws RemoteException {
		conv = null;
		deconv = null;
	}

	
	/**
	 * Initialize with X dimension and Z dimension as well as hidden neurons.
	 * @param xDim X dimension.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronDecode number of decoded hidden neurons.
	 * @param volume volume.
	 * @return true if initialization is successful.
	 */
	public abstract boolean initialize(int xDim, int zDim, int[] nHiddenNeuronDecode, Size volume);
	
	
	/**
	 * Initialize with Z dimension as well as other specifications.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronDecode number of decoded hidden neurons.
	 * @param convFilterArrays arrays of convolutional filters. Filters in the same array have the same size.
	 * @param deconvFilterArrays deconvolutional filters. Filters in the same array have the same size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronDecode,
			Filter[][] convFilterArrays, Filter[][] deconvFilterArrays) {
		int xDim = 0;
		
		Size size = new Size(width, height, depth, time);
		xDim = width*height*depth*time;
		if (convFilterArrays != null && convFilterArrays.length > 0) {
			conv = createConvNetwork();
			if (conv == null)
				return false;
			else if (thickStack) {
				if (!new StackNetworkInitializer(conv).initialize(new Size(width, height, depth, time), convFilterArrays)) return false;
			}
			else if (convFilterArrays.length == 1) {
				if (!new StackNetworkInitializer(conv).initialize(new Size(width, height, depth, time), convFilterArrays[0])) return false;
			}
			else {
				if (!new StackNetworkInitializer(conv).initialize(new Size(width, height, depth, time), convFilterArrays)) return false;
			}
			
			try {
				size = conv.getFeatureSize();
				xDim = size.width * size.height * size.depth * size.time;
			} catch (Throwable e) {Util.trace(e);}
		}
		else
			conv = null;
		
		int ratio = rasterChannel / neuronChannel;
		ratio = ratio < 1 ? 1 : ratio; 
		xDim = xDim * ratio;
		if(!initialize(xDim, zDim, nHiddenNeuronDecode, size)) return false;

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
				if (!new StackNetworkInitializer(deconv).initialize(deconvSize, deconvFilterArrays)) return false;
			}
			else if (deconvFilterArrays.length == 1) {
				if (!new StackNetworkInitializer(deconv).initialize(deconvSize, deconvFilterArrays[0])) return false;
			}
			else {
				if (!new StackNetworkInitializer(deconv).initialize(deconvSize, deconvFilterArrays)) return false;
			}
		}
		else
			deconv = null;
		
		return true;
	}

	
	@Override
	public boolean initialize(int zDim,
			Filter[][] convFilterArrays, Filter[][] deconvFilterArrays) throws RemoteException {
		int xDimTemp = Filter.calcLengthSimply(width*height*depth*time, convFilterArrays);
		if (xDimTemp == 0) return false;
		return initialize(zDim,
			NetworkStandard.constructHiddenNeuronNumbers(zDim, xDimTemp, getHiddenLayerMin()),
			convFilterArrays, deconvFilterArrays);
	}
	
	
	/**
	 * Initialize with Z dimension as well as other specifications.
	 * @param zDim Z dimension
	 * @param convFilterArrays arrays of convolutional filters. Filters in the same array have the same size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim,
			Filter[][] convFilterArrays) {
		try {
			return initialize(zDim, convFilterArrays, (Filter[][])null);
		} catch (Throwable e) {Util.trace(e);}
		return false;
	}

	
	/**
	 * Initialize with Z dimension as well as other specifications.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronDecode number of decoded hidden neurons.
	 * @param convFilters convolutional filters.
	 * @param deconvFilters deconvolutional filters.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronDecode,
			Filter[] convFilters, Filter[] deconvFilters) {
		Filter[][] convFilterArrays = null, deconvFilterArrays = null;
		if (convFilters != null && convFilters.length > 0) convFilterArrays = new Filter[][] {convFilters};
		if (deconvFilters != null && deconvFilters.length > 0) deconvFilterArrays = new Filter[][] {deconvFilters};
		return initialize(zDim, nHiddenNeuronDecode, convFilterArrays, deconvFilterArrays);
	}
	
	
	/**
	 * Initialize with Z dimension as well as other specifications.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronDecode number of decoded hidden neurons.
	 * @param convFilters convolutional filters.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronDecode, 
			Filter[] convFilters) {
		return initialize(zDim, nHiddenNeuronDecode, convFilters, (Filter[])null);
	}

	
	/**
	 * Initialize with Z dimension as well as other specifications.
	 * @param zDim Z dimension
	 * @param nHiddenNeuronDecode number of decoded hidden neurons.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(int zDim, int[] nHiddenNeuronDecode) {
		return initialize(zDim, nHiddenNeuronDecode, (Filter[])null);
	}

	
	@Override
	public boolean initialize(int zDim,
			Filter[] convFilters, Filter[] deconvFilters) throws RemoteException {
		int xDimTemp = Filter.calcLength(width*height*depth*time, convFilters);
		if (xDimTemp == 0) return false;
		return initialize(zDim,
			NetworkStandard.constructHiddenNeuronNumbers(zDim, xDimTemp, getHiddenLayerMin()),
			convFilters, deconvFilters);
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
		catch (Throwable e) {Util.trace(e);}
		
		return false;
	}

	
	/**
	 * Initialize with Z dimension and zooming out ratio.
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
	public NeuronValue[] learnRasterOneByOne(Iterable<Raster> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = paramGetLearningRate();
		return learnRasterOneByOne(sample, learningRate, terminatedThreshold, maxIteration);
	}

	
	@Override
	public NeuronValue[] learnRaster(Iterable<Raster> sample) throws RemoteException {
		int maxIteration = config.getAsInt(LEARN_MAX_ITERATION_FIELD);
		double terminatedThreshold = config.getAsReal(LEARN_TERMINATED_THRESHOLD_FIELD);
		double learningRate = paramGetLearningRate();
		return learnRaster(sample, learningRate, terminatedThreshold, maxIteration);
	}

	
	/**
	 * Learning neural network by back propagate algorithm with image specifications.
	 * @param sample learning sample. There is only inputs in the sample.
	 * @param learningRate learning rate.
	 * @param terminatedThreshold terminated threshold.
	 * @param maxIteration maximum iteration.
	 * @return learned error.
	 */
	private NeuronValue[] learnRasterOneByOne(Iterable<Raster> sample, double learningRate, double terminatedThreshold, int maxIteration) {
		return learnOneByOne(RasterAssoc.toInputSample(sample), learningRate, terminatedThreshold, maxIteration);
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
	 * @param conv convolutional network.
	 * @param deconv convolutional network.
	 * @param featureToX feature utility. 
	 * @param rasterUtil raster utility.
	 * @return converted from X data.
	 */
	public static Raster convertXDataToRaster(NeuronValue[] dataX, StackNetworkAbstract conv, StackNetworkAbstract deconv, FeatureToX featureToX, RasterUtility rasterUtil) {
		if (dataX == null)
			return null;
		else if (conv == null && deconv == null)
			return rasterUtil.createRaster(dataX);
		else if (conv != null && deconv == null)
			return conv.createRaster(featureToX.convertXToFeature(dataX));
		else  if (conv == null)
			return deconv.createRaster(featureToX.convertXToFeature(dataX));
		
		StackNetworkAssoc convAssoc = new StackNetworkAssoc(conv);
		if (convAssoc.getFullNetwork() == null)
			return deconv.createRaster(featureToX.convertXToFeature(dataX));
		NeuronValue[] feature = featureToX.convertXToFeature(dataX);
		NeuronValue[] unifiedContent = convAssoc.convertFeatureToUnifiedContentData(feature);
		return deconv.createRaster(unifiedContent);
	}
	
	
	/**
	 * Converting X data to raster.
	 * @param dataX X data.
	 * @return raster converted from X data.
	 */
	private Raster convertXDataToRaster(NeuronValue[] dataX) {
		return convertXDataToRaster(dataX, conv, deconv, this, this);
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


	/**
	 * Generate X data.
	 * @param dataZ Z data is encoded data.
	 * @return generated values (X data).
	 */
	protected abstract NeuronValue[] generateByZ(NeuronValue...dataZ);
	
	
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
	public G recover(NeuronValue[] dataX, Cube region, boolean random, boolean calcError) throws RemoteException {
		return recover(this, this, dataX, region, random, calcError);
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
		return reproduceRaster(this, raster, region, random, calcError);
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
	 * Recovering values (X values) from original data X.
	 * @param model (convolutional) generative model.
	 * @param f function on feature and data X.
	 * @param dataX original data X.
	 * @param region specified region.
	 * @param random flag to indicate whether or not to random generation.
	 * @param calcError flag to indicate whether or not to calculate error.
	 * @return generated structure.
	 */
	public static G recover(GenModel model, FeatureToX f, NeuronValue[] dataX, Cube region, boolean random, boolean calcError) {
		G g = null;
		try {
			g = random ? model.generate() : model.generateBest();
		} catch (Throwable e) {Util.trace(e);}
		if (g == null || g.xgen == null || g.xgen.length == 0) return null;

		NeuronValue[] regionGenX = null;
		dataX = f.convertXToFeature(dataX);
		Size sourceSize = f.getFeatureSize();
		if (region != null) {
			regionGenX = new NeuronValue[dataX.length];
			for (int i = 0; i < regionGenX.length; i++) regionGenX[i] = dataX[i];
			NeuronValue[] xgen = f.convertXToFeature(g.xgen);
			RasterAssoc.copyRange3D(xgen, sourceSize, region, regionGenX, sourceSize, new Point(region.x, region.y, region.z, region.t));
		}
		else
			regionGenX = f.convertXToFeature(g.xgen);
		
		double error = 0;
		int n = 0;
		if (calcError) {
			if (region != null) {
				for (int z = 0; z < region.depth; z++) {
					int indexZ = sourceSize.width*sourceSize.height;
					for (int y = 0; y < region.height; y++) {
						for (int x = 0; x < region.width; x++) {
							if (!region.contains(x, y, z)) continue;
							int index = indexZ + y*region.width + x;
							double d = regionGenX[index].subtract(dataX[index]).norm();
							error += Math.abs(d*(1-d));
							n++;
						}
					}
				}
			}
			else {
				for (int i = 0; i < regionGenX.length; i++) {
					double d = regionGenX[i].subtract(dataX[i]).norm();
					error += Math.abs(d*(1-d));
					n++;
				}
			}
		}
		
		g.xgen = f.convertFeatureToX(regionGenX);
		g.error = n != 0 ? error/(double)n : 0;
		g.x = dataX;
		return g;
	}


	/**
	 * Reproducing raster, which is similar to method {@link #recoverRaster(Raster, Cube, boolean, boolean)} except that
	 * reproducing method firstly learns from the raster itself that will be reproduced.
	 * @param model convolutional generative model.
	 * @param raster original raster.
	 * @param region specified region. If it is null, entire raster will be recovered.
	 * @param random flag to indicate whether or not to random generation.
	 * @param calcError flag to indicate whether or not to calculate error.
	 * @return generated structure.
	 */
	public static G reproduceRaster(ConvGenModel model, Raster raster, Cube region, boolean random, boolean calcError) {
		try {
			model.learnRasterOneByOne(Arrays.asList(raster));
			return model.recoverRaster(raster, region, random, calcError);
		} catch (Throwable e) {Util.trace(e);}
		
		return null;
	}
	
	
	/**
	 * Creating convolutional neural network.
	 * @param gm generative model.
	 * @param isNorm  checking whether to normalized pixels value in range [0, 1].
	 * @param idRef ID reference which can be null.
	 * @return convolutional neural network.
	 */
	public static StackNetworkAbstract defaultConvNetwork(ConvGenModel gm, boolean isNorm, Id idRef) {
		int rasterChannel = 0;
		try {
			rasterChannel = gm.getRasterChannel();
		} catch (Throwable e) {Util.trace(e);}
		if (rasterChannel < 1) return null;
		
		Function activateRef = Raster.toActivationRef(rasterChannel, isNorm);
		Function contentActivateRef = Raster.toConvActivationRef(rasterChannel, isNorm);
		if (!(gm instanceof Network))
			return StackNetworkImpl.create(rasterChannel, activateRef, contentActivateRef, idRef);

		NetworkConfig config = null;
		try {
			config = ((Network)gm).getConfig();
		} catch (Throwable e) {Util.trace(e);}
		if (config == null)
			return StackNetworkImpl.create(rasterChannel, activateRef, contentActivateRef, idRef);
		
		StackNetworkAbstract sn = null;
		if (config.getAsBoolean(CONV_CLASSIFIER_FIELD))
			sn = StackClassifier.create(rasterChannel, activateRef, contentActivateRef, idRef);
		else if (config.getAsBoolean(Raster2D.LEARN_FIELD))
			sn = StackNetworkImpl.createRSN(rasterChannel, contentActivateRef, idRef);
		else
			sn = StackNetworkImpl.create(rasterChannel, activateRef, contentActivateRef, idRef);
			
		try {
			NetworkConfig snaConfig = sn.getConfig();
			snaConfig.put(Raster.NORM_FIELD, config.get(Raster.NORM_FIELD));
			snaConfig.put(Image.ALPHA_FIELD, config.get(Image.ALPHA_FIELD));
			snaConfig.put(Raster2D.LEARN_FIELD, config.get(Raster2D.LEARN_FIELD));
			snaConfig.put(HIDDEN_LAYER_MIN_FILED, config.get(HIDDEN_LAYER_MIN_FILED));
		} catch (Throwable e) {Util.trace(e);}
		
		return sn;
	}


	/**
	 * Checking whether stack network can have learning mechanism.
	 * @param sn stack network.
	 * @return whether stack network can have learning mechanism.
	 */
	public static boolean hasLearning(StackNetworkAbstract sn) {
		if (sn == null)
			return false;
		else
			return new StackNetworkAssoc(sn).hasLearning() || sn instanceof Classifier;
	}
	
	
}
