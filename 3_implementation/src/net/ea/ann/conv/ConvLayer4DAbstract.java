/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import net.ea.ann.conv.filter.BiasFilter;
import net.ea.ann.conv.filter.DeconvConvFilter;
import net.ea.ann.conv.filter.DeconvConvFilter4D;
import net.ea.ann.conv.filter.DeconvFilter;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.conv.filter.Filter4D;
import net.ea.ann.conv.filter.ProductFilter4D;
import net.ea.ann.core.Id;
import net.ea.ann.core.Network;
import net.ea.ann.core.NetworkAbstract;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Cube;
import net.ea.ann.raster.NeuronRaster;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.Size;

/**
 * This class is an abstract implementation of convolutional layer in 3D space.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class ConvLayer4DAbstract extends ConvLayer3DAbstract implements ConvLayerSingle4D {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Raster time.
	 */
	protected int time = 1;

	
	/**
	 * Constructor with neuron channel, activation function, width, height, depth, time, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param depth layer depth.
	 * @param time layer time.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	protected ConvLayer4DAbstract(int neuronChannel, Function activateRef, int width, int height, int depth, int time, Filter filter,
			Id idRef) {
		this(neuronChannel, activateRef, filter, idRef);
		
		this.width = width;
		this.height = height;
		this.depth = depth;
		this.time = time;
		int wh = width*height;
		int whd = wh*depth;
		this.neurons = new ConvNeuron[whd*time];
		NeuronValue zero = this.newNeuronValue().zero();
		for (int t = 0; t < time; t++) {
			int indexT = t*whd;
			for (int z = 0; z < depth; z++) {
				int indexZ = indexT + z*wh;
				for (int y = 0; y < height; y++) {
					int indexY = indexZ + y*width;
					for (int x = 0; x < width; x++) {
						int index = indexY + x;
						ConvNeuron neuron = this.newNeuron();
						neuron.setValue(zero);
						
						this.neurons[index] = neuron;
					}
				}
			}
		}
		
	}

	
	/**
	 * Constructor with neuron channel, activation function, width, height, depth, time, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param depth layer depth.
	 * @param time layer time.
	 * @param filter kernel filter.
	 */
	public ConvLayer4DAbstract(int neuronChannel, Function activateRef, int width, int height, int depth, int time,
			Filter filter) {
		this(neuronChannel, activateRef, width, height, depth, time, filter, null);
	}

	
	/**
	 * Constructor with neuron channel, activation function, width, height, depth, and time.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param depth layer depth.
	 * @param time layer time.
	 */
	public ConvLayer4DAbstract(int neuronChannel, Function activateRef, int width, int height, int time, int depth) {
		this(neuronChannel, activateRef, width, height, depth, time, null, null);
	}

	
	/**
	 * Default constructor with neuron channel, activation function, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	ConvLayer4DAbstract(int neuronChannel, Function activateRef, Filter filter, Id idRef) {
		super(neuronChannel, activateRef, filter, idRef);
	}

	
	@Override
	public Filter4D getFilter4D() {
		if (filter == null)
			return null;
		else if (filter instanceof Filter4D)
			return (Filter4D)filter;
		else
			return null;
	}

	
	
	@Override
	public int getTime() {
		return time;
	}


	@Override
	public ConvNeuron get(int x, int y, int z, int t) {
		int wh = width*height;
		int whd = wh*depth;
		return neurons[t*whd + z*wh + y*width + x];
	}

	
	@Override
	public NeuronValue set(int x, int y, int z, int t, NeuronValue value) {
		int wh = width*height;
		int whd = wh*depth;
		ConvNeuron neuron = neurons[t*whd + z*wh + y*width + x];
		if (neuron == null)
			return null;
		else {
			NeuronValue prevValue = neuron.getValue();
			neuron.setValue(value);
			return prevValue;
		}
	}


	@Override
	protected NeuronValue[] getData(Cube region) {
		if (region == null) return getData();
		int width = getWidth();
		int height = getHeight();
		int depth = getDepth();
		int time = getTime();
		
		region.x = region.x < 0 ? 0 : region.x;
		region.y = region.y < 0 ? 0 : region.y;
		region.z = region.z < 0 ? 0 : region.z;
		region.t = region.t < 0 ? 0 : region.t;
		region.width = region.x + region.width <= width ? region.width : width - region.x;
		region.height = region.y + region.height <= height ? region.height : height - region.y;
		region.depth = region.z + region.depth <= depth ? region.depth : depth - region.z;
		region.time = region.t + region.time <= time ? region.time : time - region.t;
		if (region.width <= 0 || region.height <= 0 || region.depth <= 0 || region.time <= 0) return null;
		
		int regionIndex = 0;
		NeuronValue[] data = new NeuronValue[region.width*region.height*region.depth*region.time];
		int ttime = region.t + region.time;
		int zdepth = region.z + region.depth;
		int yheight = region.y + region.height;
		int xwidth = region.x + region.width;
		for (int t = region.t; t < ttime; t++) {
			int indexT = t*width*height*depth;
			for (int z = region.z; z < zdepth; z++) {
				int indexZ = indexT + z*width*height;
				for (int y = region.y; y < yheight; y++) {
					int indexY = indexZ + y*width;
					for (int x = region.x; x < xwidth; x++) {
						int index = indexY + x;
						data[regionIndex] = neurons[index].getValue();
						regionIndex++;
					}
				}
			}
		}
		
		return data;
	}

	
	@Override
	protected NeuronValue[] setData(NeuronValue[] data, Cube region) {
		if (region == null) setData(data);
		if (data == null || neurons.length == 0) return null;
		int width = getWidth();
		int height = getHeight();
		int depth = getDepth();
		int time = getTime();
		
		region.x = region.x < 0 ? 0 : region.x;
		region.y = region.y < 0 ? 0 : region.y;
		region.z = region.z < 0 ? 0 : region.z;
		region.t = region.t < 0 ? 0 : region.t;
		region.width = region.x + region.width <= width ? region.width : width - region.x;
		region.height = region.y + region.height <= height ? region.height : height - region.y;
		region.depth = region.z + region.depth <= depth ? region.depth : depth - region.z;
		region.time = region.t + region.time <= time ? region.time : time - region.t;
		if (region.width <= 0 || region.height <= 0 || region.depth <= 0|| region.time <= 0) return null;
		
		int regionIndex = 0;
		data = NeuronValue.adjustArray(data, region.width*region.height*region.depth*region.time, this);
		int ttime = region.t + region.time;
		int zdepth = region.z + region.depth;
		int yheight = region.y + region.height;
		int xwidth = region.x + region.width;
		for (int t = region.t; t < ttime; t++) {
			int indexT = t*width*height*depth;
			for (int z = region.z; z < zdepth; z++) {
				int indexZ = indexT + z*width*height;
				for (int y = region.y; y < yheight; y++) {
					int indexY = indexZ + y*width;
					for (int x = region.x; x < xwidth; x++) {
						int index = indexY + x;
						neurons[index].setValue(data[regionIndex]);
						regionIndex++;
					}
				}
			}
		}
		
		return data;
	}

	
	/**
	 * Getting the region in next layer, corresponding to the current neuron at specified coordination at current layer.
	 * @param x X coordination.
	 * @param y Y coordination.
	 * @param z Z coordination.
	 * @param t T coordination.
	 * @return the region in next layer, corresponding to the current neuron at specified coordination at current layer.
	 */
	protected Cube getNextRegion(int x, int y, int z, int t) {
		Cube nextArea = getNextRegion(x, y, z);
		if (nextArea == null) return null;
		Filter filter = getFilter();
		
		int filterStrideTime = filter.getStrideTime();
		int nextTime = ((ConvLayerSingle)nextLayer).getTime();
		
		Cube nextRegion = new Cube(nextArea.x, nextArea.y, nextArea.z, 0, nextArea.width, nextArea.height, nextArea.depth, 0);
		if (filter instanceof DeconvFilter) {
			nextRegion.t = t * filterStrideTime;
			nextRegion.time = filterStrideTime;
		}
		else {
			nextRegion.t = t / filterStrideTime;
			nextRegion.time = 1;
		}
		
		nextRegion.time = nextRegion.t + nextRegion.time <= nextTime ? nextRegion.time : nextTime - nextRegion.t;
		if (nextRegion.time <= 0)
			return null;
		else
			return nextRegion;
	}

	
	@Override
	protected Cube getNextRegion(Cube thisRegion) {
		thisRegion = thisRegion != null ? thisRegion : new Cube(0, 0, 0, 0, getWidth(), getHeight(), getDepth(), getTime());
		Cube nextArea = getNextRegion(thisRegion.x, thisRegion.y, thisRegion.z);
		if (nextArea == null) return null;
		Filter filter = getFilter();
		
		int filterStrideTime = filter.getStrideTime();
		int nextTime = ((ConvLayerSingle)nextLayer).getTime();
		
		Cube nextRegion = new Cube(nextArea.x, nextArea.y, nextArea.z, 0, nextArea.width, nextArea.height, nextArea.depth, 0);
		if (filter instanceof DeconvFilter) {
			nextRegion.t = thisRegion.t * filterStrideTime;
			nextRegion.time = thisRegion.time * filterStrideTime;
		}
		else {
			nextRegion.t = thisRegion.t / filterStrideTime;
			nextRegion.time = thisRegion.time / filterStrideTime;
			nextRegion.time = nextRegion.time < 1 ? 1 : nextRegion.time; 
		}
		
		nextRegion.time = nextRegion.t + nextRegion.time <= nextTime ? nextRegion.time : nextTime - nextRegion.t;
		if (nextRegion.time <= 0)
			return null;
		else
			return nextRegion;
	}

	
	/**
	 * Getting the region in next layer, corresponding to the current neuron at specified coordination at current layer.
	 * @param x X coordination.
	 * @param y Y coordination.
	 * @param z Z coordination.
	 * @param t T coordination.
	 * @return the region in next layer, corresponding to the current neuron at specified coordination at current layer.
	 */
	protected Cube getPrevRegion(int x, int y, int z, int t) {
		Cube prevArea = getPrevRegion(x, y, z);
		if (prevArea == null) return null;
		ConvLayerSingle prevLayer = (ConvLayerSingle)this.prevLayer;
		Filter filter = prevLayer.getFilter();
		
		int filterStrideTime = filter.getStrideTime();
		int prevTime = prevLayer.getTime();
		int prevBlockTime = filter.isMoveStride() ? prevTime / filterStrideTime : prevTime;
		
		Cube prevRegion = new Cube(prevArea.x, prevArea.y, prevArea.z, 0, prevArea.width, prevArea.height, prevArea.depth, 0);
		if (filter instanceof DeconvFilter) {
			prevRegion.t = t / filterStrideTime;
			prevRegion.t = prevRegion.t < prevTime ? prevRegion.t : prevTime-1;
			prevRegion.time = 1;
		}
		else {
			int tBlock = t < prevBlockTime ? t : prevBlockTime-1;
			prevRegion.t = tBlock*filterStrideTime;
			prevRegion.time = filterStrideTime;
		}
		
		prevRegion.time = prevRegion.t + prevRegion.time <= prevTime ? prevRegion.time : prevTime - prevRegion.t;
		if (prevRegion.time <= 0)
			return null;
		else
			return prevRegion;
	}

	
	@Override
	protected Cube getPrevRegion(Cube thisRegion) {
		thisRegion = thisRegion != null ? thisRegion : new Cube(0, 0, 0, 0, getWidth(), getHeight(), getDepth(), getTime());
		Cube prevArea = getPrevRegion(thisRegion.x, thisRegion.y, thisRegion.z);
		if (prevArea == null) return null;
		ConvLayerSingle prevLayer = (ConvLayerSingle)this.prevLayer;
		Filter filter = prevLayer.getFilter();
		
		int filterStrideTime = filter.getStrideTime();
		int prevTime = prevLayer.getTime();
		int prevBlockTime = filter.isMoveStride() ? prevTime / filterStrideTime : prevTime;
		
		Cube prevRegion = new Cube(prevArea.x, prevArea.y, prevArea.z, 0, prevArea.width, prevArea.height, prevArea.depth, 0);
		if (filter instanceof DeconvFilter) {
			prevRegion.t = thisRegion.t / filterStrideTime;
			prevRegion.t = prevRegion.t < prevTime ? prevRegion.t : prevTime-1;
			prevRegion.time = thisRegion.time / filterStrideTime;
			prevRegion.time = prevRegion.time < 1 ? 1 : prevRegion.time; 
		}
		else {
			int tBlock = thisRegion.t < prevBlockTime ? thisRegion.t : prevBlockTime-1;
			prevRegion.t = tBlock*filterStrideTime;
			prevRegion.time = prevRegion.time*filterStrideTime;
		}
		
		prevRegion.time = prevRegion.t + prevRegion.time <= prevTime ? prevRegion.time : prevTime - prevRegion.t;
		if (prevRegion.time <= 0)
			return null;
		else
			return prevRegion;
	}

	
	@Override
	public ConvLayer forward() {
		ConvLayer nextLayer = getNextLayer();
		if ((nextLayer == null) || !(nextLayer instanceof ConvLayerSingle4D)) return null;
		NeuronRaster result = forward(this, (ConvLayerSingle4D)nextLayer, getFilter(), (Cube)null, (Cube)null, true);
		return result != null ? nextLayer : null;
	}

	
	@Override
	public ConvLayerSingle forward(ConvLayerSingle nextLayer, Filter filter) {
		NeuronRaster result = forward(this, (ConvLayerSingle4D)nextLayer, filter, (Cube)null, (Cube)null, true);
		return result != null ? nextLayer : null;
	}

	
	/**
	 * Forwarding evaluation from current layer to next layer.
	 * @param thisLayer current layer.
	 * @param nextLayer next layer.
	 * @param f filter of current layer.
	 * @param thisFilterRegion filtering region of current layer. It can be null.
	 * @param nextFilterRegion filtering region of next layer. It can be null.
	 * @param nextAffected flag to indicate whether the next layer is affected
	 * @return arrays of neurons filtered.
	 */
	static NeuronRaster forward(ConvLayerSingle4D thisLayer, ConvLayerSingle4D nextLayer, Filter f, Cube thisFilterRegion, Cube nextFilterRegion, boolean nextAffected) {
		if (thisLayer == null) return null;
		Filter4D filter = (f != null && f instanceof Filter4D) ? (Filter4D)f : thisLayer.getFilter4D();
		if (filter == null) {
			return ConvLayer3DAbstract.forward(thisLayer, nextLayer, f, thisFilterRegion, nextFilterRegion, nextAffected);
		}
		
		NeuronValue nextZero = nextLayer != null ? nextLayer.newNeuronValue().zero() : thisLayer.newNeuronValue().zero();
		ConvNeuron[] nextNeurons = nextLayer.getNeurons();
		if (nextAffected)
			nextNeurons = nextLayer.getNeurons();
		else {
			nextNeurons = new ConvNeuron[nextLayer.length()];
			for (int i = 0; i < nextNeurons.length; i++) {
				nextNeurons[i].setValue(nextZero);
				nextNeurons[i].setInput(nextZero);
			}
		}
		if (nextNeurons == null || nextNeurons.length == 0) return null;
		
		if (filter instanceof DeconvConvFilter) {
			for (ConvNeuron nextNeuron : nextNeurons) {
				nextNeuron.setValue(null);
				nextNeuron.setInput(null);
			}
		}
		
		if (thisFilterRegion != null && nextFilterRegion != null) nextFilterRegion = null;
		
		int filterStrideWidth = filter.getStrideWidth();
		int filterStrideHeight = filter.getStrideHeight();
		int filterStrideDepth = filter.getStrideDepth();
		int filterStrideTime = filter.getStrideTime();
		int thisWidth = thisLayer.getWidth();
		int thisHeight = thisLayer.getHeight();
		int thisDepth = thisLayer.getDepth();
		int thisTime = thisLayer.getTime();
		int thisBlockWidth = filter.isMoveStride() ? thisWidth / filterStrideWidth : thisWidth;
		int thisBlockHeight = filter.isMoveStride() ? thisHeight / filterStrideHeight : thisHeight;
		int thisBlockDepth = filter.isMoveStride() ? thisDepth / filterStrideDepth : thisDepth;
		int thisBlockTime = filter.isMoveStride() ? thisTime / filterStrideTime : thisTime;
		int nextWidth = nextLayer.getWidth();
		int nextHeight = nextLayer.getHeight();
		int nextDepth = nextLayer.getDepth();
		int nextTime = nextLayer.getTime();
		Function activateRef = nextLayer.getActivateRef();
		activateRef = activateRef == null ? thisLayer.getActivateRef() : activateRef;
		
		for (int nextT = 0; nextT < nextTime; nextT++) {
			int thisT = 0;
			if (filter instanceof DeconvFilter) {
				thisT = nextT / filterStrideTime;
				if (!nextLayer.isPadZeroFilter()) thisT = thisT < thisTime ? thisT : thisTime-1;
			}
			else {
				int tBlock = nextLayer.isPadZeroFilter() ? nextT : (nextT < thisBlockTime ? nextT : thisBlockTime-1);
				thisT = tBlock*filterStrideTime;
			}
			
			int nextIndexT = nextT*nextWidth*nextHeight*nextDepth;
			for (int nextZ = 0; nextZ < nextDepth; nextZ++) {
				int thisZ = 0;
				if (filter instanceof DeconvFilter) {
					thisZ = nextZ / filterStrideDepth;
					if (!nextLayer.isPadZeroFilter()) thisZ = thisZ < thisDepth ? thisZ : thisDepth-1;
				}
				else {
					int zBlock = nextLayer.isPadZeroFilter() ? nextZ : (nextZ < thisBlockDepth ? nextZ : thisBlockDepth-1);
					thisZ = zBlock*filterStrideDepth;
				}
				
				int nextIndexZ = nextIndexT + nextZ*nextWidth*nextHeight;
				for (int nextY = 0; nextY < nextHeight; nextY++) {
					int thisY = 0;
					if (filter instanceof DeconvFilter) {
						thisY = nextY / filterStrideHeight;
						if (!nextLayer.isPadZeroFilter()) thisY = thisY < thisHeight ? thisY : thisHeight-1;
					}
					else {
						int yBlock = nextLayer.isPadZeroFilter() ? nextY : (nextY < thisBlockHeight ? nextY : thisBlockHeight-1);
						thisY = yBlock*filterStrideHeight;
					}
					
					int nextIndexY = nextIndexZ + nextY*nextWidth;
					for (int nextX = 0; nextX < nextWidth; nextX++) {
						int thisX = 0;
						if (filter instanceof DeconvFilter) {
							thisX = nextX / filterStrideWidth;
							if (!nextLayer.isPadZeroFilter()) thisX = thisX < thisWidth ? thisX : thisWidth-1;
						}
						else {
							int xBlock = nextLayer.isPadZeroFilter() ? nextX : (nextX < thisBlockWidth ? nextX : thisBlockWidth-1);
							thisX = xBlock*filterStrideWidth;
						}
						
						//Setting zero to outline pixels.
						int nextIndex = nextIndexY + nextX;
						if (thisT >= thisTime || thisZ >= thisDepth || thisY >= thisHeight || thisX >= thisWidth) {
							nextNeurons[nextIndex].setValue(nextZero);
							continue;
						}
	
						//Checking region.
						if (thisFilterRegion != null && !thisFilterRegion.contains(thisX, thisY, thisZ, thisT)) continue;
						if (nextFilterRegion != null && !nextFilterRegion.contains(nextX, nextY, nextZ, nextT)) continue;
						
						//Filtering
						NeuronValue filteredValue = null;
						if (filter instanceof DeconvConvFilter)
							filteredValue = ((DeconvConvFilter4D)filter).apply(thisX, thisY, thisZ, thisT, thisLayer, nextX, nextY, nextZ, nextT, nextLayer);
						else
							filteredValue = filter.apply(thisX, thisY, thisZ, thisT, thisLayer);
						
						if (filteredValue != null) {
							filteredValue = filteredValue.add(thisLayer.getBias());
							nextNeurons[nextIndex].setInput(filteredValue);
							if (activateRef != null) filteredValue = activateRef.evaluate(filteredValue);
							nextNeurons[nextIndex].setValue(filteredValue);
						}
						else
							nextNeurons[nextIndex].setValue(nextZero);
					} //End next X.
				} //End next Y.
			} //End next Z.
		} //End next T.
		
		if (filter instanceof DeconvConvFilter) {
			for (ConvNeuron nextNeuron : nextNeurons) {
				if (nextNeuron.getValue() == null) nextNeuron.setValue(nextZero);
				if (nextNeuron.getInput() == null) nextNeuron.setInput(nextZero);
			}
		}

		if ((!(thisLayer instanceof ConvLayer4DAbstract)) || (thisFilterRegion == null && nextFilterRegion == null))
			return new NeuronRaster(nextLayer.getNeuronChannel(), nextNeurons, new Size(nextWidth, nextHeight, nextDepth, nextTime));
		
		Cube nextRegion = null;
		if (thisFilterRegion == null && nextFilterRegion == null)
			nextRegion = ((ConvLayer4DAbstract)thisLayer).getNextRegion(new Cube(0, 0, 0, 0, thisWidth, thisHeight, thisDepth, thisTime));
		else if (thisFilterRegion != null)
			nextRegion = ((ConvLayer4DAbstract)thisLayer).getNextRegion(thisFilterRegion);
		else
			nextRegion = nextFilterRegion;
		if (nextRegion == null) return new NeuronRaster(nextLayer.getNeuronChannel(), nextNeurons, new Size(nextWidth, nextHeight, nextDepth, nextTime));

		ConvNeuron[] regionNeurons = new ConvNeuron[nextRegion.width*nextRegion.height*nextRegion.depth*nextRegion.time];
		int regionIndex = 0;
		int rtime = nextRegion.t + nextRegion.time;
		int rdepth = nextRegion.z + nextRegion.depth;
		int rheight = nextRegion.y + nextRegion.height;
		int rwidth = nextRegion.x + nextRegion.width;
		for (int nextT = nextRegion.t; nextT < rtime; nextT++) {
			int nextIndexT = nextT*nextWidth*nextHeight*nextDepth;
			for (int nextZ = nextRegion.z; nextZ < rdepth; nextZ++) {
				int nextIndexZ = nextIndexT + nextZ*nextWidth*nextHeight;
				for (int nextY = nextRegion.y; nextY < rheight; nextY++) {
					int nextIndexY = nextIndexZ + nextY*nextWidth;
					for (int nextX = nextRegion.x; nextX < rwidth; nextX++) {
						int nextIndex = nextIndexY + nextX;
						regionNeurons[regionIndex] = nextNeurons[nextIndex];
						regionIndex++;
					}
				}
			}
		}
		return new NeuronRaster(nextLayer.getNeuronChannel(), regionNeurons, new Size(nextRegion.width, nextRegion.height, nextRegion.depth, nextRegion.time));
	}

	
	/**
	 * Forwarding evaluation from current layer to next layer.
	 * @param input input data.
	 * @param thisLayer current layer.
	 * @param nextLayer next layer.
	 * @param f filter of current layer.
	 * @param thisFilterRegion filtering region of current layer. It can be null.
	 * @param nextFilterRegion filtering region of next layer. It can be null.
	 * @param nextAffected flag to indicate whether the next layer is affected
	 * @return arrays of neurons filtered.
	 */
	public static NeuronRaster forward(NeuronValue[] input, ConvLayerSingle4D thisLayer, ConvLayerSingle4D nextLayer, Filter f, Cube thisFilterRegion, Cube nextFilterRegion, boolean nextAffected) {
		if (input != null) thisLayer.setData(input);
		return forward(thisLayer, nextLayer, f, thisFilterRegion, nextFilterRegion, nextAffected);
	}

	
	@Override
	public Raster createRaster(NeuronValue[] values,
			boolean isNorm, int defaultAlpha) {
		return RasterAssoc.createRaster(this, values, isNorm, defaultAlpha);
	}


	@Override
	public BiasFilter learnFilter(BiasFilter initialFilter, boolean learningBias, double learningRate, int maxIteration) {
		if (nextLayer == null) return null;
		ConvLayerSingle4D smallLayer = null, largeLayer = null;
		if (this.length() >= ((ConvLayerSingle)nextLayer).length()) {
			smallLayer = (ConvLayerSingle4D)nextLayer;
			largeLayer = this;
		}
		else {
			smallLayer = this;
			largeLayer = (ConvLayerSingle4D)nextLayer;
		}
		
		maxIteration = maxIteration > 0 ? maxIteration : Network.LEARN_MAX_ITERATION_DEFAULT;
		int iteration = 0;
		BiasFilter filter = initialFilter;
		while (iteration < maxIteration) {
			double lr = NetworkAbstract.calcLearningRate(learningRate, iteration, NetworkAbstract.LEARN_RATE_FIXED_DEFAULT);
			filter = learnFilter(smallLayer, largeLayer, filter, learningBias, lr);
			iteration++;
		}
		
		return filter;
	}

	
	/**
	 * Learning product filter from large layer to small layer.
	 * @param smallLayer specified small layer.
	 * @param largeLayer specified large layer.
	 * @param initialFilter initial filter. This is initial filter of larger layer.
	 * @param learningBias flag to indicate whether to learn bias.
	 * @param learningRate learning rate.
	 * @return filter learned from large layer to small layer. It is filter stored in large layer.
	 */
	static BiasFilter learnFilter(ConvLayerSingle4D smallLayer, ConvLayerSingle4D largeLayer, BiasFilter initialFilter, boolean learningBias, double learningRate) {
		if (smallLayer == null || largeLayer == null) return null;
		if (largeLayer.getTime() <= 1) return ConvLayer3DAbstract.learnFilter(smallLayer, largeLayer, initialFilter, learningBias, learningRate);
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? 1 : learningRate;
		
		ConvLayerSingle4D thisLayer = largeLayer, nextLayer = smallLayer;
		int n = Math.min(largeLayer.getWidth()/smallLayer.getWidth(), largeLayer.getHeight()/smallLayer.getHeight());
		n = Math.min(n, largeLayer.getDepth()/smallLayer.getDepth());
		n = Math.min(n, largeLayer.getTime()/smallLayer.getTime());
		boolean isMoveStride = true;
		if (n <= 1) {
			n = 4;
			isMoveStride = false;
		}
		
		NeuronValue zero = thisLayer.newNeuronValue().zero();
		ProductFilter4D filter = null;
		ProductFilter4D initialProductFilter = ((initialFilter != null) && (initialFilter.filter != null) && (initialFilter.filter instanceof ProductFilter4D)) ?
			(ProductFilter4D)(initialFilter.filter) : null;
		if (initialProductFilter == null || initialProductFilter.width() != n || initialProductFilter.height() != n || initialProductFilter.depth() != n || initialProductFilter.time() != n) {
			NeuronValue[][][][] kernel = new NeuronValue[n][n][n][n];
			for (int h = 0; h < n; h++) {
				for (int i = 0; i < n; i++) {
					for (int j = 0; j < n; j++) {
						for (int k = 0; k < n; k++) kernel[h][i][j][k] = zero;
					}
				}
			}
			filter = ProductFilter4D.create(kernel, zero.unit());
		}
		else {
			NeuronValue[][][][] kernel = new NeuronValue[n][n][n][n];
			for (int h = 0; h < n; h++) {
				for (int i = 0; i < n; i++) {
					for (int j = 0; j < n; j++) {
						for (int k = 0; k < n; k++) kernel[h][i][j][k] = initialProductFilter.getKernel()[h][i][j][k];
					}
				}
			}
			filter = ProductFilter4D.create(kernel, initialProductFilter.getWeight());
		}
		
		filter.setMoveStride(isMoveStride);
		NeuronValue[][][][] kernel = filter.getKernel();
		
		NeuronValue bias = null;
		if ((initialFilter != null) && (initialFilter.bias != null))
			bias = initialFilter.bias;
		else
			bias = zero; 
		
		int filterStrideWidth = filter.getStrideWidth();
		int filterStrideHeight = filter.getStrideHeight();
		int filterStrideDepth = filter.getStrideDepth();
		int filterStrideTime = filter.getStrideTime();
		int thisWidth = thisLayer.getWidth();
		int thisHeight = thisLayer.getHeight();
		int thisDepth = thisLayer.getDepth();
		int thisTime = thisLayer.getTime();
		int thisBlockWidth = filter.isMoveStride() ? thisWidth / filterStrideWidth : thisWidth;
		int thisBlockHeight = filter.isMoveStride() ? thisHeight / filterStrideHeight : thisHeight;
		int thisBlockDepth = filter.isMoveStride() ? thisDepth / filterStrideDepth : thisDepth;
		int thisBlockTime = filter.isMoveStride() ? thisTime / filterStrideTime : thisTime;
		int nextWidth = nextLayer.getWidth();
		int nextHeight = nextLayer.getHeight();
		int nextDepth = nextLayer.getDepth();
		int nextTime = nextLayer.getTime();
	
		for (int nextT = 0; nextT < nextTime-1; nextT++) { //Please pay attention to minus one (-1) which prevents overlapping estimation.
			int thisT = 0;
			int tBlock = nextT < thisBlockTime ? nextT : thisBlockTime-1;
			thisT = tBlock*filterStrideTime;
			
			for (int nextZ = 0; nextZ < nextDepth-1; nextZ++) { //Please pay attention to minus one (-1) which prevents overlapping estimation.
				int thisZ = 0;
				int zBlock = nextZ < thisBlockDepth ? nextZ : thisBlockDepth-1;
				thisZ = zBlock*filterStrideDepth;
				
				for (int nextY = 0; nextY < nextHeight-1; nextY++) { //Please pay attention to minus one (-1) which prevents overlapping estimation.
					int thisY = 0;
					int yBlock = nextY < thisBlockHeight ? nextY : thisBlockHeight-1;
					thisY = yBlock*filterStrideHeight;
					
					for (int nextX = 0; nextX < nextWidth-1; nextX++) { //Please pay attention to minus one (-1) which prevents overlapping estimation.
						int thisX = 0;
						int xBlock = nextX < thisBlockWidth ? nextX : thisBlockWidth-1;
						thisX = xBlock*filterStrideWidth;
						
						NeuronValue filteredNextValue = filter.apply(thisX, thisY, thisZ, thisT, thisLayer);
						if (filteredNextValue == null) continue;
						Function nextActivateRef = nextLayer.getActivateRef(); 
						if (nextActivateRef != null) filteredNextValue = nextActivateRef.evaluate(filteredNextValue); 
						
						NeuronValue realNextValue = nextLayer.get(nextX, nextY, nextZ, nextT).getValue();
						NeuronValue error = realNextValue.subtract(filteredNextValue);
						if (nextActivateRef != null)
							error = error.multiplyDerivative(nextActivateRef.derivative(filteredNextValue));
						
						//Learning kernel.
						for (int h = 0; h < n; h++) {
							for (int i = 0; i < n; i++) {
								for (int j = 0; j < n; j++) {
									for (int k = 0; k < n; k++) {
										NeuronValue thisValue = thisLayer.get(thisX+k, thisY+j, thisZ+i, thisT+h).getValue();
										NeuronValue delta = error.multiply(thisValue).multiply(learningRate);
										kernel[h][i][j][k] = kernel[h][i][j][k].add(delta);
									}
								}
							}
						}
						
						//Learning bias.
						if (learningBias) bias = bias.add(error.multiply(learningRate));
					} //End X.
				} //End Y.
			} //End Z.
		} //End T.
		
		return new BiasFilter(filter, learningBias?bias:null);
	}


}
