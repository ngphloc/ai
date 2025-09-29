/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import java.awt.Rectangle;

import net.ea.ann.conv.filter.BiasFilter;
import net.ea.ann.conv.filter.DeconvConvFilter;
import net.ea.ann.conv.filter.DeconvConvFilter3D;
import net.ea.ann.conv.filter.DeconvFilter;
import net.ea.ann.conv.filter.Filter;
import net.ea.ann.conv.filter.Filter3D;
import net.ea.ann.conv.filter.ProductFilter3D;
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
public abstract class ConvLayer3DAbstract extends ConvLayer2DAbstract implements ConvLayerSingle3D {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Raster height.
	 */
	protected int depth = 1;

	
	/**
	 * Constructor with neuron channel, activation function, width, height, depth, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param depth layer depth.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	protected ConvLayer3DAbstract(int neuronChannel, Function activateRef, int width, int height, int depth, Filter filter, Id idRef) {
		this(neuronChannel, activateRef, filter, idRef);
		
		this.width = width;
		this.height = height;
		this.depth = depth;
		int wh = width*height;
		this.neurons = new ConvNeuron[wh*depth];
		NeuronValue zero = this.newNeuronValue().zero();
		for (int z = 0; z < depth; z++) {
			int indexZ = z*wh;
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int index = indexZ + y*width + x;
					ConvNeuron neuron = this.newNeuron();
					neuron.setValue(zero);
					
					this.neurons[index] = neuron;
				}
			}
		}
		
	}

	
	/**
	 * Constructor with neuron channel, activation function, width, height, depth, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param depth layer depth.
	 * @param filter kernel filter.
	 */
	protected ConvLayer3DAbstract(int neuronChannel, Function activateRef, int width, int height, int depth, Filter filter) {
		this(neuronChannel, activateRef, width, height, depth, filter, null);
	}

	
	/**
	 * Constructor with neuron channel, activation function, width, height, and depth.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param width layer width.
	 * @param height layer height.
	 * @param depth layer depth.
	 */
	protected ConvLayer3DAbstract(int neuronChannel, Function activateRef, int width, int height, int depth) {
		this(neuronChannel, activateRef, width, height, depth, null, null);
	}

	
	/**
	 * Default constructor with neuron channel, activation function, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	ConvLayer3DAbstract(int neuronChannel, Function activateRef, Filter filter, Id idRef) {
		super(neuronChannel, activateRef, filter, idRef);
	}


	@Override
	public Filter3D getFilter3D() {
		if (filter == null)
			return null;
		else if (filter instanceof Filter3D)
			return (Filter3D)filter;
		else
			return null;
	}
	
	
	@Override
	public int getDepth() {
		return depth;
	}


	@Override
	public ConvNeuron get(int x, int y, int z) {
		return neurons[z*width*height + y*width + x];
	}


	@Override
	public NeuronValue set(int x, int y, int z, NeuronValue value) {
		ConvNeuron neuron = neurons[z*width*height + y*width + x];
		if (neuron == null)
			return null;
		else {
			NeuronValue prevValue = neuron.getValue();
			neuron.setValue(value);
			return prevValue;
		}
	}


	/**
	 * Getting data from specified region.
	 * @param region specified region.
	 * @return data extracted from specified region.
	 */
	protected NeuronValue[] getData(Cube region) {
		if (region == null) return getData();
		int width = getWidth();
		int height = getHeight();
		int depth = getDepth();
		
		region.x = region.x < 0 ? 0 : region.x;
		region.y = region.y < 0 ? 0 : region.y;
		region.z = region.z < 0 ? 0 : region.z;
		region.width = region.x + region.width <= width ? region.width : width - region.x;
		region.height = region.y + region.height <= height ? region.height : height - region.y;
		region.depth = region.z + region.depth <= depth ? region.depth : depth - region.z;
		if (region.width <= 0 || region.height <= 0 || region.depth <= 0) return null;
		
		int regionIndex = 0;
		NeuronValue[] data = new NeuronValue[region.width*region.height*region.depth];
		int zdepth = region.z + region.depth;
		int yheight = region.y + region.height;
		int xwidth = region.x + region.width;
		for (int z = region.z; z < zdepth; z++) {
			int indexZ = z*width*height;
			for (int y = region.y; y < yheight; y++) {
				int indexY = indexZ + y*width;
				for (int x = region.x; x < xwidth; x++) {
					int index = indexY + x;
					data[regionIndex] = neurons[index].getValue();
					regionIndex++;
				}
			}
		}
		
		return data;
	}

	
	/**
	 * Setting data at specified region.
	 * @param data data as array of neuron value.
	 * @param region specified region.
	 * @return data to be set.
	 */
	protected NeuronValue[] setData(NeuronValue[] data, Cube region) {
		if (region == null) setData(data);
		if (data == null || neurons.length == 0) return null;
		int width = getWidth();
		int height = getHeight();
		int depth = getDepth();
		
		region.x = region.x < 0 ? 0 : region.x;
		region.y = region.y < 0 ? 0 : region.y;
		region.z = region.z < 0 ? 0 : region.z;
		region.width = region.x + region.width <= width ? region.width : width - region.x;
		region.height = region.y + region.height <= height ? region.height : height - region.y;
		region.depth = region.z + region.depth <= depth ? region.depth : depth - region.z;
		if (region.width <= 0 || region.height <= 0 || region.depth <= 0) return null;
		
		int regionIndex = 0;
		data = NeuronValue.adjustArray(data, region.width*region.height*region.depth, this);
		int zdepth = region.z + region.depth;
		int yheight = region.y + region.height;
		int xwidth = region.x + region.width;
		for (int z = region.z; z < zdepth; z++) {
			int indexZ = z*width*height;
			for (int y = region.y; y < yheight; y++) {
				int indexY = indexZ + y*width;
				for (int x = region.x; x < xwidth; x++) {
					int index = indexY + x;
					neurons[index].setValue(data[regionIndex]);
					regionIndex++;
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
	 * @return the region in next layer, corresponding to the current neuron at specified coordination at current layer.
	 */
	protected Cube getNextRegion(int x, int y, int z) {
		Rectangle nextArea = getNextRegion(x, y);
		if (nextArea == null) return null;
		Filter filter = getFilter();
		
		int filterStrideDepth = filter.getStrideDepth();
		int nextDepth = ((ConvLayerSingle)nextLayer).getDepth();
		
		Cube nextRegion = new Cube(nextArea.x, nextArea.y, 0, nextArea.width, nextArea.height, 0);
		if (filter instanceof DeconvFilter) {
			nextRegion.z = z * filterStrideDepth;
			nextRegion.depth = filterStrideDepth;
		}
		else {
			nextRegion.z = z / filterStrideDepth;
			nextRegion.depth = 1;
		}
		
		nextRegion.depth = nextRegion.z + nextRegion.depth <= nextDepth ? nextRegion.depth : nextDepth - nextRegion.z;
		if (nextRegion.depth <= 0)
			return null;
		else
			return nextRegion;
	}
	

	/**
	 * Getting the next region of this region.
	 * @param thisRegion this region.
	 * @return the next region of this region.
	 */
	protected Cube getNextRegion(Cube thisRegion) {
		thisRegion = thisRegion != null ? thisRegion : new Cube(0, 0, 0, getWidth(), getHeight(), getDepth());
		Rectangle nextArea = getNextRegion(thisRegion.x, thisRegion.y);
		if (nextArea == null) return null;
		Filter filter = getFilter();
		
		int filterStrideDepth = filter.getStrideDepth();
		int nextDepth = ((ConvLayerSingle)nextLayer).getDepth();
		
		Cube nextRegion = new Cube(nextArea.x, nextArea.y, 0, nextArea.width, nextArea.height, 0);
		if (filter instanceof DeconvFilter) {
			nextRegion.z = thisRegion.z * filterStrideDepth;
			nextRegion.depth = thisRegion.depth * filterStrideDepth;
		}
		else {
			nextRegion.z = thisRegion.z / filterStrideDepth;
			nextRegion.depth = thisRegion.depth / filterStrideDepth;
			nextRegion.depth = nextRegion.depth < 1 ? 1 : nextRegion.depth; 
		}
		
		nextRegion.depth = nextRegion.z + nextRegion.depth <= nextDepth ? nextRegion.depth : nextDepth - nextRegion.z;
		if (nextRegion.depth <= 0)
			return null;
		else
			return nextRegion;
	}

	
	/**
	 * Getting the region in next layer, corresponding to the current neuron at specified coordination at current layer.
	 * @param x X coordination.
	 * @param y Y coordination.
	 * @param z Z coordination.
	 * @return the region in next layer, corresponding to the current neuron at specified coordination at current layer.
	 */
	protected Cube getPrevRegion(int x, int y, int z) {
		Rectangle prevArea = getPrevRegion(x, y);
		if (prevArea == null) return null;
		ConvLayerSingle prevLayer = (ConvLayerSingle)this.prevLayer;
		Filter filter = prevLayer.getFilter();
		
		int filterStrideDepth = filter.getStrideDepth();
		int prevDepth = prevLayer.getDepth();
		int prevBlockDepth = filter.isMoveStride() ? prevDepth / filterStrideDepth : prevDepth;
		
		Cube prevRegion = new Cube(prevArea.x, prevArea.y, 0, prevArea.width, prevArea.height, 0);
		if (filter instanceof DeconvFilter) {
			prevRegion.z = z / filterStrideDepth;
			prevRegion.z = prevRegion.z < prevDepth ? prevRegion.z : prevDepth-1;
			prevRegion.depth = 1;
		}
		else {
			int zBlock = z < prevBlockDepth ? z : prevBlockDepth-1;
			prevRegion.z = zBlock*filterStrideDepth;
			prevRegion.depth = filterStrideDepth;
		}
		
		prevRegion.depth = prevRegion.z + prevRegion.depth <= prevDepth ? prevRegion.depth : prevDepth - prevRegion.z;
		if (prevRegion.depth <= 0)
			return null;
		else
			return prevRegion;
	}
	
	
	/**
	 * Getting the previous region of this region.
	 * @param thisRegion this region.
	 * @return the previous region of this region.
	 */
	protected Cube getPrevRegion(Cube thisRegion) {
		thisRegion = thisRegion != null ? thisRegion : new Cube(0, 0, 0, getWidth(), getHeight(), getDepth());
		Rectangle prevArea = getPrevRegion(thisRegion.x, thisRegion.y);
		if (prevArea == null) return null;
		ConvLayerSingle prevLayer = (ConvLayerSingle)this.prevLayer;
		Filter filter = prevLayer.getFilter();
		
		int filterStrideDepth = filter.getStrideDepth();
		int prevDepth = prevLayer.getDepth();
		int prevBlockDepth = filter.isMoveStride() ? prevDepth / filterStrideDepth : prevDepth;
		
		Cube prevRegion = new Cube(prevArea.x, prevArea.y, 0, prevArea.width, prevArea.height, 0);
		if (filter instanceof DeconvFilter) {
			prevRegion.z = thisRegion.z / filterStrideDepth;
			prevRegion.z = prevRegion.z < prevDepth ? prevRegion.z : prevDepth-1;
			prevRegion.depth = thisRegion.depth / filterStrideDepth;
			prevRegion.depth = prevRegion.depth < 1 ? 1 : prevRegion.depth; 
		}
		else {
			int zBlock = thisRegion.z < prevBlockDepth ? thisRegion.z : prevBlockDepth-1;
			prevRegion.z = zBlock*filterStrideDepth;
			prevRegion.depth = prevRegion.depth*filterStrideDepth;
		}
		
		prevRegion.depth = prevRegion.z + prevRegion.depth <= prevDepth ? prevRegion.depth : prevDepth - prevRegion.z;
		if (prevRegion.depth <= 0)
			return null;
		else
			return prevRegion;
	}

	
	@Override
	public ConvLayer forward() {
		ConvLayer nextLayer = getNextLayer();
		if ((nextLayer == null) || !(nextLayer instanceof ConvLayerSingle3D)) return null;
		NeuronRaster result = forward(this, (ConvLayerSingle3D)nextLayer, getFilter(), (Cube)null, (Cube)null, true);
		return result != null ? nextLayer : null;
	}


	@Override
	public ConvLayerSingle forward(ConvLayerSingle nextLayer, Filter filter) {
		NeuronRaster result = forward(this, (ConvLayerSingle3D)nextLayer, filter, (Cube)null, (Cube)null, true);
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
	static NeuronRaster forward(ConvLayerSingle3D thisLayer, ConvLayerSingle3D nextLayer, Filter f, Cube thisFilterRegion, Cube nextFilterRegion, boolean nextAffected) {
		if (thisLayer == null) return null;
		Filter3D filter = (f != null && f instanceof Filter3D) ? (Filter3D)f : thisLayer.getFilter3D();
		if (filter == null) {
			return ConvLayer2DAbstract.forward(thisLayer, nextLayer, f,
				thisFilterRegion != null ? thisFilterRegion.toRectangle() : null,
				nextFilterRegion != null ? nextFilterRegion.toRectangle() : null,
				nextAffected);
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
		int thisWidth = thisLayer.getWidth();
		int thisHeight = thisLayer.getHeight();
		int thisDepth = thisLayer.getDepth();
		int thisBlockWidth = filter.isMoveStride() ? thisWidth / filterStrideWidth : thisWidth;
		int thisBlockHeight = filter.isMoveStride() ? thisHeight / filterStrideHeight : thisHeight;
		int thisBlockDepth = filter.isMoveStride() ? thisDepth / filterStrideDepth : thisDepth;
		int nextWidth = nextLayer.getWidth();
		int nextHeight = nextLayer.getHeight();
		int nextDepth = nextLayer.getDepth();
		Function activateRef = nextLayer.getActivateRef();
		activateRef = activateRef == null ? thisLayer.getActivateRef() : activateRef;
		
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
			
			int nextIndexZ = nextZ*nextWidth*nextHeight;
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
					if (thisZ >= thisDepth || thisY >= thisHeight || thisX >= thisWidth) {
						nextNeurons[nextIndex].setValue(nextZero);
						continue;
					}

					//Checking region.
					if (thisFilterRegion != null && !thisFilterRegion.contains(thisX, thisY, thisZ)) continue;
					if (nextFilterRegion != null && !nextFilterRegion.contains(nextX, nextY, nextZ)) continue;
					
					//Filtering
					NeuronValue filteredValue = null;
					if (filter instanceof DeconvConvFilter)
						filteredValue = ((DeconvConvFilter3D)filter).apply(thisX, thisY, thisZ, thisLayer, nextX, nextY, nextZ, nextLayer);
					else
						filteredValue = filter.apply(thisX, thisY, thisZ, thisLayer);
					
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
		
		if (filter instanceof DeconvConvFilter) {
			for (ConvNeuron nextNeuron : nextNeurons) {
				if (nextNeuron.getValue() == null) nextNeuron.setValue(nextZero);
				if (nextNeuron.getInput() == null) nextNeuron.setInput(nextZero);
			}
		}

		if ((!(thisLayer instanceof ConvLayer3DAbstract)) || (thisFilterRegion == null && nextFilterRegion == null))
			return new NeuronRaster(nextLayer.getNeuronChannel(), nextNeurons, new Size(nextWidth, nextHeight, nextDepth, 1));
		
		Cube nextRegion = null;
		if (thisFilterRegion == null && nextFilterRegion == null)
			nextRegion = ((ConvLayer3DAbstract)thisLayer).getNextRegion(new Cube(0, 0, 0, thisWidth, thisHeight, thisDepth));
		else if (thisFilterRegion != null)
			nextRegion = ((ConvLayer3DAbstract)thisLayer).getNextRegion(thisFilterRegion);
		else
			nextRegion = nextFilterRegion;
		if (nextRegion == null) return new NeuronRaster(nextLayer.getNeuronChannel(), nextNeurons, new Size(nextWidth, nextHeight, nextDepth, 1));

		ConvNeuron[] regionNeurons = new ConvNeuron[nextRegion.width*nextRegion.height*nextRegion.depth];
		int regionIndex = 0;
		int rdepth = nextRegion.z + nextRegion.depth;
		int rheight = nextRegion.y + nextRegion.height;
		int rwidth = nextRegion.x + nextRegion.width;
		for (int nextZ = nextRegion.z; nextZ < rdepth; nextZ++) {
			int nextIndexZ = nextZ*nextHeight*nextWidth;
			for (int nextY = nextRegion.y; nextY < rheight; nextY++) {
				int nextIndexY = nextIndexZ + nextY*nextWidth;
				for (int nextX = nextRegion.x; nextX < rwidth; nextX++) {
					int nextIndex = nextIndexY + nextX;
					regionNeurons[regionIndex] = nextNeurons[nextIndex];
					regionIndex++;
				}
			}
		}
		return new NeuronRaster(nextLayer.getNeuronChannel(), regionNeurons, new Size(nextRegion.width, nextRegion.height, nextRegion.depth, 1));
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
	public static NeuronRaster forward(NeuronValue[] input, ConvLayerSingle3D thisLayer, ConvLayerSingle3D nextLayer, Filter f, Cube thisFilterRegion, Cube nextFilterRegion, boolean nextAffected) {
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
		ConvLayerSingle3D smallLayer = null, largeLayer = null;
		if (this.length() >= ((ConvLayerSingle)nextLayer).length()) {
			smallLayer = (ConvLayerSingle3D)nextLayer;
			largeLayer = this;
		}
		else {
			smallLayer = this;
			largeLayer = (ConvLayerSingle3D)nextLayer;
		}
		
		maxIteration = maxIteration > 0 ? maxIteration : Network.LEARN_MAX_ITERATION_MAX;
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
	static BiasFilter learnFilter(ConvLayerSingle3D smallLayer, ConvLayerSingle3D largeLayer, BiasFilter initialFilter, boolean learningBias, double learningRate) {
		if (smallLayer == null || largeLayer == null) return null;
		if (largeLayer.getDepth() <= 1) return ConvLayer2DAbstract.learnFilter(smallLayer, largeLayer, initialFilter, learningBias, learningRate);
		learningRate = Double.isNaN(learningRate) || learningRate <= 0 || learningRate > 1 ? 1 : learningRate;
		
		ConvLayerSingle3D thisLayer = largeLayer, nextLayer = smallLayer;
		int n = Math.min(largeLayer.getWidth()/smallLayer.getWidth(), largeLayer.getHeight()/smallLayer.getHeight());
		n = Math.min(n, largeLayer.getDepth()/smallLayer.getDepth());
		boolean isMoveStride = true;
		if (n <= 1) {
			n = 3;
			isMoveStride = false;
		}
		
		NeuronValue zero = thisLayer.newNeuronValue().zero();
		ProductFilter3D filter = null;
		ProductFilter3D initialProductFilter = ((initialFilter != null) && (initialFilter.filter != null) && (initialFilter.filter instanceof ProductFilter3D)) ?
			(ProductFilter3D)(initialFilter.filter) : null;
		if (initialProductFilter == null || initialProductFilter.width() != n || initialProductFilter.height() != n || initialProductFilter.depth() != n) {
			NeuronValue[][][] kernel = new NeuronValue[n][n][n];
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					for (int k = 0; k < n; k++) kernel[i][j][k] = zero;
				}
			}
			filter = ProductFilter3D.create(kernel, zero.unit());
		}
		else {
			NeuronValue[][][] kernel = new NeuronValue[n][n][n];
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					for (int k = 0; k < n; k++) kernel[i][j][k] = initialProductFilter.getKernel()[i][j][k];
				}
			}
			filter = ProductFilter3D.create(kernel, initialProductFilter.getWeight());
		}
		
		filter.setMoveStride(isMoveStride);
		NeuronValue[][][] kernel = filter.getKernel();
		
		NeuronValue bias = null;
		if ((initialFilter != null) && (initialFilter.bias != null))
			bias = initialFilter.bias;
		else
			bias = zero; 
		
		int filterStrideWidth = filter.getStrideWidth();
		int filterStrideHeight = filter.getStrideHeight();
		int filterStrideDepth = filter.getStrideDepth();
		int thisWidth = thisLayer.getWidth();
		int thisHeight = thisLayer.getHeight();
		int thisDepth = thisLayer.getDepth();
		int thisBlockWidth = filter.isMoveStride() ? thisWidth / filterStrideWidth : thisWidth;
		int thisBlockHeight = filter.isMoveStride() ? thisHeight / filterStrideHeight : thisHeight;
		int thisBlockDepth = filter.isMoveStride() ? thisDepth / filterStrideDepth : thisDepth;
		int nextWidth = nextLayer.getWidth();
		int nextHeight = nextLayer.getHeight();
		int nextDepth = nextLayer.getDepth();
	
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
					
					NeuronValue filteredNextValue = filter.apply(thisX, thisY, thisZ, thisLayer);
					if (filteredNextValue == null) continue;
					Function nextActivateRef = nextLayer.getActivateRef(); 
					if (nextActivateRef != null) filteredNextValue = nextActivateRef.evaluate(filteredNextValue); 
					
					NeuronValue realNextValue = nextLayer.get(nextX, nextY, nextZ).getValue();
					NeuronValue error = realNextValue.subtract(filteredNextValue);
					if (nextActivateRef != null)
						error = error.multiplyDerivative(nextActivateRef.derivative(filteredNextValue));
					
					//Learning kernel.
					for (int i = 0; i < n; i++) {
						for (int j = 0; j < n; j++) {
							for (int k = 0; k < n; k++) {
								NeuronValue thisValue = thisLayer.get(thisX+k, thisY+j, thisZ+i).getValue();
								NeuronValue delta = error.multiply(thisValue).multiply(learningRate);
								kernel[i][j][k] = kernel[i][j][k].add(delta);
							}
						}
					}
					
					//Learning bias.
					if (learningBias) bias = bias.add(error.multiply(learningRate));
				} //End X.
			} //End Y.
		} //End Z.
		
		return new BiasFilter(filter, learningBias?bias:null);
	}


}


