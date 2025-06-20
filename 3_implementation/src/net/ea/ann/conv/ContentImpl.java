/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import java.util.List;

import net.ea.ann.conv.filter.Filter;
import net.ea.ann.conv.filter.FilterAssoc;
import net.ea.ann.conv.filter.FilterFactoryImpl;
import net.ea.ann.conv.filter.FilterAssoc.PlainRaster;
import net.ea.ann.core.Id;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.WeightValue;
import net.ea.ann.raster.Cube;
import net.ea.ann.raster.NeuronRaster;
import net.ea.ann.raster.RasterAssoc;
import net.ea.ann.raster.Size;

/**
 * This class is the default implementation of content.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ContentImpl extends ConvLayer4DImpl implements Content {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Maximum dimension.
	 */
	public final static int MAX_DIM = 4;
	
	
	/**
	 * Internal list of previous contents.
	 */
	private List<Content> prevContents = Util.newList(0);

	
	/**
	 * Internal list of next contents.
	 */
	protected List<Content> nextContents = Util.newList(0);
	
	
	/**
	 * Constructor with neuron channel, activation function, size, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function, which is often activation function related to convolutional pixel like ReLU function.
	 * @param size layer size.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	protected ContentImpl(int neuronChannel, Function activateRef, Size size, Filter filter, Id idRef) {
		super(neuronChannel, activateRef, size.width, size.height, size.depth, size.time, filter, idRef);
	}

	
	/**
	 * Constructor with neuron channel, activation function, size, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function, which is often activation function related to convolutional pixel like ReLU function.
	 * @param size layer size.
	 * @param filter kernel filter.
	 */
	protected ContentImpl(int neuronChannel, Function activateRef, Size size, Filter filter) {
		this(neuronChannel, activateRef, size, filter, null);
	}

	
	/**
	 * Constructor with neuron channel, activation function, and size.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function, which is often activation function related to convolutional pixel like ReLU function.
	 * @param size layer size.
	 */
	protected ContentImpl(int neuronChannel, Function activateRef, Size size) {
		this(neuronChannel, activateRef, size, null, null);
	}

	
	/**
	 * Default constructor with neuron channel, activation function, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function, which is often activation function related to convolutional pixel like ReLU function.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	ContentImpl(int neuronChannel, Function activateRef, Filter filter, Id idRef) {
		super(neuronChannel, activateRef, filter, idRef);
	}


	@Override
	public Size getSize() {
		return new Size(this.getWidth(), this.getHeight(), this.getDepth(), this.getTime());
	}
	
	
	@Override
	public ContentImpl newContent(Size newSize) {
		return new ContentImpl(neuronChannel, activateRef, newSize, filter, idRef);
	}
	
	
	/**
	 * Creating a new content from data, size, and bias.
	 * @param data specified data.
	 * @param newSize specified size.
	 * @param bias specified bias.
	 * @return new content.
	 */
	private ContentImpl newContent(Size newSize, NeuronValue[] data, NeuronValue bias) {
		ContentImpl content = newContent(newSize);
		if (data != null) content.setData(data);
		if (bias != null) content.setBias(bias);
		return content;
	}

	
	@Override
	public Content newContent(NeuronValue[] data, NeuronValue bias) {
		return newContent(getSize(), data, bias);
	}
	
	
	@Override
	public Content duplicateContent() {
		return newContent(getData(), bias);
	}
	
	
	@Override
	public Content resizeContent(Size newSize) {
		return resizeContent(newSize, false);
	}


	/**
	 * Making this content fit to specified new size.
	 * @param newSize specified new size.
	 * @param allowZoom flag to indicate whether allowing zooming.
	 * @return fitting contents.
	 */
	private ContentImpl resizeContent(Size newSize, boolean allowZoom) {
		Size thisSize = this.getSize(); 
		if (thisSize.equals(newSize)) return this;
		
		NeuronValue zero = this.newNeuronValue().zero();
		if (!allowZoom) {
			int thisLength = thisSize.length();
			int newLength = newSize.length();
			NeuronValue[] newData = new NeuronValue[newLength];
			for (int t = 0; t < newSize.time; t++) {
				int thisIndexT = t*thisSize.width*thisSize.height*thisSize.depth;
				int newIndexT = t*newSize.width*newSize.height*newSize.depth;
				for (int z = 0; z < newSize.depth; z++) {
					int thisIndexZ = thisIndexT + z*thisSize.width*thisSize.height;
					int newIndexZ = newIndexT + z*newSize.width*newSize.height;
					for (int y = 0; y < newSize.height; y++) {
						int thisIndexY = thisIndexZ + y*thisSize.width;
						int newIndexY = newIndexZ + y*newSize.width;
						for (int x = 0; x < newSize.width; x++) {
							int thisIndex = thisIndexY + x;
							int newIndex = newIndexY + x;
							if (thisIndex < thisLength && newIndex < newLength)
								newData[newIndex] = this.get(x, y, z, t).getValue();
							else
								newData[newIndex] = zero;
						}
					}
				}
			}
			
			return newContent(newSize, newData, this.getBias());
		}

		int zoom = Filter.zoomRatioOf(this.getSize(), newSize);
		zoom = (int)(Math.pow(zoom, 0.25) + 0.5);
		if (zoom <= 1) return this.resizeContent(newSize, false);
		
		Filter filter = null;
		FilterFactoryImpl factory = new FilterFactoryImpl(this);
		if (this.length() < newSize.length())
			filter = factory.zoomIn(zoom, zoom, zoom, zoom);
		else
			filter = factory.zoomOut(zoom, zoom, zoom, zoom);
		
		FilterAssoc assoc = new FilterAssoc(this.getNeuronChannel(), this.getActivateRef(), filter);
		PlainRaster raster = assoc.apply4D(this.getData(), this.getSize(), false);
		return newContent(raster.size, raster.data, this.getBias()).resizeContent(newSize, false);
	}

	
	/**
	 * This enum represents operators. 
	 * @author Loc Nguyen
	 * @version 1.0
	 */
	protected enum Operator {

		/**
		 * Adding operator.
		 */
		add,

		/**
		 * Subtracting operator.
		 */
		subtract,

		/**
		 * Multiplication operator.
		 */
		multiply,

		/**
		 * Division operator.
		 */
		divide,
		
	}
	
	
	/**
	 * Making operator with two operands.
	 * @param content other content.
	 * @param operator specified operator.
	 * @return new content.
	 */
	protected Content operatorTwo(Content content, Operator operator) {
		return operatorTwo4D(content, operator);
	}
	
	
	/**
	 * Making operator with two operands.
	 * @param content other content.
	 * @param operator specified operator.
	 * @return new content.
	 */
	private Content operatorTwo4D(Content content, Operator operator) {
		NeuronValue zero = this.newNeuronValue().zero();
		Size size = this.getSize();
		Content otherContent = content.resizeContent(size);
		Content newContent = newContent(this.getData(), this.getBias());
		for (int t = 0; t < size.time; t++) {
			int indexT = t*this.getWidth()*this.getHeight()*this.getDepth();
			for (int z = 0; z < size.depth; z++) {
				int indexZ = indexT + z*this.getWidth()*this.getHeight();
				for (int y = 0; y < size.height; y++) {
					int indexY = indexZ + y*this.getWidth();
					for (int x = 0; x < size.width; x++) {
						int index = indexY + x;
						if (index >= newContent.length()) continue;
						NeuronValue value = newContent.get(index).getValue();
						NeuronValue otherValue = otherContent.get(x, y, z, t).getValue(); 
						switch(operator) {
						case add:
							value = value.add(otherValue);
							break;
						case subtract:
							value = value.subtract(otherValue);
							break;
						case multiply:
							value = value.multiply(otherValue);
							break;
						case divide:
							value = value.divide(otherValue);
							break;
						default:
							value = zero;
							break;
						}
						newContent.set(index, value);
					}
				}
			}
		}
		
		return newContent;
	}
	
	
	@Override
	public Content add(Content content) {
		return operatorTwo(content, Operator.add);
	}

	
	@Override
	public Content subtract(Content content) {
		return operatorTwo(content, Operator.subtract);
	}

	
	@Override
	public Content multiply(Content content) {
		return operatorTwo(content, Operator.multiply);
	}

	
	@Override
	public Content multiply0(WeightValue weight) {
		NeuronValue[] thisData = this.getData();
		NeuronValue[] newData = new NeuronValue[thisData.length];
		for (int i = 0; i < newData.length; i++) newData[i] = thisData[i].multiply(weight);
		return newContent(newData, this.getBias());
	}

	
	@Override
	public Content multiplyDerivative(Content content) {
		return multiply(content);
	}

	
	@Override
	public Content multiply0(double value) {
		NeuronValue[] thisData = this.getData();
		NeuronValue[] newData = new NeuronValue[thisData.length];
		for (int i = 0; i < newData.length; i++) newData[i] = thisData[i].multiply(value);
		return newContent(newData, this.getBias());
	}

	
	@Override
	public Content divide0(double value) {
		if (value == 0) return null;
		NeuronValue[] thisData = this.getData();
		NeuronValue[] newData = new NeuronValue[thisData.length];
		for (int i = 0; i < newData.length; i++) newData[i] = thisData[i].divide(value);
		return newContent(newData, this.getBias());
	}

	
	@Override
	public NeuronValue mean0() {
		NeuronValue[] thisData = this.getData();
		if (thisData.length == 0) return null;
		NeuronValue mean = newNeuronValue().zero();
		for (int i = 0; i < thisData.length; i++) mean = mean.add(thisData[i]);
		return mean.divide(thisData.length);
	}

	
	@Override
	public Content derivative0(Function f) {
		if (f == null) return null;
		NeuronValue[] thisData = this.getData();
		NeuronValue[] newData = new NeuronValue[thisData.length];
		for (int i = 0; i < newData.length; i++) newData[i] = thisData[i].derivative(f);
		return newContent(newData, this.getBias());
	}

	
	/**
	 * Getting previous contents.
	 * @return array of previous contents.
	 */
	protected Content[] getPrevContents() {
		return prevContents.toArray(new Content[] {});
	}
	
	
	@Override
	public boolean setNextLayer(ConvLayer nextLayer) {
		if (nextLayer == null)
			return super.setNextLayer(null);
		else if (nextLayer instanceof Content)
			return nextContents.contains(nextLayer) ? super.setNextLayer(nextLayer) : false;
		else
			return false;
	}


	@Override
	public int getNextContentSize() {
		return nextContents.size();
	}


	@Override
	public Content getNextContent(int index) {
		return nextContents.get(index);
	}


	@Override
	public int indexOfNextContent(Content nextContent) {
		return nextContents.indexOf(nextContent);
	}


	@Override
	public boolean addNextContent(Content nextContent) {
		if (nextContent == null || nextContents.contains(nextContent)) return false;
		boolean result = nextContents.add(nextContent);
		if (!result) return result;
		
		if (nextContent instanceof ContentImpl) {
			List<Content> prevContents = ((ContentImpl)nextContent).prevContents;
			if (!prevContents.contains(this)) prevContents.add(this);
		}
		
		if (getNextLayer() == null) setNextLayer(nextContent);
		return result;
	}


	@Override
	public Content removeNextContent(int index) {
		Content removedNextContent = nextContents.remove(index);
		if (removedNextContent == null) return removedNextContent;
		updateRemovedNextContent(removedNextContent);
		return removedNextContent;
	}

	
	@Override
	public boolean removeNextContent(Content nextContent) {
		boolean result = nextContents.remove(nextContent);
		if (!result) return result;
		updateRemovedNextContent(nextContent);
		return result;
	}


	/**
	 * Updating removed next content.
	 * @param removedNextContent removed next content.
	 */
	private void updateRemovedNextContent(Content removedNextContent) {
		if (removedNextContent == null || removedNextContent != nextLayer) return;
		
		if (removedNextContent instanceof ContentImpl) {
			List<Content> prevContents = ((ContentImpl)removedNextContent).prevContents;
			if (prevContents.contains(this)) prevContents.remove(this);
		}

		ConvLayer oldNextLayer = nextLayer;
		setNextLayer(null);
		if (oldNextLayer != null && nextContents.size() > 0) setNextLayer(nextContents.get(0));
	}

	
	@Override
	public void clearNextContents() {
		while (nextContents.size() > 0) {
			removeNextContent(0);
		}
	}


	@Override
	public ConvLayer forward() {
		if (nextContents.size() == 0) return null;
		for (Content nextContent : nextContents) forward(nextContent);
		return nextContents.get(nextContents.size()-1);
	}


	@Override
	public NeuronRaster forward(Content nextContent) {
		if (nextContent == null)
			return forward(this, (Content)getNextLayer(), getFilter(), (Cube)null, (Cube)null, true);
		else
			return forward(this, nextContent, getFilter(), (Cube)null, (Cube)null, true);
	}

	
	@Override
	public NeuronRaster forward(Content nextContent, Filter filter) {
		if (nextContent == null)
			return forward(this, (Content)getNextLayer(), filter, (Cube)null, (Cube)null, true);
		else
			return forward(this, nextContent, filter, (Cube)null, (Cube)null, true);
	}


	/**
	 * Aggregating a collection of contents.
	 * The important feature of this method is that these contents may not have the same size.
	 * @param contents a collection of contents.
	 * @return aggregated content.
	 */
	public static Content aggregate(Iterable<Content> contents) {
		if (contents == null) return null;
		List<ContentImpl> contentList = Util.newList(0);
		for (Content content : contents) {
			ContentImpl c = c(content);
			if (c != null && c.dim() > 0) contentList.add(c);
		}
		if (contentList.size() == 0) return null;
		if (contentList.size() == 1) return contentList.get(0);
		
		//Finding maximum dimension and the content whose dimension is such maximum dimension.
		int maxDim = 0; //Maximum dimension.
		ContentImpl maxContent = null; //The content whose dimension is the maximum dimension.
		for (ContentImpl content : contentList) {
			int dim = content.dim();
			if (dim > maxDim && (maxContent == null || maxContent.length() < content.length())) {
				maxDim = dim;
				maxContent = content;
			}
		}
		if (maxDim <= 0 || maxDim > MAX_DIM || maxContent == null) return null;
		
		//Making all contents to have the same maximum dimension.
		int maxDimLength = 0; //Length of the maximum dimension.
		List<ContentImpl> newContentList = Util.newList(0);
		boolean diff = false;
		for (ContentImpl content : contentList) {
			ContentImpl newContent = content;
			if (content.dim() == maxDim) {
				maxDimLength += content.lengthOfDim();
			}
			else {
				newContent = content.increaseDim(maxDim);
				maxDimLength += 2;
				diff = true;
			}
			
			newContent = (ContentImpl)newContent.resizeContent(maxContent.getSize());
			newContentList.add(newContent);
		}
		if (maxDimLength < 2) return null;
		
		NeuronValue[] aggregatedData = null;
		for (ContentImpl newContent : newContentList) {
			aggregatedData = NeuronValue.concatArray(aggregatedData, newContent.getData());
		}
		
		Size aggregatedSize = maxContent.getSize();
		if (diff || maxDim == MAX_DIM) {
			switch(maxDim) {
			case 4:
				aggregatedSize.time = maxDimLength;
				break;
			case 3:
				aggregatedSize.depth = maxDimLength;
				break;
			case 2:
				aggregatedSize.height = maxDimLength;
				break;
			case 1:
				aggregatedSize.width = maxDimLength;
				break;
			}
		}
		else {
			int newDimLength = newContentList.size();
			switch(maxDim) {
			case 3:
				aggregatedSize.time = newDimLength;
				break;
			case 2:
				aggregatedSize.depth = newDimLength;
				break;
			case 1:
				aggregatedSize.height = newDimLength;
				break;
			}
		}

		ContentImpl aggregatedContent = maxContent.newContent(aggregatedSize);
		aggregatedContent.setData(aggregatedData);
		return aggregatedContent;
	}
	
	
	/**
	 * Increasing dimension.
	 * @param largerDim larger dimension.
	 * @return new content with larger dimension.
	 */
	protected ContentImpl increaseDim(int largerDim) {
		if (largerDim <= 1 || largerDim > MAX_DIM || this.dim() <= 0 || largerDim <= this.dim()) return this;
		int thisDim = this.dim();
		ContentImpl newContent = this;
		for (int dim = thisDim + 1; dim <= largerDim; dim++) {
			newContent = newContent.increaseDim0(dim);
		}
		return newContent;
	}
	
	
	/**
	 * Increasing dimension.
	 * @param largerDim larger dimension.
	 * @return new content with larger dimension.
	 */
	private ContentImpl increaseDim0(int largerDim) {
		if (largerDim <= 1 || largerDim > MAX_DIM || this.dim() <= 0 || largerDim <= this.dim()) return this;
		Size newSize = getSize();
		switch(largerDim) {
		case 4:
			newSize.time = 2;
			break;
		case 3:
			newSize.depth = 2;
			break;
		case 2:
			newSize.height = 2;
			break;
		}
		
		NeuronValue[] thisData = this.getData();
		NeuronValue[] newData = new NeuronValue[2*thisData.length];
		RasterAssoc.copyRange1D(thisData, 0, thisData.length, newData);
		return newContent(newSize, newData, bias);
	}
	
	
	/**
	 * Decreasing dimension.
	 * @param smallerDim smaller dimension.
	 * @return new content with smaller dimension.
	 */
	protected ContentImpl decreaseDim(int smallerDim) {
		if (smallerDim <= 0 || smallerDim >= MAX_DIM || this.dim() <= 1 || smallerDim >= this.dim()) return this;
		int thisDim = this.dim();
		ContentImpl newContent = this;
		for (int dim = thisDim - 1; dim >= smallerDim; dim--) {
			newContent = newContent.decreaseDim0(dim);
		}
		return newContent;
	}

	
	/**
	 * Decreasing dimension.
	 * @param smallerDim smaller dimension.
	 * @return new content with smaller dimension.
	 */
	private ContentImpl decreaseDim0(int smallerDim) {
		if (smallerDim <= 0 || smallerDim >= MAX_DIM || this.dim() <= 1 || smallerDim >= this.dim()) return this;
		Size newSize = getSize();
		switch(smallerDim) {
		case 3:
			newSize.time = 1;
			break;
		case 2:
			newSize.depth = 1;
			break;
		case 1:
			newSize.height = 1;
			break;
		}
		
		NeuronValue[] thisData = this.getData();
		NeuronValue[] newData = new NeuronValue[thisData.length/2];
		RasterAssoc.copyRange1D(thisData, 0, newData.length, newData);
		return newContent(newSize, newData, bias);
	}

	
	/**
	 * Splitting by smaller dimension.
	 * @param smallerDim smaller dimension.
	 * @return new content with new dimension.
	 */
	protected ContentImpl[] splitDim(int smallerDim) {
		if (this.dim() <= 1 || smallerDim >= this.dim()) return new ContentImpl[] {this};
		List<ContentImpl> results = Util.newList(0);
		splitDim(this, this.dim()-1, smallerDim, results);
		return results.size() > 0 ? results.toArray(new ContentImpl[] {}) : new ContentImpl[] {this};
	}
	
	
	/**
	 * Splitting by smaller dimension.
	 * @param content content.
	 * @param smallerDim smaller dimension.
	 * @param dimToAdd the specified dimension that corresponding split content is added to results.
	 * @param results list of split contents.
	 */
	private static void splitDim(ContentImpl content, int smallerDim, int dimToAdd, List<ContentImpl> results) {
		if (smallerDim <= 0 || smallerDim >= MAX_DIM || content.dim() <= 1 || smallerDim >= content.dim()) return;
		if (dimToAdd > smallerDim) return;
		
		int thisDim = content.dim();
		for (int dim = thisDim - 1; dim >= smallerDim && dim >= dimToAdd; dim--) {
			ContentImpl[] splits = content.splitDim0(dim);
			if (dim == dimToAdd) {
				results.add(splits[0]);
				results.add(splits[1]);
			}
			else {
				splitDim(splits[0], dim-1, dimToAdd, results);
				splitDim(splits[1], dim-1, dimToAdd, results);
			}
		}
	}
	
	
	/**
	 * Splitting by smaller dimension.
	 * @param smallerDim smaller dimension.
	 * @return new content with new dimension.
	 */
	private ContentImpl[] splitDim0(int smallerDim) {
		if (smallerDim <= 0 || smallerDim >= MAX_DIM || this.dim() <= 1 || smallerDim >= this.dim()) return new ContentImpl[] {};
		Size newSize = getSize();
		switch(smallerDim) {
		case 3:
			newSize.time = 1;
			break;
		case 2:
			newSize.depth = 1;
			break;
		case 1:
			newSize.height = 1;
			break;
		}
		
		NeuronValue[] thisData = this.getData();
		int n = thisData.length/2;
		NeuronValue[] newData1 = new NeuronValue[n];
		RasterAssoc.copyRange1D(thisData, 0, n, newData1);
		ContentImpl v1 = newContent(newSize, newData1, bias);
		
		NeuronValue[] newData2 = new NeuronValue[n];
		RasterAssoc.copyRange1D(thisData, n, n, newData2);
		ContentImpl v2 = newContent(newSize, newData2, bias);
		
		return new ContentImpl[] {v1, v2};
	}

	
	/**
	 * Getting dimension.
	 * @return dimension of this content.
	 */
	public int dim() {
		if (time > 1)
			return 4;
		else if (depth > 1)
			return 3;
		else if (height > 1)
			return 2;
		else if (width >= 1)
			return 1;
		else
			return 0;
	}
	
	
	/**
	 * Dimension length of this content.
	 * @return dimension length of this content.
	 */
	private int lengthOfDim() {
		int dim = dim();
		if (dim <= 0 || dim > MAX_DIM) return 0;
		switch (dim) {
		case 4:
			return time;
		case 3:
			return depth;
		case 2:
			return height;
		case 1:
			return width;
		}
		
		return 0;
	}
	
	
	/**
	 * Converting normal content to specified content.
	 * @param content normal content.
	 * @return specified content.
	 */
	private static ContentImpl c(Content content) {
		if (content == null)
			return null;
		else if (content instanceof ContentImpl)
			return (ContentImpl)content;
		else {
			ContentImpl newContent = new ContentImpl(content.getNeuronChannel(), content.getActivateRef(), content.getSize(), content.getFilter(), content.getIdRef());
			newContent.setData(content.getData());
			newContent.setBias(content.getBias());
			return newContent;
		}
	}
	
	
	/**
	 * Creating content with neuron channel, activation function, size, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function, which is often activation function related to convolutional pixel like ReLU function.
	 * @param size layer size.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 * @return created content.
	 */
	public static ContentImpl create(int neuronChannel, Function activateRef, Size size, Filter filter, Id idRef) {
		int width = size.width < 1 ? 1 : size.width;
		int height = size.height < 1 ? 1 : size.height;
		int depth = size.depth < 1 ? 1 : size.depth;
		int time = size.time < 1 ? 1 : size.time;
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		return new ContentImpl(neuronChannel, activateRef, new Size(width, height, depth, time), filter, idRef);
	}


	/**
	 * Creating content with neuron channel, activation function, size, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function, which is often activation function related to convolutional pixel like ReLU function.
	 * @param size layer size.
	 * @param filter kernel filter.
	 * @return created content.
	 */
	public static ContentImpl create(int neuronChannel, Function activateRef, Size size, Filter filter) {
		return create(neuronChannel, activateRef, size, filter, null);
	}
	
	
	/**
	 * Creating content with neuron channel, activation function, width, height, and depth.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function, which is often activation function related to convolutional pixel like ReLU function.
	 * @param size layer size.
	 * @return created content.
	 */
	public static ContentImpl create(int neuronChannel, Function activateRef, Size size) {
		return create(neuronChannel, activateRef, size, null, null);
	}


}
