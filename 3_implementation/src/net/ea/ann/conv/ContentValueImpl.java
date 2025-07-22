/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import java.util.Arrays;
import java.util.List;

import net.ea.ann.conv.filter.Filter;
import net.ea.ann.core.Id;
import net.ea.ann.core.NormSupporter;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.function.FunctionInvertible;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueComposite;
import net.ea.ann.core.value.WeightValue;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Size;

/**
 * This class is default implementation of content value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ContentValueImpl extends ContentImpl implements ContentValue, NormSupporter {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Zero.
	 */
	private static ContentValueImpl zero = null;
	
	
	/**
	 * Zero.
	 */
	private static ContentValueImpl unit = null;

	
	/**
	 * Constructor with neuron channel, activation function, width, height, depth, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param size layer size.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	public ContentValueImpl(int neuronChannel, Function activateRef, Size size, Filter filter, Id idRef) {
		super(neuronChannel, activateRef, size, filter, idRef);
	}

	
	/**
	 * Constructor with neuron channel, activation function, width, height, depth, and filter.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param size layer size.
	 * @param filter kernel filter.
	 */
	public ContentValueImpl(int neuronChannel, Function activateRef, Size size, Filter filter) {
		this(neuronChannel, activateRef, size, filter, null);
	}

	
	/**
	 * Constructor with neuron channel, activation function, width, height, and depth.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param size layer size.
	 */
	public ContentValueImpl(int neuronChannel, Function activateRef, Size size) {
		this(neuronChannel, activateRef, size, null, null);
	}

	
	/**
	 * Default constructor with neuron channel, activation function, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param activateRef activation function.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	ContentValueImpl(int neuronChannel, Function activateRef, Filter filter, Id idRef) {
		super(neuronChannel, activateRef, filter, idRef);
	}


	@Override
	public ContentImpl newContent(Size newSize) {
		return new ContentValueImpl(neuronChannel, activateRef, newSize, filter, idRef);
	}


	/**
	 * Creating a new content with neuron channel, normalization flag, size, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param size layer size.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 */
	ContentValueImpl newContent(int neuronChannel, Size size, Filter filter, Id idRef) {
		Function activateRef = Raster.toConvActivationRef(neuronChannel, isNorm());
		return new ContentValueImpl(neuronChannel, activateRef, size, filter, idRef);
	}

	
	/**
	 * Getting atomic zero value.
	 * @return atomic zero value.
	 */
	private NeuronValue getAtomicZeroValue() {
		return getBias().zero();
	}
	
	
	@Override
	public NeuronValue zero() {
		if (zero == this) return zero;
		if (zero != null && zero.getSize().equals(this.getSize()) && zero.getAtomicZeroValue() == this.getAtomicZeroValue()) return zero;
		zero = (ContentValueImpl)this.valueOf(0);
		return zero;
	}


	@Override
	public NeuronValue unit() {
		if (unit == this) return unit;
		if (unit != null && unit.getSize().equals(this.getSize()) && unit.getAtomicZeroValue() == this.getAtomicZeroValue()) return unit;
		unit = (ContentValueImpl)this.valueOf(1);
		return unit;
	}


	@Override
	public boolean isNorm() {
		return activateRef != null && activateRef instanceof NormSupporter ? ((NormSupporter)activateRef).isNorm() : false;
	}


	@Override
	public NeuronValue resize(int newDim) {
		int thisDim = dim();
		if (newDim == thisDim)
			return this;
		else if (newDim > thisDim)
			return (NeuronValue)increaseDim(newDim);
		else
			return (NeuronValue)decreaseDim(newDim);
	}


	@Override
	public NeuronValue resizeByChannel(int newChannel) {
		if (newChannel <= 0 || newChannel == getNeuronChannel()) return this;
		
		NeuronValue[] data = getData();
		NeuronValue[] newData = new NeuronValue[data.length];
		for (int i = 0; i < data.length; i++) {
			NeuronValue[] array = new NeuronValue[] {data[i]};
			if (newChannel < getNeuronChannel())
				newData[i] = NeuronValue.flattenByChannel(array, newChannel)[0];
			else if (data[i] instanceof NeuronValueComposite)
				newData[i] = ((NeuronValueComposite)data[i]).aggregateByChannel(array);
			else
				newData[i] = data[i].aggregate(array);
		}
		ContentValue content = newContent(newChannel, getSize(), null, idRef);
		content.setData(newData);
		return content;
	}


	@Override
	public NeuronValue duplicate() {
		return (NeuronValue)super.duplicateContent();
	}


	@Override
	public boolean equals(NeuronValue value) {
		NeuronValue[] thisData = this.getData();
		NeuronValue[] otherData = ((Content)value).getData();
		if (thisData.length != otherData.length) return false;
		for (int i = 0; i < thisData.length; i++) {
			if (!thisData[i].equals(otherData[i])) return false;
		}
		return true;
	}


	/**
	 * This class is default implementation of content weight.
	 * 
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	public static class ContentWeightImpl extends ContentImpl implements ContentWeight {

		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Zero.
		 */
		private static ContentWeightImpl zero = null;
		
		/**
		 * Zero.
		 */
		private static ContentWeightImpl unit = null;

		/**
		 * Constructor with neuron channel, activation function, width, height, depth, filter, and ID reference.
		 * @param neuronChannel neuron channel or depth.
		 * @param activateRef activation function.
		 * @param size layer size.
		 * @param filter kernel filter.
		 * @param idRef ID reference.
		 */
		public ContentWeightImpl(int neuronChannel, Function activateRef, Size size, Filter filter, Id idRef) {
			super(neuronChannel, activateRef, size, filter, idRef);
		}

		/**
		 * Constructor with neuron channel, activation function, width, height, depth, and filter.
		 * @param neuronChannel neuron channel or depth.
		 * @param activateRef activation function.
		 * @param size layer size.
		 * @param filter kernel filter.
		 */
		public ContentWeightImpl(int neuronChannel, Function activateRef, Size size, Filter filter) {
			this(neuronChannel, activateRef, size, filter, null);
		}

		/**
		 * Constructor with neuron channel, activation function, width, height, and depth.
		 * @param neuronChannel neuron channel or depth.
		 * @param activateRef activation function.
		 * @param size layer size.
		 */
		public ContentWeightImpl(int neuronChannel, Function activateRef, Size size) {
			this(neuronChannel, activateRef, size, null, null);
		}

		/**
		 * Default constructor with neuron channel, activation function, and ID reference.
		 * @param neuronChannel neuron channel or depth.
		 * @param activateRef activation function.
		 * @param filter kernel filter.
		 * @param idRef ID reference.
		 */
		ContentWeightImpl(int neuronChannel, Function activateRef, Filter filter, Id idRef) {
			super(neuronChannel, activateRef, filter, idRef);
		}

		@Override
		public ContentImpl newContent(Size newSize) {
			return new ContentWeightImpl(neuronChannel, activateRef, newSize, filter, idRef);
		}

		/**
		 * Getting atomic zero value.
		 * @return atomic zero value.
		 */
		private NeuronValue getAtomicZeroValue() {
			return getBias().zero();
		}
		
		@Override
		public WeightValue zeroW() {
			if (zero == this) return zero;
			if (zero != null && zero.getSize().equals(this.getSize()) && zero.getAtomicZeroValue() == this.getAtomicZeroValue()) return zero;
			zero = (ContentWeightImpl)this.valueOf(0);
			return zero;
		}

		@Override
		public WeightValue unitW() {
			if (unit == this) return unit;
			if (unit != null && unit.getSize().equals(this.getSize()) && unit.getAtomicZeroValue() == this.getAtomicZeroValue()) return unit;
			unit = (ContentWeightImpl)this.valueOf(1);
			return unit;
		}

		@Override
		public NeuronValue toValue() {
			ContentValueImpl contentValue = new ContentValueImpl(neuronChannel, activateRef, getSize(), filter, idRef);
			return (NeuronValue)contentValue.newContent(getData(), getBias());
		}

		@Override
		public WeightValue addValue(NeuronValue value) {
			return (WeightValue)super.add((Content)value);
		}

		@Override
		public WeightValue subtractValue(NeuronValue value) {
			return (WeightValue)super.subtract((Content)value);
		}

		/**
		 * Converting double value to weight value.
		 * @param value specific double value.
		 * @return weight value.
		 */
		private WeightValue valueOf(double value) {
			NeuronValue[] thisData = this.getData();
			NeuronValue[] newData = new NeuronValue[thisData.length];
			for (int i = 0; i < newData.length; i++) newData[i] = thisData[i].valueOf(value);
			return (WeightValue)newContent(newData, this.getBias());
		}

	}
	
	
	@Override
	public WeightValue newWeightValue() {
		return new ContentWeightImpl(neuronChannel, activateRef, getSize(), filter, idRef);
	}


	@Override
	public WeightValue toWeightValue() {
		return (WeightValue) ((ContentWeightImpl)newWeightValue()).newContent(getData(), getBias());
	}


	@Override
	public NeuronValue negative() {
		NeuronValue[] thisData = this.getData();
		NeuronValue[] newData = new NeuronValue[thisData.length];
		for (int i = 0; i < newData.length; i++) newData[i] = thisData[i].negative();
		return (NeuronValue)newContent(newData, this.getBias());
	}


	@Override
	public boolean canInvert() {
		NeuronValue[] thisData = this.getData();
		if (thisData.length == 0) return false;
		for (int i = 0; i < thisData.length; i++) {
			if (!thisData[i].canInvert()) return false;
		};
		return true;
	}


	@Override
	public NeuronValue inverse() {
		NeuronValue[] thisData = this.getData();
		NeuronValue[] newData = new NeuronValue[thisData.length];
		for (int i = 0; i < newData.length; i++) newData[i] = thisData[i].inverse();
		return (NeuronValue)newContent(newData, this.getBias());
	}


	@Override
	public NeuronValue add(NeuronValue value) {
		return (NeuronValue)super.add((Content)value);
	}


	@Override
	public NeuronValue subtract(NeuronValue value) {
		return (NeuronValue)super.subtract((Content)value);
	}


	@Override
	public NeuronValue multiply(NeuronValue value) {
		return (NeuronValue)super.multiply((Content)value);
	}

	
	@Override
	public NeuronValue multiply(WeightValue value) {
		return (NeuronValue)super.multiply0(value);
	}


	@Override
	public NeuronValue multiply(double value) {
		return (NeuronValue)super.multiply0(value);
	}


	@Override
	public NeuronValue multiplyDerivative(NeuronValue derivative) {
		return (NeuronValue)super.multiplyDerivative((Content)derivative);
	}


	@Override
	public NeuronValue divide(NeuronValue value) {
		return (NeuronValue)operatorTwo((Content)value, Operator.divide);
	}


	@Override
	public NeuronValue divide(double value) {
		return (NeuronValue)super.divide0(value);
	}


	@Override
	public NeuronValue power(double exponent) {
		NeuronValue[] thisData = this.getData();
		NeuronValue[] newData = new NeuronValue[thisData.length];
		for (int i = 0; i < newData.length; i++) newData[i] = thisData[i].power(exponent);
		return (NeuronValue)newContent(newData, this.getBias());
	}


	@Override
	public NeuronValue sqrt() {
		NeuronValue[] thisData = this.getData();
		NeuronValue[] newData = new NeuronValue[thisData.length];
		for (int i = 0; i < newData.length; i++) newData[i] = thisData[i].sqrt();
		return (NeuronValue)newContent(newData, this.getBias());
	}


	@Override
	public NeuronValue exp() {
		NeuronValue[] thisData = this.getData();
		NeuronValue[] newData = new NeuronValue[thisData.length];
		for (int i = 0; i < newData.length; i++) newData[i] = thisData[i].exp();
		return (NeuronValue)newContent(newData, this.getBias());
	}


	@Override
	public NeuronValue log() {
		NeuronValue[] thisData = this.getData();
		NeuronValue[] newData = new NeuronValue[thisData.length];
		for (int i = 0; i < newData.length; i++) newData[i] = thisData[i].log();
		return (NeuronValue)newContent(newData, this.getBias());
	}


	@Override
	public double mean() {
		NeuronValue mean = (NeuronValue)super.mean0();
		return mean != null ? mean.mean() : Double.NaN;
	}


	@Override
	public double norm() {
		NeuronValue[] thisData = this.getData();
		if (thisData.length == 0) return 0;
		double norm = 0;
		NeuronValue[] newData = new NeuronValue[thisData.length];
		for (int i = 0; i < newData.length; i++) norm += thisData[i].norm();
		return norm / (double)thisData.length;
	}


	@Override
	public NeuronValue valueOf(double value) {
		NeuronValue[] thisData = this.getData();
		NeuronValue[] newData = new NeuronValue[thisData.length];
		for (int i = 0; i < newData.length; i++) newData[i] = thisData[i].valueOf(value);
		return (NeuronValue)newContent(newData, this.getBias());
	}


	@Override
	public NeuronValue min(NeuronValue value) {
		NeuronValue[] thisData = this.getData();
		NeuronValue[] otherData = ((Content)value).getData();
		NeuronValue[] newData = new NeuronValue[thisData.length];
		for (int i = 0; i < newData.length; i++) newData[i] = thisData[i].min(otherData[i]);
		return (NeuronValue)newContent(newData, this.getBias());
	}


	@Override
	public NeuronValue max(NeuronValue value) {
		NeuronValue[] thisData = this.getData();
		NeuronValue[] otherData = ((Content)value).getData();
		NeuronValue[] newData = new NeuronValue[thisData.length];
		for (int i = 0; i < newData.length; i++) newData[i] = thisData[i].max(otherData[i]);
		return (NeuronValue)newContent(newData, this.getBias());
	}


	@Override
	public boolean matrixIsInvertible(NeuronValue[][] matrix) {
		List<NeuronValue[][]> matrixList = toMatrixList((ContentValue[][]) matrix);
		if (matrixList == null || matrixList.size() == 0) return false;
		for (int i = 0; i < matrixList.size(); i++) {
			boolean invertible = NeuronValue.isInvertible(matrixList.get(i));
			if (!invertible) return false;
		}
		return true;
	}


	@Override
	public NeuronValue matrixDet(NeuronValue[][] matrix) {
		List<NeuronValue[][]> matrixList = toMatrixList((ContentValue[][]) matrix);
		if (matrixList == null || matrixList.size() == 0) return null;

		ContentValue detContentValue = (ContentValue)newContent(getSize());
		for (int i = 0; i < detContentValue.length(); i++) {
			NeuronValue det = NeuronValue.det(matrixList.get(i));
			detContentValue.set(i, det);
		}
		return detContentValue;
	}


	@Override
	public NeuronValue[][] matrixInverse(NeuronValue[][] matrix) {
		List<NeuronValue[][]> matrixList = toMatrixList((ContentValue[][]) matrix);
		if (matrixList == null || matrixList.size() == 0) return null;

		List<NeuronValue[][]> inverseList = Util.newList(matrixList.size());
		for (int i = 0; i < matrixList.size(); i++) {
			NeuronValue[][] inverse = NeuronValue.inverse(matrixList.get(i));
			if (inverse == null || inverse.length == 0) return null;
			
			inverseList.add(inverse);
		}
		return fromMatrixList(inverseList, this);
	}


	@Override
	public NeuronValue[][] matrixSqrt(NeuronValue[][] matrix) {
		List<NeuronValue[][]> matrixList = toMatrixList((ContentValue[][]) matrix);
		if (matrixList == null || matrixList.size() == 0) return null;

		List<NeuronValue[][]> sqrtList = Util.newList(matrixList.size());
		for (int i = 0; i < matrixList.size(); i++) {
			NeuronValue[][] sqrt = NeuronValue.sqrt(matrixList.get(i));
			if (sqrt == null || sqrt.length == 0) return null;
			
			sqrtList.add(sqrt);
		}
		return fromMatrixList(sqrtList, this);
	}


	@Override
	public NeuronValue[] flatten(int smallerDim) {
		return (NeuronValue[])super.splitDim(smallerDim);
	}


	@Override
	public NeuronValue[] flatten(NeuronValue[] array, int smallerDim) {
		if (array == null || array.length == 0 || smallerDim < 1) return array;
		if (smallerDim >= array[0].length()) return array;
		int ratio = array[0].length() / smallerDim;
		ratio = ratio < 1 ? 1 : ratio;
		
		NeuronValue[] result = new NeuronValue[ratio*array.length];
		for (int i = 0; i < array.length; i++) {
			NeuronValue[] flat = array[i].flatten(smallerDim);
			for (int j = 0; j < flat.length; j++) result[i*ratio + j] = flat[j];
		}
		return result;
	}


	@Override
	public NeuronValue[] flattenByChannel(int smallerChannel) {
		if (this.length() == 0 || smallerChannel <= 0) return new NeuronValue[] {this};
		NeuronValue firstValue = this.get(0).getValue();
		boolean composite = firstValue instanceof NeuronValueComposite;
		if (composite) return flatten(smallerChannel);
		if (smallerChannel >= firstValue.length()) return new NeuronValue[] {this};
		
		int k = firstValue.flatten(smallerChannel).length;
		List<NeuronValue[]> dataList = Util.newList(k);
		NeuronValue[] thisData = getData();
		for (int j = 0; j < k; j++) dataList.add(new NeuronValue[thisData.length]);
		
		for (int i = 0; i < thisData.length; i++) {
			NeuronValue[] flatten = thisData[i].flatten(smallerChannel);
			for (int j = 0; j < flatten.length; j++) dataList.get(j)[i] = flatten[j]; //flatten.length = k
		}

		NeuronValue[] flatten = new NeuronValue[k];
		for (int j = 0; j < k; j++) {
			ContentValue contentValue = newContent(smallerChannel, getSize(), null, idRef);
			contentValue.setData(dataList.get(j));
			flatten[j] = contentValue;
		}
		return flatten;
	}


	@Override
	public NeuronValue[] flattenByChannel(NeuronValue[] array, int smallerChannel) {
		if (array == null || array.length == 0 || smallerChannel <= 0) return array;
		NeuronValue firstValue = ((ContentValue)array[0]).get(0).getValue();
		boolean composite = firstValue instanceof NeuronValueComposite;
		if (composite) return flatten(array, smallerChannel);
		if (smallerChannel >= firstValue.length()) return array;
		
		int ratio = firstValue.length() / smallerChannel;
		ratio = ratio < 1 ? 1 : ratio;
		NeuronValue[] result = new NeuronValue[ratio*array.length];
		for (int i = 0; i < array.length; i++) {
			ContentValue contentValue = (ContentValue)array[i];
			NeuronValue[] flat = contentValue.flattenByChannel(smallerChannel);
			for (int j = 0; j < flat.length; j++) result[i*ratio + j] = flat[j];
		}
		return result;
	}


	@Override
	public NeuronValue aggregate(NeuronValue[] array) {
		Arrays.asList((Content[])array);
		return (NeuronValue)aggregate(Arrays.asList((Content[])array));
	}


	@Override
	public NeuronValue[] aggregate(NeuronValue[] array, int largerDim) {
		return NeuronValue.aggregateByDim(array, largerDim);
	}


	@Override
	public NeuronValue aggregateByChannel(NeuronValue[] array) {
		if (array == null || array.length == 0) return null;
		
		NeuronValue[] data = new NeuronValue[length()];
		for (int i = 0; i < data.length; i++) {
			NeuronValue[] values = new ContentValue[array.length];
			for (int j = 0; j < array.length; j++) values[j] = ((ContentValue)array[j]).get(i).getValue();
			if (values[0] instanceof NeuronValueComposite)
				data[i] = ((NeuronValueComposite)values[0]).aggregateByChannel(values);
			else
				data[i] = values[0].aggregate(values);
		}
		return newContent(data[0].length(), getSize(), null, idRef);
	}


	@Override
	public NeuronValue[] aggregateByChannel(NeuronValue[] array, int largerChannel) {
		return NeuronValue.aggregateByChannel(array, largerChannel);
	}


	@Override
	public NeuronValue evaluate(Function f) {
		NeuronValue[] thisData = this.getData();
		NeuronValue[] newData = new NeuronValue[thisData.length];
		for (int i = 0; i < newData.length; i++) newData[i] = thisData[i].evaluate(f);
		return (NeuronValue)newContent(newData, this.getBias());
	}


	@Override
	public NeuronValue derivative(Function f) {
		return (NeuronValue)super.derivative0(f);
	}


	@Override
	public NeuronValue evaluateInverse(FunctionInvertible f) {
		NeuronValue[] thisData = this.getData();
		NeuronValue[] newData = new NeuronValue[thisData.length];
		for (int i = 0; i < newData.length; i++) newData[i] = thisData[i].evaluateInverse(f);
		return (NeuronValue)newContent(newData, this.getBias());
	}


	@Override
	public NeuronValue derivativeInverse(FunctionInvertible f) {
		NeuronValue[] thisData = this.getData();
		NeuronValue[] newData = new NeuronValue[thisData.length];
		for (int i = 0; i < newData.length; i++) newData[i] = thisData[i].derivativeInverse(f);
		return (NeuronValue)newContent(newData, this.getBias());
	}


	/**
	 * Converting list of value matrices to value matrix.
	 * @param matrixList list of value matrices.
	 * @param creator creator.
	 * @return value matrix.
	 */
	public static ContentValue[][] fromMatrixList(List<NeuronValue[][]> matrixList, ContentValue creator) {
		if (matrixList == null || matrixList.size() == 0) return null;
		
		int length = matrixList.size();
		NeuronValue[][] first = matrixList.get(0);
		ContentValue[][] matrix = new ContentValue[first.length][];
		for (int i = 0; i < first.length; i++) {
			matrix[i] = new ContentValue[first[i].length];
			
			for (int j = 0; j < first[i].length; j++) {
				matrix[i][j] = (ContentValue)creator.newContent(creator.getSize());
				for (int d = 0; d < length; d++) matrix[i][j].set(d, matrixList.get(d)[i][j]);
			}
		}
		return matrix;
	}

	
	/**
	 * Converting value matrix to list of value matrices.
	 * @param matrix value matrix.
	 * @return list of value matrices.
	 */
	public static List<NeuronValue[][]> toMatrixList(ContentValue[][] matrix) {
		if (matrix == null || matrix.length == 0) return null;
		
		int length = matrix[0][0].length();
		List<NeuronValue[][]> matrixList = Util.newList(length);
		for (int d = 0; d < length; d++) matrixList.add(new NeuronValue[matrix.length][]);
		
		for (int i = 0; i < matrix.length; i++) {
			for (int d = 0; d < length; d++) matrixList.get(d)[i] = new NeuronValue[matrix[i].length];

			for (int j = 0; j < matrix[i].length; j++) {
				ContentValue value = ((ContentValue)matrix[i][j]);
				for (int d = 0; d < length; d++) matrixList.get(d)[i][j] = value.get(d).getValue();
			}
		}
		return matrixList;
	}


	/**
	 * Creating content with neuron channel, activation function, width, height, depth, filter, and ID reference.
	 * @param neuronChannel neuron channel or depth.
	 * @param isNorm normalization flag.
	 * @param size layer size.
	 * @param filter kernel filter.
	 * @param idRef ID reference.
	 * @return created content value.
	 */
	public static ContentValueImpl create(int neuronChannel, boolean isNorm, Size size, Filter filter, Id idRef) {
		int width = size.width < 1 ? 1 : size.width;
		int height = size.height < 1 ? 1 : size.height;
		int depth = size.depth < 1 ? 1 : size.depth;
		int time = size.time < 1 ? 1 : size.time;
		neuronChannel = neuronChannel < 1 ? 1 : neuronChannel;
		Function activateRef = Raster.toConvActivationRef(neuronChannel, isNorm);
		size = new Size(width, height, depth, time);
		return new ContentValueImpl(neuronChannel, activateRef, size, filter, idRef);
	}

	
	/**
	 * Creating content value with neuron channel and size.
	 * @param neuronChannel neuron channel or depth.
	 * @param isNorm normalization flag.
	 * @param size layer size.
	 * @return created content value.
	 */
	public static ContentValueImpl create(int neuronChannel, boolean isNorm, Size size) {
		return create(neuronChannel, isNorm, size, null, null);
	}


}



