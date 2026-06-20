/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value;

import java.io.Serializable;

/**
 * This interface represents accumulator.
 * @author Loc Nguyen
 * @version 1.0
 *
 * @param <T> value type.
 */
public interface Accumulator<T> extends Cloneable, Serializable {

	/**
	 * Accumulating value.
	 * @param value value.
	 * @return this accumulator.
	 */
	Accumulator<T> accum(T value);
	
	
	/**
	 * Get accumulated value.
	 * @return accumulated value.
	 */
	T get();
	
	
	/**
	 * Returning count.
	 * @return count.
	 */
	int count();
	
	
	/**
	 * Getting mean value.
	 * @return mean value.
	 */
	T mean();
	
	
	/**
	 * Clearing.
	 */
	Accumulator<T> clear();
	
	
	/**
	 * Calculating mean and clearing.
	 * @return mean.
	 */
	default T meanAndClear() {
		T mean = mean();
		clear();
		return mean;
	}

	
	/**
	 * This class is abstract class of accumulator.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 * @param <T> value type.
	 */
	abstract class AccumulatorAbstract<T> implements Accumulator<T> {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;
		
		/**
		 * Internal sum.
		 */
		protected T sum = null;
		
		/**
		 * Count.
		 */
		protected int count = 0;
		
		/**
		 * Constructor with value.
		 * @param value value.
		 */
		public AccumulatorAbstract(T value) {
			if (value == null) return;
			this.sum = value;
			this.count = 1;
		}

		@Override
		public T get() {return sum;}

		@Override
		public Accumulator<T> clear() {
			sum = null;
			count = 0;
			return this;
		}
		
	}
	
	
	/**
	 * This class represents accumulator with matrix.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	class MatrixAccumulator extends AccumulatorAbstract<Matrix> {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Constructor with matrix value.
		 * @param value matrix value.
		 */
		public MatrixAccumulator(Matrix value) {super(value);}

		@Override
		public Accumulator<Matrix> accum(Matrix value) {
			if (value == null) return this;
			this.sum = this.sum != null ? this.sum.add(value) : value;
			this.count++;
			return this;
		}

		@Override
		public int count() {return count;}

		@Override
		protected Object clone() throws CloneNotSupportedException {
			// TODO Auto-generated method stub
			return super.clone();
		}

		@Override
		public Matrix mean() {
			return sum != null && count > 0 ? sum.divide0(count) : null;
		}
		
	}
	
	
	
}
