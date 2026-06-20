/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value;

import java.io.Serializable;

import net.ea.ann.core.function.Function;

/**
 * This interface represent similarity between two objects.
 * @author Loc Nguyen
 *
 * @param <T> object type.
 */
public interface Sim<T> extends Serializable, Cloneable {

	
	/**
	 * Calculating similarity between two objects.
	 * @param o1 first object.
	 * @param o2 second object.
	 * @return similarity between two objects.
	 */
	Object sim(T o1, T o2);
	
	
	/**
	 * Calculating similarity derivative between two objects.
	 * @param o1 first object.
	 * @param o2 second object.
	 * @param first flag to indicate the first of the second variable to be taken derivative.
	 * @return similarity derivative between two objects.
	 */
	Object dsim(T o1, T o2, boolean first);
	
	
	/**
	 * Calculating similarity derivative between two objects.
	 * @param o1 first object.
	 * @param o2 second object.
	 * @param f function.
	 * @param prev1 the previous of first object which can be null.
	 * @param prev2 the previous of second object which can be null.
	 * @return similarity derivative between two objects.
	 */
	Object dsim(T o1, T o2, Function f, T prevo1, T prevo2);


	/**
	 * This interface represent similarity between two matrices.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	interface MatrixSim extends Sim<Matrix> {
		
		@Override
		NeuronValue sim(Matrix o1, Matrix o2);

		@Override
		Matrix dsim(Matrix o1, Matrix o2, boolean first);
		
		@Override
		Matrix dsim(Matrix o1, Matrix o2, Function f, Matrix prevo1, Matrix prevo2);
		
	}
	
	
	/**
	 * This class implement distance-based similarity between two matrices.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	static class DistanceMatrixSim implements MatrixSim {

		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Default constructor.
		 */
		public DistanceMatrixSim() {}
		
		@Override
		public NeuronValue sim(Matrix o1, Matrix o2) {
			if (o1.rows() != o2.rows() || o1.columns() != o2.columns()) {
				o1 = o1.columns() > 1 ? o1.vec() : o1;
				o2 = o2.columns() > 1 ? o2.vec() : o2;
			}
			if (o1.rows() != o2.rows()) throw new IllegalArgumentException();
			
			Matrix d = o1.subtract(o2);
			NeuronValue sim = d.transpose().multiply(d).get(0, 0);
			return sim.divide(2).negative();
		}
		
		@Override
		public Matrix dsim(Matrix o1, Matrix o2, boolean first) {
			int vecRows = first ? o1.rows() : o2.rows();
			if (o1.rows() != o2.rows() || o1.columns() != o2.columns()) {
				o1 = o1.columns() > 1 ? o1.vec() : o1;
				o2 = o2.columns() > 1 ? o2.vec() : o2;
			}
			if (o1.rows() != o2.rows()) throw new IllegalArgumentException();

			Matrix d = first ? o1.subtract(o2).negative0() : o1.subtract(o2);
			return d.rows() != vecRows ? d.vecInverse(vecRows) : d;
		}
		

		@Override
		public Matrix dsim(Matrix o1, Matrix o2, Function f, Matrix prevo1, Matrix prevo2) {
			Matrix dsim = dsim(o1, o2, true);
			if (f == null) return dsim;
			if (prevo1 == null && prevo2 == null) throw new IllegalArgumentException();
			
			Matrix derivative = null;
			if (prevo1 != null && prevo2 != null)
				derivative = prevo1.derivativeWise(f).subtract(prevo2.derivativeWise(f));
			else if (prevo1 != null)
				derivative = prevo1.derivativeWise(f);
			else
				derivative = prevo2.derivativeWise(f).negative0();
			return derivative.multiplyWise(dsim);
		}
	
	}
	
	
	/**
	 * This class implement product-based similarity between two matrices.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	static class ProductMatrixSim implements MatrixSim {

		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Default constructor.
		 */
		public ProductMatrixSim() {}
		
		@Override
		public NeuronValue sim(Matrix o1, Matrix o2) {
			if (o1.rows() != o2.rows() || o1.columns() != o2.columns()) {
				o1 = o1.columns() > 1 ? o1.vec() : o1;
				o2 = o2.columns() > 1 ? o2.vec() : o2;
			}
			if (o1.rows() != o2.rows()) throw new IllegalArgumentException();
			
			return o1.transpose().multiply(o2).get(0, 0);
		}
		
		@Override
		public Matrix dsim(Matrix o1, Matrix o2, boolean first) {
			return first ? o1 : o2;
		}
		

		@Override
		public Matrix dsim(Matrix o1, Matrix o2, Function f, Matrix prevo1, Matrix prevo2) {
			if (f == null) return o1.add(o2);
			if (prevo1 == null && prevo2 == null) throw new IllegalArgumentException();
			
			Matrix d1 = prevo1 != null ? prevo1.derivativeWise(f).multiplyWise(o1) : null;
			Matrix d2 = prevo2 != null ? prevo2.derivativeWise(f).multiplyWise(o2) : null;
			if (d1 != null && d2 != null)
				return d1.add(d2);
			else if (d1 != null)
				return d1;
			else if (d2 != null)
				return d2;
			else
				return null;
		}
	
	}


}
