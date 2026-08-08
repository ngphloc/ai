/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.io.Serializable;

/**
 * This class represents parameter.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Parameter extends Cloneable, Serializable {

	
	/**
	 * Copying from other parameter.
	 * @param other other parameter.
	 * @return this parameter.
	 */
	Parameter pcopy(Parameter other);
	
	
	/**
	 * Adding other parameter.
	 * @param other other parameter.
	 * @return this parameter.
	 */
	Parameter padd(Parameter other);
	
	
	/**
	 * Subtract other parameter.
	 * @param other other parameter.
	 * @return this parameter.
	 */
	Parameter psubtract(Parameter other);
	
	
	/**
	 * Multiplying with factor.
	 * @param factor factor.
	 * @return this parameter.
	 */
	Parameter pmultiply(double factor);
	
	
	/**
	 * Initializing by randomizer.
	 * @param rnd randomizer.
	 * @return this parameter.
	 */
	Parameter pinit(Randomizer rnd);
	
	
	/**
	 * Initializing by randomizer.
	 * @param rnd randomizer.
	 * @return this parameter.
	 */
	Parameter pmultiplyRandom(Randomizer rnd);

	
	/**
	 * This interface represents cloneable parameter.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	interface CloneableParameter extends Parameter {

		/**
		 * Cloning parameter.
		 * @return cloned parameter.
		 * @throws CloneNotSupportedException if cloning is not supported.
		 */
		Object clone() throws CloneNotSupportedException;
		
	}
	
	
//	/**
//	 * This interface represents layer parameter.
//	 * @author Loc Nguyen
//	 * @version 1.0
//	 *
//	 */
//	interface LayerParameter extends Parameter {
//		
//		/**
//		 * Getting weight.
//		 * @return the weight.
//		 */
//		Weight getWeight();
//
//		/**
//		 * Getting bias.
//		 * @return bias.
//		 */
//		Matrix getBias();
//		
//		/**
//		 * Getting convolutional filter.
//		 * @return convolutional filter.
//		 */
//		Filter getFilter();
//
//		/**
//		 * Getting convolutional filter bias.
//		 * @return convolutional filter bias.
//		 */
//		NeuronValue getFilterBias();
//
//	}
	
	
	/**
	 * This class represents null parameter.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	class NullParameter implements Parameter {

		/**
		 * Serial version UID for serializable class.
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Default constructor.
		 */
		public NullParameter() {}
		
		@Override
		public Parameter pcopy(Parameter other) {return this;}

		@Override
		public Parameter padd(Parameter other) {return this;}

		@Override
		public Parameter psubtract(Parameter other) {return this;}

		@Override
		public Parameter pmultiply(double factor) {return this;}

		@Override
		public Parameter pinit(Randomizer rnd) {return this;}
		
		@Override
		public Parameter pmultiplyRandom(Randomizer rnd) {return this;}

	}
	
	
	/**
	 * This interface represents randomizer.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	@FunctionalInterface
	interface Randomizer {
		
		/**
		 * Randomizing real number.
		 * @return radom number.
		 */
		double rand();
		
	}
	
	
}
