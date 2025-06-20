/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value.indexed;

import net.ea.ann.core.value.WeightValue;

/**
 * This interface represents an indexed weight value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface IndexedWeightValue extends WeightValue {


	/**
	 * Getting current indexed value.
	 * @return current indexed value.
	 */
	WeightValue v();


	/**
	 * Getting index.
	 * @return index.
	 */
	int getIndex();
	
	
	/**
	 * Setting index
	 * @param index specified index.
	 */
	void setIndex(int index);
	
	
	/**
	 * Getting size.
	 * @return size.
	 */
	int size();
	
	
	/**
	 * Get value at specified index.
	 * @param index specified index.
	 * @return value at specified index.
	 */
	WeightValue get(int index);
	
	
	/**
	 * Setting value at specified index.
	 * @param index specified index.
	 * @param value specified value.
	 * @return previous value.
	 */
	WeightValue set(int index, WeightValue value);


}
