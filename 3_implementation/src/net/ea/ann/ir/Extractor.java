/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir;

import java.io.Serializable;

/**
 * This interface represents feature extraction component.
 * @author Loc Nguyen
 * @version 1.0
 *
 * @param <U> record type.
 * @param <V> feature type.
 */
public interface Extractor<U extends Record, V extends Feature> extends Serializable, Cloneable {

	
	/**
	 * Feature of record.
	 * @param record record.
	 * @return feature of record.
	 */
	V featureOf(U record);
	
	
}
