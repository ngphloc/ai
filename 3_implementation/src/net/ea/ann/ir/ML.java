/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir;

import java.io.Serializable;

import net.ea.ann.ir.Corpus.RecordCorpus;

/**
 * This interface represents (deep) metric learning algorithm.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
/**
 * This interface represents (deep) metric learning algorithm.
 * @author Loc Nguyen
 * @version 1.0
 *
 * @param <T> record type.
 */
public interface ML<T extends Record> extends Serializable, Cloneable {

	
	/**
	 * Building metric learning component from corpus.
	 * @param <T> record type.
	 * @param corpus corpus.
	 * @param refresh refreshment flag.
	 * @return true if building is successful.
	 */
	boolean build(RecordCorpus<T> corpus, boolean refresh);
	
	
	/**
	 * Building metric learning component from corpus.
	 * @param <T> record type.
	 * @param corpus corpus.
	 * @return true if building is successful.
	 */
	default boolean build(RecordCorpus<T> corpus) {return build(corpus, false);}
	
	
}
