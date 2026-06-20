/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir;

import java.io.Serializable;
import java.util.List;

import net.ea.ann.ir.Corpus.FeatureCorpus;

/**
 * This interface represents library for indexing features and implementing approximate nearest neighbor algorithms.
 * @author Loc Nguyen
 * @version 1.0
 *
 * @param <T> feature type.
 */
public interface Library<T extends Feature> extends Serializable, Cloneable {

	
	/**
	 * Building library from corpus.
	 * @param <T> record type.
	 * @param corpus corpus.
	 * @param refresh refreshment flag.
	 * @return true if building is successful.
	 */
	boolean build(FeatureCorpus<T> corpus, boolean refresh);
	
	
	/**
	 * Building library from corpus.
	 * @param <T> record type.
	 * @param corpus corpus.
	 * @return true if building is successful.
	 */
	default boolean build(FeatureCorpus<T> corpus) {return build(corpus, false);}


	/**
	 * Obtaining candidates.
	 * @return candidates.
	 */
	List<T> obtainCandidates(T query);


}

