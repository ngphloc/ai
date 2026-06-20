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

import net.ea.ann.ir.Corpus.RecordCorpus;

/**
 * This interface represents information retrieval system.
 * @author Loc Nguyen
 * @version 1.0
 *
 * @param <U> record type.
 * @param <V> feature type.
 */
public interface IR<U extends Record, V extends Feature> extends Cloneable, Serializable {

	
	/**
	 * Building information retrieval system from corpus.
	 * @param <T> record type.
	 * @param corpus corpus.
	 * @param refresh refreshment flag.
	 * @return true if building is successful.
	 */
	boolean build(RecordCorpus<U> corpus, boolean refresh);
	
	
	/**
	 * Searching for given query feature.
	 * @param query query feature.
	 * @param maxFound the maximum number of found features which can be zero for all.
	 * @return list of found features.
	 */
	List<ScoredFeature<V>> search(V query, int maxFound);

	
}
