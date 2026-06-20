/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir;

import java.io.Serializable;
import java.rmi.RemoteException;
import java.util.List;

import net.ea.ann.core.Util;
import net.hudup.core.logistic.AbstractRunner;

/**
 * This class implements partially search engine.
 * @author Loc Nguyen
 * @version 1.0
 *
 * @param <U> record type.
 * @param <V> feature type.
 */
public abstract class SearchEngineAbstract<U extends Record, V extends Feature> implements SearchEngine<U>, Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal information retrieval system.
	 */
	protected volatile IRAbstract<U, V> ir = null;
	
	
	/**
	 * Shadow information retrieval system.
	 */
	protected IRAbstract<U, V> irShadow = null;

	
	/**
	 * Internal crawler.
	 */
	protected Crawler<U> crawler = null;
	
	
	/**
	 * Runner.
	 */
	protected AbstractRunner runner = null;
	
	
	/**
	 * Default constructor.
	 */
	public SearchEngineAbstract() {
		this.ir = createIR();
		this.irShadow = createIR();
		this.crawler = createCrawler();
		this.runner = new AbstractRunner() {
			@Override
			protected void task() {
				taskBackground();
			}
			
			@Override
			protected void clear() {
				clearBackground();
			}
		};
	}


	@Override
	public List<U> search(U query) throws RemoteException {
		V featureQuery = ir.extractor().featureOf(query);
		if (featureQuery == null) return Util.newList(0);
		List<ScoredFeature<V>> scoredFeatures = ir.search(featureQuery);
		List<U> results = Util.newList(0);
		for (ScoredFeature<V> scoredFeature : scoredFeatures) {
			@SuppressWarnings("unchecked")
			U record = (U)scoredFeature.getRecord();
			if (record != null) results.add(record);
		}
		return results;
	}


	/**
	 * Creating information retrieval system.
	 * @return information retrieval system.
	 */
	protected abstract IRAbstract<U, V> createIR();
	
	
	/**
	 * Creating crawler.
	 * @return crawler.
	 */
	protected abstract Crawler<U> createCrawler();


	@Override
	public boolean start() throws RemoteException {
		return runner.start();
	}


	@Override
	public boolean stop() throws RemoteException {
		return runner.stop();
	}
	

	/**
	 * The actual tasks (works) which are performed when runner is running.
	 */
	protected abstract void taskBackground();
	
	
	/**
	 * Clearing all resources after runner run (stopped).
	 */
	protected abstract void clearBackground();


}
