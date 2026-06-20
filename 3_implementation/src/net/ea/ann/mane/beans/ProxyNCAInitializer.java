/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.beans;

import java.io.Serializable;

/**
 * This class provides initialization methods for Proxy-NCA (Proxy-Neighborhood Component Analysis) model.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ProxyNCAInitializer implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Proxy-NCA model.
	 */
	protected ProxyNCA proxyNCA = null;
	
	
	/**
	 * Constructor with Proxy-NCA model.
	 * @param proxyNCA Proxy-NCA model.
	 */
	public ProxyNCAInitializer(ProxyNCA proxyNCA) {
		this.proxyNCA = proxyNCA;
	}
	

}
