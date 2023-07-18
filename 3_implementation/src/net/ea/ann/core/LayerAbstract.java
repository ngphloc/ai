/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

/**
 * This class is abstract implementation of layer.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class LayerAbstract implements Layer {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal identifier reference.
	 */
	protected Id idRef = new Id();
	
	
	/**
	 * Identifier.
	 */
	protected int id = -1;
	
	
	/**
	 * Constructor with identifier reference.
	 * @param idRef identifier reference.
	 */
	protected LayerAbstract(Id idRef) {
		if (idRef != null) this.idRef = idRef;
		this.id = this.idRef.get();
	}

	
	/**
	 * Default constructor.
	 */
	protected LayerAbstract() {
		this(null);
	}

	
	@Override
	public Id getIdRef() {
		return idRef;
	}

	
	@Override
	public int id() {
		return id;
	}
	


}
