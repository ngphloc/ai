/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.vae.beans;

import net.ea.ann.adapter.vae.ConvVAEAbstract;

/**
 * This class is the bean implementation of convolutional Variational Autoencoders for Hudup framework.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ConvVAEImpl extends ConvVAEAbstract {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Default depth or neuron channel.
	 */
	public static final int DEFAULT_DEPTH = 3;
	
	
	/**
	 * Default constructor.
	 */
	public ConvVAEImpl() {

	}

	
	@Override
	public String getName() {
		return "convvae";
	}

	
	@Override
	protected net.ea.ann.gen.vae.ConvVAEImpl createConvVAE() {
		return net.ea.ann.gen.vae.ConvVAEImpl.create(DEFAULT_DEPTH);
	}

	
}
