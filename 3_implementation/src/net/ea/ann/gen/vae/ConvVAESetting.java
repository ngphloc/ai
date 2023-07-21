/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.gen.vae;

import java.io.Serializable;

import net.ea.ann.core.NetworkConfig;

/**
 * This class represent setting or parameters of convolutional Variational Autoencoders.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ConvVAESetting implements Serializable, Cloneable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Name of width field.
	 */
	public final static String WIDTH_FIELD = "convvae_width";
	
	
	/**
	 * Width default.
	 */
	public final static int WIDTH__DEFAULT = 1;

	
	/**
	 * Name of height field.
	 */
	public final static String HEIGHT_FIELD = "convvae_height";

	
	/**
	 * Height default.
	 */
	public final static int HEIGHT__DEFAULT = 1;

	
	/**
	 * Width of the convolutional Variational Autoencoders.
	 */
	public int width = 0;
	
	
	/**
	 * Height of the convolutional Variational Autoencoders.
	 */
	public int height = 0;
	
	
	/**
	 * Default construction.
	 */
	public ConvVAESetting() {

	}

	
	/**
	 * Extracting from configuration.
	 * @param config network configuration.
	 */
	public void extractConfig(NetworkConfig config) {
		if (config == null) return;
		
		if (config.containsKey(WIDTH_FIELD)) width = config.getAsInt(WIDTH_FIELD);
		if (config.containsKey(HEIGHT_FIELD)) height = config.getAsInt(HEIGHT_FIELD);
	}
	
	
	
}
