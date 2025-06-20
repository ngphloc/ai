/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.nio.file.Path;

import net.ea.ann.core.value.NeuronValue;

/**
 * This class is a serializable wrapper of sound record.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class SoundWrapper implements Sound {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Internal sound record.
	 */
	protected Sound sound = null;
	
	
	/**
	 * Constructor with sound.
	 * @param sound sound record.
	 */
	public SoundWrapper(Sound sound) {
		this.sound = sound;
	}

	
	@Override
	public int getLength() {
		return sound.getLength();
	}

	
	@Override
	public boolean save(Path path) {
		return sound.save(path);
	}

	
	@Override
	public NeuronValue[] convertFromSoundToNeuronValues(int neuronChannel, int length, boolean isNorm) {
		return sound.convertFromSoundToNeuronValues(neuronChannel, length, isNorm);
	}

	
}
