/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.raster;

import java.nio.file.Path;

import net.ea.ann.conv.ConvLayerSingle;
import net.ea.ann.core.value.NeuronValue;

/**
 * This class is an default implementation of raster in 1D space which is often sound record.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class Raster1DImpl extends RasterAbstract implements Raster1D {


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
	 * @param sound specified sound.
	 */
	protected Raster1DImpl(Sound sound) {
		super();
		this.sound = sound;
	}


	@Override
	public int getWidth() {
		return sound.getLength();
	}


	@Override
	public java.awt.Image getRepImage() {
		throw new RuntimeException("Raster1DImpl.getRepImage() not implemented yet.");
	}


	@Override
	public String getDefaultFormat() {
		return Sound.getDefaultFormat();
	}


	@Override
	public boolean save(Path path) {
		return sound.save(path);
	}


	@Override
	public NeuronValue[] toNeuronValues(ConvLayerSingle layer, boolean isNorm) {
		if (layer == null) return null;
		return this.sound.convertFromSoundToNeuronValues(layer.getNeuronChannel(), layer.getWidth(), isNorm);
	}


	@Override
	public NeuronValue[] toNeuronValues(int neuronChannel, Size size,
			boolean isNorm) {
		return sound.convertFromSoundToNeuronValues(neuronChannel, size.width, isNorm);
	}


	/**
	 * Creating raster from specified sound.
	 * @param sound specified sound.
	 * @return raster created from specified sound.
	 */
	public static Raster1DImpl create(Sound sound) {
		if (sound == null)
			return null;
		else
			return new Raster1DImpl(sound);
	}


	/**
	 * Create raster from neuron values.
	 * @param layer specified layer.
	 * @param values neuron values.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @return raster created from neuron values.
	 */
	public static Raster1DImpl create(ConvLayerSingle layer, NeuronValue[] values,
			boolean isNorm) {
		if (layer == null) return null;
		throw new RuntimeException("Raster1DImpl.create(ConvLayerSingle, NeuronValue[], boolean) not implemented yet.");
	}
	
	
	/**
	 * Create raster from neuron values.
	 * @param values neuron values.
	 * @param neuronChannel neuron channel.
	 * @param size raster size.
	 * @param isNorm flag to indicate whether pixel is normalized in range [0, 1].
	 * @return raster created from neuron values.
	 */
	public static Raster1DImpl create(NeuronValue[] values, int neuronChannel, Size size,
			boolean isNorm) {
		throw new RuntimeException("Raster1DImpl.create(NeuronValue[], Size, boolean) not implemented yet.");
	}


	/**
	 * Loading raster from path.
	 * @param path specific path.
	 * @return raster loaded from path.
	 */
	public static Raster2DImpl load(Path path) {
		throw new RuntimeException("Raster1DImpl.load(Path) not implemented yet.");
	}


}
