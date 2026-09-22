/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.weight;

import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Size;

/**
 * This class implement extensive group norm weight without storing norm information, developed by Yuxin Wu and Kaiming He.
 * @author Yuxin Wu, Kaiming He, developed by Loc Nguyen
 * @version 1.0
 *
 */
public class NormWeightGroup2 extends NormWeightGroup {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Constructor with the kernel.
	 * @param kernel the kernel.
	 */
	public NormWeightGroup2(WKernel kernel) {
		super(kernel);
	}


	@Override
	public Matrix retrieveDefaultNorm() {return null;}


	@Override
	MeanStd[] getMeanStds() {return null;}

	
	@Override
	public Object clone() throws CloneNotSupportedException {
		WKernel clonedKernel = (WKernel)this.kernel.clone();
		NormWeightGroup2 cloned = new NormWeightGroup2(clonedKernel);
		cloned.layer = this.layer;
		return cloned;
	}


	/**
	 * Creating extensive norm weight.
	 * @param prevSize previous size.
	 * @param size current size.
	 * @param hint hint value.
	 * @return norm weight.
	 */
	public static NormWeightGroup2 create(Size prevSize, Size size, NeuronValue hint) {
		if (prevSize.width != size.width || prevSize.height != size.height || prevSize.depth != size.depth) throw new IllegalArgumentException();
		Matrix W = MatrixUtil.create(new Size(1, 1, size.depth, 1), hint.unit());
		Matrix bias = MatrixUtil.create(new Size(1, 1, size.depth, 1), hint.zero());
		WKernel kernel = new WKernel(W instanceof MatrixStack ? (MatrixStack)W : new MatrixStack(W),
			bias instanceof MatrixStack ? (MatrixStack)bias : new MatrixStack(bias));
		return new NormWeightGroup2(kernel);
	}


}
