/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.filter;

import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Size;

/**
 * This class represents Global Average Pooling (GAP) filter.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class GAP extends PoolFilter {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public GAP() {
		super(Size.unit());
	}

	
	@Override
	int depth() {return 1;}
	

	@Override
	void forward(Matrix prevLayer, Matrix thisInputLayer, Matrix thisOutputLayer) {
		throw new IllegalArgumentException();
	}


	@Override
	public void forward(Matrix prevLayer, Matrix thisInputLayer, Matrix thisOutputLayer, NeuronValue bias, Function thisActivateRef) {
		if (thisInputLayer instanceof MatrixStack || thisOutputLayer instanceof MatrixStack) throw new IllegalArgumentException();
		if (thisInputLayer.columns() != 1 || thisOutputLayer.columns() != 1) throw new IllegalArgumentException();
		MatrixStack prevLayers = prevLayer instanceof MatrixStack ? (MatrixStack)prevLayer : new MatrixStack(prevLayer);
		int depth = prevLayers.depth();
		if (thisInputLayer.rows() != depth || thisOutputLayer.rows() != depth) throw new IllegalArgumentException();
		
		for (int d = 0; d < depth; d++) {
			NeuronValue mean = MatrixUtil.valueMean(prevLayers.get(d));
			thisInputLayer.set(d, 0, mean);
			thisOutputLayer.set(d, 0, mean);
		}
	}


	@Override
	Matrix dValue(int thisX, int thisY, Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer) {
		throw new IllegalArgumentException();
	}


	@Override
	public Matrix dValue(Matrix prevInputLayer, Matrix prevOutputLayer, Matrix thisErrorLayer, Function thisActivateRef) {
		if (prevOutputLayer instanceof MatrixStack || thisErrorLayer instanceof MatrixStack) throw new IllegalArgumentException();
		if (prevOutputLayer.columns() != 1 || thisErrorLayer.columns() != 1) throw new IllegalArgumentException();
		MatrixStack prevInputLayers = prevInputLayer instanceof MatrixStack ? (MatrixStack)prevInputLayer : new MatrixStack(prevInputLayer);
		int depth = prevInputLayers.depth();
		if (prevOutputLayer.rows() != depth || thisErrorLayer.rows() != depth) throw new IllegalArgumentException();

		Matrix[] dPrevValues = new Matrix[depth];
		for (int d = 0; d < depth; d++) {
			int rows = prevInputLayers.rows(), columns = prevInputLayers.columns();
			dPrevValues[d] = prevInputLayers.get().create(new Size(columns, rows));
			
			int size = rows*columns;
			NeuronValue error = thisErrorLayer.get(d, 0).divide(size);
			MatrixUtil.fill(dPrevValues[d], error);
		}
		
		return dPrevValues.length == 1 ? dPrevValues[0] : new MatrixStack(dPrevValues);
	}


}
