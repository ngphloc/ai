/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.train;

import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.raster.Size;

/**
 * This class implements Adam optimizer for descent gradient.
 * Adam optimizer is the biggest practical gain in updating neural network weights in training.
 * @author Gemini, Loc Nguyen
 * @version 1.0
 *
 */
public class AdamOptimizer implements Optimizer {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Exponential decay rate for the 1st moment.
	 */
	public final static double BETA1 = 0.9;
	
	
	/**
	 * Exponential decay rate for the 2nd moment.
	 */
	public final static double BETA2 = 0.999;

	
	/**
	 * Minimum value.
	 */
	final static double EPSILON = Float.MIN_VALUE;
	
	
	/**
	 * Exponential decay rate for the 1st moment.
	 */
	protected double beta1 = BETA1;
	
	
	/**
	 * Exponential decay rate for the 2nd moment.
	 */
	protected double beta2 = BETA2;
	
	
	/**
	 * Time step counter.
	 */
	int t = 0;
	
	
	/**
	 * Constructor with exponential decay rates.
	 * @param beta1 exponential decay rate for the 1st moment.
	 * @param beta2 exponential decay rate for the 2nd moment.
	 */
	public AdamOptimizer(double beta1, double beta2) {
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.t = 0;
	}
	
	
	/**
	 * Default constructor.
	 */
	public AdamOptimizer() {this(BETA1, BETA2);}

	
	@Override
	public void reset() {
		this.t = 0;
	}
	
	
	/**
	 * Increasing time.
	 * @return increased time.
	 */
	public int incTime() {return ++t;}
	
	
	/**
	 * Getting time.
	 * @return time.
	 */
	public int time() {return t;}
	
	
	/**
	 * Re-calculating gradient.
	 * @param grad gradient.
	 * @param time time.
	 * @return recalculated gradient.
	 */
	public Matrix recalcGradient(Matrix grad, int time) {
		assert (time > 0);
		if (grad instanceof MatrixStack) return recalcGradient((MatrixStack)grad, time);
		
		//Initializing the first moment and second moment.
		Matrix m = grad.create(new Size(grad.columns(), grad.rows()));
		Matrix v = grad.create(new Size(grad.columns(), grad.rows()));
		
		//Computing the first moment.
		Matrix m1 = m.multiply0(this.beta1);
		Matrix m2 = grad.multiply0(1.0 - this.beta1);
		m = m1.add(m2);
		
		//Computing the second moment.
		Matrix v1 = v.multiply0(this.beta2);
		Matrix grad2 = grad.multiplyWise(grad);
		Matrix v2 = grad2.multiply0(1.0 - this.beta2);
		v = v1.add(v2);
		
		//Computing bias-corrected version of the first moment.
		double bias1 = 1.0 - Math.pow(this.beta1, time);
        Matrix mHat = m.multiply0(1.0 / bias1);
		
		//Computing bias-corrected version of the second moment.
        double bias2 = 1.0 - Math.pow(this.beta2, time);
        Matrix vHat = v.multiply0(1.0 / bias2);
        
        //Computing the Adam update gradient.
		Matrix adam = grad.create(new Size(grad.columns(), grad.rows()));
		NeuronValue epsilon = adam.get(0, 0).valueOf(EPSILON);
		for (int row = 0; row < adam.rows(); row++) {
			for (int column = 0; column < adam.columns(); column++) {
				NeuronValue mnum = mHat.get(row, column);
				NeuronValue vdenom = vHat.get(row, column).sqrt().add(epsilon);
				adam.set(row, column, mnum.divide(vdenom));
			}
		}
        
		return grad;
	}
	
	
	/**
	 * Re-calculating gradient.
	 * @param grad gradient.
	 * @param time time.
	 * @return recalculated gradient.
	 */
	public Matrix recalcGradient(MatrixStack grad, int time) {
		Matrix[] grads = MatrixUtil.split(grad);
		Matrix[] adams = new Matrix[grads.length];
		for (int d = 0; d < adams.length; d++) {
			adams[d] = recalcGradient(grads[d], time);
		}
		return MatrixUtil.join(adams);
	}

	
	/**
	 * Re-calculating gradient.
	 * @param grad gradient.
	 * @param time time.
	 * @return recalculated gradient.
	 */
	public NeuronValue recalcGradient(NeuronValue grad, int time) {
		Matrix gradM = MatrixUtil.create(Size.unit(), grad);
		return recalcGradient(gradM, time).get(0, 0);
	}
	
	
	/**
	 * Re-calculating gradient.
	 * @param grad gradient.
	 * @return recalculated gradient.
	 */
	public Matrix recalcGradient(Matrix grad) {return recalcGradient(grad, incTime());}

	
	/**
	 * Re-calculating gradient.
	 * @param grad gradient.
	 * @return recalculated gradient.
	 */
	public Matrix recalcGradient(MatrixStack grad) {return recalcGradient(grad, incTime());}


	/**
	 * Re-calculating gradient.
	 * @param grad gradient.
	 * @param time time.
	 * @return recalculated gradient.
	 */
	public NeuronValue recalcGradient(NeuronValue grad) {return recalcGradient(grad, incTime());}
	

}
