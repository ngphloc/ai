/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir.ml;

import java.util.Random;

import net.ea.ann.core.Id;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.ir.MLAbstract;
import net.ea.ann.ir.Extractor;
import net.ea.ann.mane.Error;
import net.ea.ann.mane.LikelihoodGradient;
import net.ea.ann.mane.MatrixLayer;
import net.ea.ann.mane.MatrixLayerAbstract;
import net.ea.ann.mane.beans.VGG;
import net.ea.ann.raster.Size;

/**
 * This class implements the deep metric learning component based on Proxy-NCA algorithm.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ProxyNCA extends MLAbstract implements Extractor {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Default constructor.
	 */
	public ProxyNCA() {
		
	}
	
	
}



/**
 * This class implements Proxy-NCA algorithm for deep metric learning.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
class ProxyNCACore extends VGG {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Field for proxies count.
	 */
	public final static String PROXIES_COUNT_FIELD = "dir_proxynca_proxies_count";
	
	
	/**
	 * Default value for proxies count.
	 */
	public final static int PROXIES_COUNT_DEFAULT = 10;

	
	/**
	 * Learnable parameter (proxies).
	 */
	protected Matrix proxies = null;
	
	
	/**
	 * Backward information: Accumulation of gradients of proxies.
	 */
	private Matrix dProxiesAccum = null;
	
	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public ProxyNCACore(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
	}

	
	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public ProxyNCACore(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public ProxyNCACore(int neuronChannel, Function activateRef) {
		this(neuronChannel, activateRef, null, null);
	}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public ProxyNCACore(int neuronChannel) {
		this(neuronChannel, null, null, null);
	}


	@Override
	protected boolean initialize(Size inputSize, Size middleSize, Size outputSize) {
		if (!super.initialize(inputSize, middleSize, outputSize)) return false;
		Matrix output = getOutput();
		int rows = paramGetProxiesCount(), columns = output.rows()*output.columns();
		this.proxies = getOutput().create(new Size(columns, rows));
		MatrixUtil.fill(this.proxies, new Random());
		return true;
	}


	/**
	 * Getting proxy at specified index.
	 * @param index specified index.
	 * @return proxy at specified index.
	 */
	public Matrix proxy(int index) {return proxies.getRow(index);}
	
	
	/**
	 * Getting count of proxies.
	 * @return count of proxies.
	 */
	public int proxyCount() {return proxies.rows();}

	
	@Override
	protected Matrix calcOutputError(Matrix output, Matrix realOutput, MatrixLayerAbstract outputLayer, Object... params) {
		//Calculating entropy losses.
		Matrix lossesMatrix = LikelihoodGradient.lossEntropyGradientByColumn(output, realOutput, params);
		NeuronValue[] losses = new NeuronValue[lossesMatrix.rows()];
		for (int row = 0; row < losses.length; row++) {
			losses[row] = lossesMatrix.get(row, 0);
			for (int column = 1; column < lossesMatrix.columns(); column++) losses[row] = losses[row].add(lossesMatrix.get(row, column));
			if (lossesMatrix.columns() > 1) losses[row] = losses[row].divide(lossesMatrix.columns());
		}
		
		Matrix vecOutput = output.vec();
		if (this.proxies.columns() != vecOutput.rows()) throw new IllegalArgumentException();
		Matrix proxiesError = this.proxies.create(new Size(this.proxies.columns(), this.proxies.rows()));
		if (losses.length != proxiesError.rows()) throw new IllegalArgumentException();
		Matrix lastErrorSum = null;
		for (int row = 0; row < this.proxies.rows(); row++) {
			Matrix vecProxy = this.proxies.getRow(row).transpose();
			if (vecProxy.rows() != vecOutput.rows()) throw new IllegalArgumentException();

			//Calculating proxy error.
			Matrix proxyError = calcProxyError(vecOutput, vecProxy).multiply0(losses[row]);
			for (int column = 0; column < proxiesError.columns(); column++)
				proxiesError.set(row, column, proxyError.get(column, 0));
			
			//Calculating lass error.
			Matrix lastError = calcLastError(vecOutput, vecProxy).vecInverse(output.rows());
			lastErrorSum = lastErrorSum != null ? lastErrorSum.add(lastError) : lastError;
		}
		this.dProxiesAccum = this.dProxiesAccum != null ? this.dProxiesAccum.add(proxiesError) : proxiesError;
		
		if (outputLayer == null) return lastErrorSum;
		Matrix input = outputLayer.getInput();
		Matrix derivative = input != null ? input.derivativeWise(outputLayer.getActivateRef()) : null;
		return derivative != null ? derivative.multiplyWise(lastErrorSum) : lastErrorSum;
	}


	/**
	 * Calculating proxy error.
	 * @param vecOutput vectorized output.
	 * @param vecProxy vectorized proxy.
	 * @return proxy error.
	 */
	Matrix calcProxyError(Matrix vecOutput, Matrix vecProxy) {
		return vecOutput.subtract(vecProxy); //Distance-based similarity.
//		return vecOutput; //Product-based similarity.
	}
	
	
	/**
	 * Calculating last error.
	 * @param vecOutput vectorized output.
	 * @param vecProxy vectorized proxy.
	 * @return last error.
	 */
	Matrix calcLastError(Matrix vecOutput, Matrix vecProxy) {
		return vecProxy.subtract(vecOutput);
//		return vecProxy; //Product-based similarity.
	}
	
	
	@Override
	public Error[] backward(Error[] outputErrors, MatrixLayer focus, boolean learning, double learningRate) {
		outputErrors = super.backward(outputErrors, focus, learning, learningRate);
		if (learning) updateParametersFromBackwardInfo(outputErrors.length, learningRate);
		this.dProxiesAccum = null;
		return outputErrors;
	}


	@Override
	public void updateParametersFromBackwardInfo(int recordCount, double learningRate) {
		super.updateParametersFromBackwardInfo(recordCount, learningRate);
		if (this.proxies != null && this.dProxiesAccum != null) {
			Matrix dProxiesMean = this.dProxiesAccum.divide0(recordCount);
			this.proxies = this.proxies.add(dProxiesMean.multiply0(learningRate));
		}
		this.dProxiesAccum = null;
	}


	@Override
	public void resetBackwardInfo() {
		super.resetBackwardInfo();
		this.dProxiesAccum = null;
	}

	
	/**
	 * Getting proxies count.
	 * @return proxies count.
	 */
	public int paramGetProxiesCount() {
		if (config.containsKey(PROXIES_COUNT_FIELD))
			return config.getAsInt(PROXIES_COUNT_FIELD);
		else
			return PROXIES_COUNT_DEFAULT;
	}
	
	
	/**
	 * Setting proxies count.
	 * @param proxiesCount proxies count.
	 * @return this Proxy-NCA.
	 */
	public ProxyNCACore paramSetProxiesCount(int proxiesCount) {
		proxiesCount = proxiesCount < 1 ? PROXIES_COUNT_DEFAULT : proxiesCount;
		config.put(PROXIES_COUNT_FIELD, proxiesCount);
		return this;
	}


}
