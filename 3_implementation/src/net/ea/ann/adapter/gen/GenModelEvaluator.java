/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.gen;

import java.io.Serializable;
import java.rmi.RemoteException;

import net.hudup.core.alg.Alg;
import net.hudup.core.data.Profile;
import net.hudup.core.evaluate.NoneWrapperMetricList;
import net.hudup.core.evaluate.SetupTimeMetric;
import net.hudup.core.evaluate.SpeedMetric;
import net.hudup.core.evaluate.execute.ExecuteAsLearnEvaluator;
import net.hudup.core.evaluate.execute.MAE;

/**
 * This class is the evaluator for convolutional Variational Autoencoders.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class GenModelEvaluator extends ExecuteAsLearnEvaluator {
	

	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Default constructor.
	 */
	public GenModelEvaluator() {

	}

	
	@Override
	public boolean acceptAlg(Alg alg) throws RemoteException {
		return (alg != null) && (alg instanceof GenModel);
	}


	@Override
	public String getName() throws RemoteException {
		return "genmodel";
	}


	@Override
	protected Serializable extractTestValue(Alg alg, Profile testingProfile) {
		if (!(alg instanceof GenModel)) return null;

		return Double.valueOf(0);
	}

	
	@Override
	public NoneWrapperMetricList defaultMetrics() throws RemoteException {
		NoneWrapperMetricList metricList = new NoneWrapperMetricList();
		
		SetupTimeMetric setupTime = new SetupTimeMetric();
		metricList.add(setupTime);
		
		SpeedMetric speed = new SpeedMetric();
		metricList.add(speed);
		
		MAE mae = new MAE();
		metricList.add(mae);
		
		return metricList;
	}


}
