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
import java.util.List;

import net.ea.ann.adapter.Delegator;
import net.ea.ann.adapter.Evaluator;
import net.hudup.core.PluginStorage;
import net.hudup.core.RegisterTable;
import net.hudup.core.alg.Alg;
import net.hudup.core.data.Profile;
import net.hudup.core.evaluate.EvaluatorAbstract;
import net.hudup.core.evaluate.NoneWrapperMetricList;
import net.hudup.core.evaluate.SetupTimeMetric;
import net.hudup.core.evaluate.SpeedMetric;
import net.hudup.core.evaluate.execute.ExecuteAsLearnEvaluator;
import net.hudup.core.evaluate.execute.MAE;

/**
 * This class implements the evaluator for convolutional Variational Autoencoders.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class GenModelEvaluator extends ExecuteAsLearnEvaluator implements Evaluator {
	

	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * List of delegated algorithms which can be considered as the second resulted algorithm list wheres
	 * the first resulted algorithm list is stored in {@link EvaluatorAbstract}.
	 */
	protected RegisterTable algDelegators = null;

	
	/**
	 * Default constructor.
	 */
	public GenModelEvaluator() {
		this.algDelegators = new RegisterTable(); 
	}

	
	@Override
	public boolean acceptAlg(Alg alg) throws RemoteException {
		return (alg != null) && (alg instanceof GenModel);
	}


	@Override
	public String getName() throws RemoteException {return "genmodel";}


	@Override
	protected Serializable extractTestValue(Alg alg, Profile testingProfile) {
		if (!(alg instanceof GenModel)) return null;

		return Double.valueOf(0);
	}

	
	@Override
	public Alg getEvaluatedAlg(String algName, boolean remote) throws RemoteException {
		Alg alg = super.getEvaluatedAlg(algName, remote);
		return alg != null ? alg : getAlgCall(this.algDelegators, algName, remote);
	}


	@Override
	public synchronized boolean remoteStartWithoutEvaluation(List<String> algNameList) throws RemoteException {
		if (algNameList == null) return false;
		
		boolean registered = false;
		RegisterTable algReg = PluginStorage.getNormalAlgReg();
		for (String algName : algNameList) {
			Alg alg = algReg.contains(algName) ? algReg.query(algName) : null;
			if (alg == null || !acceptAlg(alg) || !this.algDelegators.canRegister(alg)) continue;
			if (!(alg instanceof Delegator)) continue;
			
			if (this.algDelegators.register(alg)) registered = true;
		}
		return registered;
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
