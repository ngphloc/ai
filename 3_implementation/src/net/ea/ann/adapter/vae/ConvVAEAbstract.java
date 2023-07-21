/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.vae;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.rmi.RemoteException;
import java.util.Collection;

import net.ea.ann.adapter.Util;
import net.ea.ann.conv.Raster;
import net.ea.ann.core.NetworkDoEvent;
import net.ea.ann.core.NetworkInfoEvent;
import net.ea.ann.core.NetworkListener;
import net.ea.ann.gen.vae.ConvVAEUtil;
import net.hudup.core.alg.DuplicatableAlg;
import net.hudup.core.alg.ExecuteAsLearnAlgAbstract;
import net.hudup.core.alg.SetupAlgEvent;
import net.hudup.core.alg.SetupAlgEvent.Type;
import net.hudup.core.data.DataConfig;
import net.hudup.core.data.Dataset;
import net.hudup.core.data.Profile;
import net.hudup.core.logistic.Inspector;
import net.hudup.core.logistic.ui.DescriptionDlg;
import net.hudup.core.logistic.ui.UIUtil;

/**
 * This class implements partially convolutional Variational Autoencoders.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public abstract class ConvVAEAbstract extends ExecuteAsLearnAlgAbstract implements ConvVAE, ConvVAERemote, NetworkListener, /*AllowNullTrainingSet,*/ DuplicatableAlg {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Name of Z dimension field.
	 */
	public final static String ZDIM_FIELD = "convvae_zdim";
	
	
	/**
	 * Default value of Z dimension field.
	 */
	public final static int ZDIM_DEFAULT = 10;

	
	/**
	 * Name of zoom-out field.
	 */
	public final static String ZOOMOUT_FIELD = "convvae_zoomout";

	
	/**
	 * Default value of zoom-out field.
	 */
	public final static int ZOOMOUT_DEFAULT = 3;

	
	/**
	 * Default value of zoom-out field.
	 */
	public final static int GENS_DEFAULT = 10;

	
	/**
	 * Internal convolutional Variational Autoencoders.
	 */
	protected net.ea.ann.gen.vae.ConvVAEImpl vae = null;
	
	
	/**
	 * Default constructor.
	 */
	public ConvVAEAbstract() {
		vae = createConvVAE();
		
		try {
			config.putAll(Util.toConfig(vae.getConfig()));
		} catch (Throwable e) {Util.trace(e);}
		
		try {
			vae.addListener(this);
		} catch (Throwable e) {Util.trace(e);}
	}

	
	@Override
	protected Object fetchSample(Dataset dataset) {
		return dataset != null ? dataset.fetchSample2() : null;
	}


	@SuppressWarnings("unchecked")
	@Override
	public Object executeAsLearn(Object input) throws RemoteException {
		if (input != null) return null; //Only running in setup method.
		vae.getConfig().putAll(Util.transferToANNConfig(config));
		
		int count = 0;
		for (Profile profile : (Collection<Profile>)sample) {
			if (profile == null || profile.getAttCount() < 2) continue;
			
			String sourceText = profile.getValueAsString(0);
			if (sourceText == null) return null;
			Path sourceDirectory = Paths.get(sourceText);
			String targetText = profile.getValueAsString(1);
			if (targetText == null) return null;
			Path targetDirectory = Paths.get(targetText);

			int nGens = GENS_DEFAULT;
			if (profile.getAttCount() > 2) nGens = profile.getValueAsInt(2);
			nGens = nGens <= 0 ? GENS_DEFAULT : nGens;
			
			count += ConvVAEUtil.create(vae).generateRasters(sourceDirectory, targetDirectory, nGens, getZDim(), getZoomOut());
		}
		
		return Double.valueOf(count);
	}


	/**
	 * Create convolutional VAE instance.
	 * @return convolutional VAE instance.
	 */
	protected abstract net.ea.ann.gen.vae.ConvVAEImpl createConvVAE();
	
	
	/**
	 * Getting Z dimension.
	 * @return Z dimension.
	 */
	protected int getZDim() {
		int zDim = ZDIM_DEFAULT;
		if (config.containsKey(ZDIM_FIELD)) zDim = config.getAsInt(ZDIM_FIELD);
		
		zDim = zDim <= 0 ? ZDIM_DEFAULT : zDim;
		return zDim;
	}
	
	
	/**
	 * Getting zooming out ratio.
	 * @return zooming out ratio.
	 */
	protected int getZoomOut() {
		int zoomOut = ZOOMOUT_DEFAULT;
		if (config.containsKey(ZOOMOUT_FIELD)) zoomOut = config.getAsInt(ZOOMOUT_FIELD);
		
		zoomOut = zoomOut <= 0 ? ZOOMOUT_DEFAULT : zoomOut;
		return zoomOut;
	}

	
	@Override
	public Object getParameter() throws RemoteException {
		return vae;
	}

	
	@Override
	public String parameterToShownText(Object parameter, Object... info) throws RemoteException {
		if (parameter == null)
			return "";
		else if (!(parameter instanceof net.ea.ann.gen.vae.ConvVAEImpl))
			return "";
		else
			return ((net.ea.ann.gen.vae.ConvVAEImpl)parameter).toString();
	}

	
	@Override
	public synchronized String getDescription() throws RemoteException {
		return parameterToShownText(getParameter());
	}

	
	@Override
	public Inspector getInspector() {
		String desc = "";
		try {
			desc = getDescription();
		} catch (Exception e) {Util.trace(e);}
		
		return new DescriptionDlg(UIUtil.getDialogForComponent(null), "Inspector", desc);
	}

	
	@Override
	public String[] getBaseRemoteInterfaceNames() throws RemoteException {
		return new String[] {ConvVAERemote.class.getName()};
	}

	
	@Override
	public void receivedInfo(NetworkInfoEvent evt) throws RemoteException {

	}

	
	@Override
	public void receivedDo(NetworkDoEvent evt) throws RemoteException {
		if (evt.getType() == NetworkDoEvent.Type.doing) {
			fireSetupEvent(new SetupAlgEvent(this, Type.doing, getName(), null,
				evt.getLearnResult(),
				evt.getProgressStep(), evt.getProgressTotalEstimated()));
		}
		else if (evt.getType() == NetworkDoEvent.Type.done) {
			fireSetupEvent(new SetupAlgEvent(this, Type.done, getName(), null,
					evt.getLearnResult(),
					evt.getProgressStep(), evt.getProgressTotalEstimated()));
		}
	}


	@Override
	public void setName(String name) {
		getConfig().put(DUPLICATED_ALG_NAME_FIELD, name);
	}


	@Override
	public DataConfig createDefaultConfig() {
		DataConfig config = super.createDefaultConfig();
		config.put(ZDIM_FIELD, ZDIM_DEFAULT);
		config.put(ZOOMOUT_FIELD, ZOOMOUT_DEFAULT);
		
		config.addReadOnly(Raster.SOURCE_IMAGE_TYPE_FIELD);
		
		return config;
	}

	
}
