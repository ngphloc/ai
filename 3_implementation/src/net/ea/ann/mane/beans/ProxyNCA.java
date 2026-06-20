/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane.beans;

import java.io.Serializable;
import java.util.List;
import java.util.Map;
import java.util.Random;

import net.ea.ann.core.Id;
import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.value.Accumulator;
import net.ea.ann.core.value.Matrix;
import net.ea.ann.core.value.MatrixStack;
import net.ea.ann.core.value.MatrixUtil;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.Sim;
import net.ea.ann.mane.Error;
import net.ea.ann.mane.LikelihoodGradient;
import net.ea.ann.mane.MatrixLayer;
import net.ea.ann.mane.MatrixLayerAbstract;
import net.ea.ann.raster.Augmentor;
import net.ea.ann.raster.Raster;
import net.ea.ann.raster.Size;

/**
 * This class implements Proxy-NCA (Proxy-Neighborhood Component Analysis) algorithm for deep metric learning.
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ProxyNCA extends VGGClassifier {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Field for proxy flag.
	 */
	public final static String IS_PROXY_FIELD = "pnca_is_proxy";
	
	
	/**
	 * Default value for proxy flag field.
	 */
	public final static boolean IS_PROXY_DEFAULT = true;

	
	/**
	 * Field for augmentation.
	 */
	public final static String AUGMENTED_FIELD = "pnca_augmented";
	
	
	/**
	 * Default value for augmentation field.
	 */
	public final static boolean AUGMENTED_DEFAULT = true;

	
	/**
	 * Field for piece size.
	 */
	public final static String PIECE_SIZE_FIELD = "pnca_piece_size";
	
	
	/**
	 * Default value for piece size.
	 */
	public final static int PIECE_SIZE_DEFAULT = 2;

	
	/**
	 * This class consists of one anchor feature, one or many positive features, one or many negative features.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	class Triplet implements Serializable, Cloneable {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Anchor feature.
		 */
		protected Matrix anchor = null;
		
		/**
		 * Positive features.
		 */
		protected List<Matrix> positives = Util.newList(0);
		
		/**
		 * Negative features.
		 */
		protected List<Matrix> negatives = Util.newList(0);
		
		/**
		 * Previous features.
		 */
		private Map<Matrix, Matrix> prevs = Util.newMap(0);
		
		/**
		 * Input raster of anchor.
		 */
		protected Raster anchorInputRaster = null;
		
		/**
		 * Default construct.
		 */
		private Triplet() {}
		
		/**
		 * Constructor with anchor feature, positive feature, and negative features.
		 * @param anchor anchor feature.
		 * @param positive positive feature.
		 * @param negatives negative features.
		 */
		public Triplet(Matrix anchor, Matrix positive, Matrix...negatives) {
			this();
			if (anchor != null) this.anchor = anchor;
			if (positive != null) this.positives.add(positive);
			if (negatives != null) {
				for (Matrix negative : negatives) {
					if (negative != null) this.negatives.add(negative);
				}
			}
			
			if (!validate()) throw new IllegalArgumentException();
		}
		
		/**
		 * Validating the triplet.
		 * @return true if the triplet is valid.
		 */
		public boolean validate() {return anchor != null;}
		
		/**
		 * Getting previous anchor feature.
		 * @return previous anchor feature.
		 */
		public Matrix anchorPrev() {
			return anchor != null && prevs.containsKey(anchor) ? prevs.get(anchor) : null;
		}
		
		/**
		 * Adding previous anchor feature.
		 * @param anchorPrev previous anchor feature.
		 */
		public void anchorPrevAdd(Matrix anchorPrev) {
			if (anchor != null && anchorPrev != null) prevs.put(anchor, anchorPrev);
		}
		
		/**
		 * Getting number of positive features.
		 * @return number of positive features.
		 */
		public int positiveSize() {return positives.size();}
		
		/**
		 * Getting positive feature.
		 * @param index positive index.
		 * @return positive feature at specified index.
		 */
		public Matrix positive(int index) {return positives.get(index);}
		
		/**
		 * Getting previous positive feature.
		 * @param index positive index.
		 * @return previous positive feature.
		 */
		public Matrix positivePrev(int index) {
			Matrix positive = positive(index);
			return positive != null && prevs.containsKey(positive) ? prevs.get(positive) : null;
		}
		
		/**
		 * Adding previous positive feature.
		 * @param index positive index.
		 * @param positivePrev previous positive feature.
		 */
		public void positivePrevAdd(int index, Matrix positivePrev) {
			Matrix positive = positives.get(index);
			if (positive != null && positivePrev != null) prevs.put(positive, positivePrev);
		}
		
		/**
		 * Adding previous positive feature.
		 * @param positive positive feature.
		 * @param positivePrev previous positive feature.
		 */
		public void positivePrevAdd(Matrix positive, Matrix positivePrev) {
			if (positive == null || positivePrev == null) return;
			int index = positiveIndexOf(positive);
			if (index >= 0) positivePrevAdd(index, positivePrev);
		}
		
		/**
		 * Getting index of positive feature.
		 * @param positive positive feature.
		 * @return index of positive feature.
		 */
		public int positiveIndexOf(Matrix positive) {return this.positives.indexOf(positive);}
		
		/**
		 * Getting index of negative feature.
		 * @param negative negative feature.
		 * @return index of negative feature.
		 */
		public int negativeIndexOf(Matrix negative) {return this.negatives.indexOf(negative);}
		
		/**
		 * Calculating similarity between positive feature and anchor feature.
		 * @param positiveIndex positive index.
		 * @return similarity between positive feature and anchor feature.
		 */
		public NeuronValue positiveSim(int positiveIndex) {return sim().sim(positives.get(positiveIndex), anchor);}
		
		/**
		 * Calculating similarity between negative feature and anchor feature.
		 * @param negativeIndex negative index.
		 * @return similarity between negative feature and anchor feature.
		 */
		public NeuronValue negativeSim(int negativeIndex) {return sim().sim(negatives.get(negativeIndex), anchor);}

	}
	
	
	/**
	 * This class consists of one anchor feature, one or many positive features, zero or many negative features, one or many weight.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	class Quartet implements Serializable, Cloneable {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;
		
		/**
		 * Internal triplet.
		 */
		protected Triplet triplet = null;
		
		/**
		 * Proxies.
		 */
		protected List<Matrix> proxies = Util.newList(0);
		
		/**
		 * Default constructor.
		 */
		private Quartet() {}
		
		/**
		 * Constructor with triplet and proxy group.
		 * @param triplet triplet.
		 * @param proxyGroup proxy group.
		 */
		public Quartet(Triplet triplet, Matrix proxyGroup) {
			this();
			this.triplet = triplet;
			if (proxyGroup != null) {
				for (int row = 0; row < proxyGroup.rows(); row++) this.proxies.add(extractProxy(proxyGroup, row));
			}
			if (!validate()) throw new IllegalArgumentException();
		}
		
		/**
		 * Constructor with anchor feature, positive feature, and proxy group.
		 * @param anchor anchor feature.
		 * @param positive positive feature.
		 * @param proxyGroup proxy group. 
		 */
		public Quartet(Matrix anchor, Matrix positive, Matrix proxyGroup) {this(new Triplet(anchor, positive), proxyGroup);}

		/**
		 * Constructor with anchor feature, positive feature, negative feature, and proxy group.
		 * @param anchor anchor feature.
		 * @param positive positive feature.
		 * @param negative negative feature.
		 * @param proxyGroup proxy group. 
		 */
		public Quartet(Matrix anchor, Matrix positive, Matrix negative, Matrix proxyGroup) {this(new Triplet(anchor, positive, negative), proxyGroup);}
		
		/**
		 * Getting anchor feature.
		 * @return anchor feature.
		 */
		public Matrix anchor() {return triplet.anchor;}
		
		/**
		 * Getting previous anchor feature.
		 * @return previous anchor feature.
		 */
		public Matrix anchorPrev() {return triplet.anchorPrev();}
		
		/**
		 * Adding previous anchor feature.
		 * @param anchorPrev previous anchor feature.
		 */
		public void anchorPrevAdd(Matrix anchorPrev) {triplet.anchorPrevAdd(anchorPrev);}
		
		/**
		 * Getting number of positive features.
		 * @return number of positive features.
		 */
		public int positiveSize() {return triplet.positiveSize();}

		/**
		 * Getting positive feature.
		 * @param index positive index.
		 * @return positive feature at specified index.
		 */
		public Matrix positive(int index) {return triplet.positive(index);}
		
		/**
		 * Getting previous positive feature.
		 * @param index positive index.
		 * @return previous positive feature.
		 */
		public Matrix positivePrev(int index) {return triplet.positivePrev(index);}

		/**
		 * Adding previous positive feature.
		 * @param index positive index.
		 * @param positivePrev previous positive feature.
		 */
		public void positivePrevAdd(int index, Matrix positivePrev) {triplet.positivePrevAdd(index, positivePrev);}
		
		/**
		 * Adding previous positive feature.
		 * @param positive positive feature.
		 * @param positivePrev previous positive feature.
		 */
		public void positivePrevAdd(Matrix positive, Matrix positivePrev) {triplet.positivePrevAdd(positive, positivePrev);}

		/**
		 * Getting proxy group which is also called proxy matrix.
		 * @return proxy group which is also called proxy matrix.
		 */
		public Matrix proxyGroup() {
			if (this.proxies.size() == 0) return null;
			
			Matrix[] weights = new Matrix[this.proxies.size()];
			for (int i = 0; i < weights.length; i++) {
				Matrix weight = this.proxies.get(i);
				if (weight.rows() == 1)
					weights[i] = weight;
				else if (weight.columns() == 1)
					weights[i] = weight.transpose();
				else
					throw new IllegalArgumentException();
			}
			return weights[0].concatHorizontal(weights);
		}
		
		/**
		 * Getting proxy at specified index.
		 * @param index specified index.
		 * @return proxy at specified index.
		 */
		public Matrix proxy(int index) {return this.proxies.get(index);}
		
		/**
		 * Getting proxy count.
		 * @return proxy count.
		 */
		public int proxyCount() {return this.proxies.size();}
		
		/**
		 * Validating the quartet.
		 * @return true if the quartet is valid.
		 */
		public boolean validate() {
			if (triplet == null)
				return proxies.size() > 0;
			else
				return triplet.validate();
		}
		
		/**
		 * Getting index of proxy.
		 * @param proxy proxy.
		 * @return index of proxy.
		 */
		public int proxyIndexOf(Matrix proxy) {return this.proxies.indexOf(proxy);}

		/**
		 * Calculating similarity between anchor feature and proxy.
		 * @param proxyIndex proxy index.
		 * @return similarity between anchor feature and proxy.
		 */
		public NeuronValue proxySim(int proxyIndex) {return sim().sim(triplet.anchor, proxy(proxyIndex));}
		
		/**
		 * Calculating similarity between anchor feature, positive feature, and proxy.
		 * @param positiveIndex positive index.
		 * @param proxyIndex proxy index.
		 * @return similarity between anchor feature, positive feature, and proxy.
		 */
		public NeuronValue positiveSim(int positiveIndex, int proxyIndex) {
			return triplet.positiveSim(positiveIndex).add(proxySim(proxyIndex));
		}
		
		/**
		 * Calculating similarity between anchor feature, negative feature, and proxy.
		 * @param negativeIndex negative index.
		 * @param proxyIndex proxy index.
		 * @return similarity between anchor feature, negative feature, and proxy.
		 */
		public NeuronValue negativeSim(int negativeIndex, int proxyIndex) {
			return triplet.negativeSim(negativeIndex).add(proxySim(proxyIndex));
		}
		
		/**
		 * Calculating exponential sum.
		 * @return exponential sum.
		 */
		private NeuronValue expSum() {
			NeuronValue pSum = null;
			for (int pIndex = 0; triplet != null && pIndex < triplet.positives.size(); pIndex++) {
				NeuronValue pExp = triplet.positiveSim(pIndex).exp();
				pSum = pSum != null ? pSum.add(pExp) : pExp;
			}
			//Calculating negative sum.
			NeuronValue nSum = null;
			for (int nIndex = 0; triplet != null && nIndex < triplet.negatives.size(); nIndex++) {
				NeuronValue nExp = triplet.negativeSim(nIndex).exp();
				nSum = nSum != null ? nSum.add(nExp) : nExp;
			}
			//Calculating proxy sum.
			NeuronValue proxySum = null;
			for (int proxyIndex = 0; proxyIndex < proxies.size(); proxyIndex++) {
				NeuronValue proxyExp = proxySim(proxyIndex).exp();
				proxySum = proxySum != null ? proxySum.add(proxyExp) : proxyExp;
			}
			
			if (pSum == null && nSum == null && proxySum == null) throw new IllegalArgumentException();
			NeuronValue sum = null;
			if (pSum != null && nSum != null)
				sum = pSum.add(nSum);
			else if (pSum != null && nSum == null)
				sum = pSum;
			else if (pSum == null && nSum != null)
				sum = nSum;
			//
			if (sum != null)
				return proxySum != null ? sum.multiply(proxySum) : sum;
			else
				return proxySum;
		}
		
		/**
		 * Calculating positive soft-max probability.
		 * @param positiveIndex positive index.
		 * @param proxyIndex proxy index.
		 * @return positive soft-max probability.
		 */
		public NeuronValue positiveProb(int positiveIndex, int proxyIndex) {
			NeuronValue pExp = positiveSim(positiveIndex, proxyIndex).exp();
			return pExp.divide(expSum());
		}
		
		/**
		 * Calculating negative soft-max probability.
		 * @param negativeIndex negative index.
		 * @param proxyIndex proxy index.
		 * @return negative soft-max probability.
		 */
		public NeuronValue negativeProb(int negativeIndex, int proxyIndex) {
			NeuronValue nExp = negativeSim(negativeIndex, proxyIndex).exp();
			return nExp.divide(expSum());
		}

		/**
		 * Calculating positive softmax whose size is proxies count.
		 * @param positiveIndex positive index.
		 * @return positive softmax.
		 */
		public NeuronValue[] positiveSoftmax(int positiveIndex) {
			//Calculating positive exponent.
			NeuronValue pExp = positiveSim(positiveIndex, positiveIndex).exp();
			if (proxies.size() == 0) return new NeuronValue[] {pExp.divide(expSum())};

			//Calculating softmax.
			NeuronValue[] softmax = new NeuronValue[proxies.size()];
			for (int proxyIndex = 0; proxyIndex < proxies.size(); proxyIndex++) {
				NeuronValue proxyExp = proxySim(proxyIndex).exp();
				softmax[proxyIndex] = pExp.multiply(proxyExp);
			}
			NeuronValue sum = expSum();
			for (int proxyIndex = 0; proxyIndex < proxies.size(); proxyIndex++) {
				softmax[proxyIndex] = softmax[proxyIndex].divide(sum);
			}
			return softmax;
		}

		/**
		 * Calculating positive softmax. Rows indicates proxies and columns indicate positive features.
		 * In case there is no positives, there is only one row and columns indicates proxies.
		 * @return positive softmax.
		 */
		public NeuronValue[][] positiveSoftmax() {
			if (triplet == null) return null;
			if (proxies.size() == 0 && triplet.positives.size() == 0) return null;
			if (triplet.positives.size() == 0 && triplet.negatives.size() > 0) return null;
			
			return positiveSoftmax0();
		}
		
		/**
		 * Calculating positive entropy gradient. Rows indicates proxies and columns indicate positive features.
		 * In case there is no positives, there is only one row and columns indicates proxies.
		 * @param realProbs real probabilities array whose size is the number of proxies.
		 * @return positive entropy gradient.
		 */
		public NeuronValue[][] positiveEntropyGradient(NeuronValue[] realProbs) {
			if (triplet == null) return null;
			if (proxies.size() == 0 && triplet.positives.size() == 0) return null;
			if (triplet.positives.size() == 0 && triplet.negatives.size() > 0) return null;
			
			return positiveEntropyGradient0(realProbs);
		}
		
		/**
		 * Calculating positive softmax. Rows indicates proxies and columns indicate positive features.
		 * In case there is no positives, there is only one row and columns indicates proxies.
		 * @return positive softmax.
		 */
		private NeuronValue[][] positiveSoftmax0() {
			NeuronValue[][] softmax = null;
			if (proxies.size() == 0) { //No proxy.
				if (triplet == null || !triplet.validate()) throw new IllegalArgumentException();
				if (triplet.positives.size() == 0) {
					if (triplet.negatives.size() == 0) throw new IllegalArgumentException();
					softmax = new NeuronValue[1][triplet.negatives.size()];
					//Calculating negative sum.
					NeuronValue nSum = null;
					for (int nIndex = 0; nIndex < triplet.negatives.size(); nIndex++) {
						softmax[0][nIndex] = triplet.negativeSim(nIndex).exp();
						nSum = nSum != null ? nSum.add(softmax[0][nIndex]) : softmax[0][nIndex];
					}
					//Calculating softmax.
					for (int nIndex = 0; nIndex < triplet.negatives.size(); nIndex++) {
						softmax[0][nIndex] = softmax[0][nIndex].divide(nSum);
					}
				}
				else {
					softmax = new NeuronValue[1][triplet.positives.size()];
					//Calculating positive sum.
					NeuronValue pSum = null;
					for (int pIndex = 0; pIndex < triplet.positives.size(); pIndex++) {
						softmax[0][pIndex] = triplet.positiveSim(pIndex).exp();
						pSum = pSum != null ? pSum.add(softmax[0][pIndex]) : softmax[0][pIndex];
					}
					//Calculating negative sum.
					NeuronValue nSum = null;
					for (int nIndex = 0; nIndex < triplet.negatives.size(); nIndex++) {
						NeuronValue nExp = triplet.negativeSim(nIndex).exp();
						nSum = nSum != null ? nSum.add(nExp) : nExp;
					}
					//Calculating softmax.
					NeuronValue sum = pSum.add(nSum);
					for (int pIndex = 0; pIndex < triplet.positives.size(); pIndex++) {
						softmax[0][pIndex] = softmax[0][pIndex].divide(sum);
					}
				}
			}
			else if ( (triplet == null) || (!triplet.validate()) || (triplet.positives.size() == 0 && triplet.negatives.size() == 0) ) { //Having proxies but have no triplet.
				softmax = new NeuronValue[1][proxies.size()];
				//Calculating proxy sum.
				NeuronValue proxySum = null;
				for (int proxyIndex = 0; proxyIndex < proxies.size(); proxyIndex++) {
					softmax[0][proxyIndex] = proxySim(proxyIndex).exp();
					proxySum = proxySum != null ? proxySum.add(softmax[0][proxyIndex]) : softmax[0][proxyIndex];
				}
				//Calculating softmax.
				for (int proxyIndex = 0; proxyIndex < proxies.size(); proxyIndex++) {
					softmax[0][proxyIndex] = softmax[0][proxyIndex].divide(proxySum);
				}
			}
			else if (triplet.positives.size() == 0) { //Having proxies but have no positive.
				softmax = new NeuronValue[1][triplet.negatives.size()];
				//Calculating negative sum.
				NeuronValue nSum = null;
				for (int nIndex = 0; nIndex < triplet.negatives.size(); nIndex++) {
					softmax[0][nIndex] = triplet.negativeSim(nIndex).exp();
					nSum = nSum != null ? nSum.add(softmax[0][nIndex]) : softmax[0][nIndex];
				}
				//Calculating proxy sum.
				NeuronValue proxySum = null;
				for (int proxyIndex = 0; proxyIndex < proxies.size(); proxyIndex++) {
					NeuronValue proxyExp = proxySim(proxyIndex).exp();
					proxySum = proxySum != null ? proxySum.add(proxyExp) : proxyExp;
				}
				//Calculating softmax.
				NeuronValue sum = nSum.multiply(proxySum);
				for (int nIndex = 0; nIndex < triplet.negatives.size(); nIndex++) {
					softmax[0][nIndex] = softmax[0][nIndex].divide(sum);
				}
			}
			else { //Having proxies and having positives.
				//Calculating positive sum.
				NeuronValue[] positiveExps = new NeuronValue[triplet.positives.size()];
				NeuronValue pSum = null;
				for (int pIndex = 0; pIndex < triplet.positives.size(); pIndex++) {
					positiveExps[pIndex] = triplet.positiveSim(pIndex).exp();
					pSum = pSum != null ? pSum.add(positiveExps[pIndex]) : positiveExps[pIndex];
				}
				//Calculating negative sum.
				NeuronValue nSum = null;
				for (int nIndex = 0; nIndex < triplet.negatives.size(); nIndex++) {
					NeuronValue nExp = triplet.negativeSim(nIndex).exp();
					nSum = nSum != null ? nSum.add(nExp) : nExp;
				}
				//Calculating proxy sum.
				softmax = new NeuronValue[proxies.size()][triplet.positives.size()];
				NeuronValue proxySum = null;
				for (int proxyIndex = 0; proxyIndex < proxies.size(); proxyIndex++) {
					NeuronValue proxyExp = proxySim(proxyIndex).exp();
					proxySum = proxySum != null ? proxySum.add(proxyExp) : proxyExp;
					for (int pIndex = 0; pIndex < triplet.positives.size(); pIndex++) {
						softmax[proxyIndex][pIndex] = positiveExps[pIndex].multiply(proxyExp);
					}
				}
				//Calculating softmax.
				NeuronValue sum = nSum != null ? pSum.add(nSum).multiply(proxySum) : pSum.multiply(proxySum);
				for (int proxyIndex = 0; proxyIndex < proxies.size(); proxyIndex++) {
					for (int pIndex = 0; pIndex < triplet.positives.size(); pIndex++) {
						softmax[proxyIndex][pIndex] = softmax[proxyIndex][pIndex].divide(sum);
					}
				}
			}
			
			return softmax;
		}

		/**
		 * Calculating positive entropy gradient. Rows indicates proxies and columns indicate positive features.
		 * In case there is no positives, there is only one row and columns indicates proxies.
		 * @param realProbs real probabilities array whose size is the number of proxies.
		 * @return positive entropy gradient.
		 */
		private NeuronValue[][] positiveEntropyGradient0(NeuronValue[] realProbs) {
			NeuronValue[][] softmax = positiveSoftmax0();
			if (softmax == null) return null;
			
			int rows = softmax.length, columns = softmax[0].length;
			NeuronValue[][] grad = new NeuronValue[rows][columns];
			NeuronValue zero = softmax[0][0].zero(), unit = zero.unit();
			if (rows > 1) {
				NeuronValue uniform = unit.divide(rows);
				for (int column = 0; column < columns; column++) {
					for (int row = 0; row < rows; row++) {
						NeuronValue prob = softmax[row][column];
						NeuronValue sum = zero;
						for (int i = 0; i < rows; i++) {
							NeuronValue realProb = realProbs != null ? realProbs[i] : uniform;
							NeuronValue value = i==row ? realProb.multiply(unit.subtract(prob)) : realProb.multiply(prob.negative());
							sum = sum.add(value);
						}
						grad[row][column] = sum;
					}
				}
			}
			else {
				NeuronValue uniform = unit.divide(columns); //Please pay attention this code line which is only correct in degradation case.
				for (int column = 0; column < columns; column++) {
					NeuronValue prob = softmax[0][column];
					NeuronValue sum = zero;
					for (int i = 0; i < columns; i++) {
						NeuronValue realProb = realProbs != null ? realProbs[i] : uniform;
						NeuronValue value = i==column ? realProb.multiply(unit.subtract(prob)) : realProb.multiply(prob.negative());
						sum = sum.add(value);
					}
					grad[0][column] = sum;
				}
			}
			
			return grad;
		}
		
	}
	
	
	/**
	 * This class represents list of proxy group (proxy group list).
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	class ProxyGroupList implements Cloneable, Serializable {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;
		
		/**
		 * Stack of proxy groups (stack of proxy matrices).
		 */
		protected MatrixStack groups = null;
		
		/**
		 * Constructor with proxy groups.
		 * @param proxyGroups proxy groups.
		 */
		public ProxyGroupList(Matrix...proxyGroups) {this.groups = new MatrixStack(proxyGroups);}
		
		/**
		 * Constructor with proxy groups.
		 * @param proxyGroups proxy groups.
		 */
		public ProxyGroupList(MatrixStack proxyGroups) {this.groups = proxyGroups;}

		/**
		 * Getting size of proxy group list.
		 * @return size of this proxy group list.
		 */
		public int size() {return this.groups.depth();}
		
		/**
		 * Getting proxy group at specified index.
		 * @param groupIndex group index.
		 * @return proxy group at specified index.
		 */
		public Matrix get(int groupIndex) {return this.groups.get(groupIndex);}
		
		/**
		 * Getting proxy.
		 * @param groupIndex group index.
		 * @param proxyIndex proxy index.
		 * @return proxy at specified index.
		 */
		public Matrix proxy(int groupIndex, int proxyIndex) {return extractProxy(get(groupIndex), proxyIndex);}
		
		/**
		 * Getting count of proxies.
		 * @param groupIndex group index.
		 * @return count of proxies.
		 */
		public int proxyCount(int groupIndex) {return get(groupIndex).rows();}

		/**
		 * Accumulating proxy group.
		 * @param groupIndex group index.
		 * @param proxyGroup proxy group.
		 * @return this proxy group list.
		 */
		public ProxyGroupList accum(int groupIndex, Matrix proxyGroup) {
			Matrix matrix = get(groupIndex).add(proxyGroup);
			this.groups.set(groupIndex, matrix);
			return this;
		}
		
		/**
		 * Accumulating proxy group list.
		 * @param proxyGroupList proxy group list.
		 * @return this proxy group list.
		 */
		public ProxyGroupList accum(MatrixStack proxyGroupList) {
			this.groups = (MatrixStack)this.groups.add(proxyGroupList);
			return this;
		}

		/**
		 * Accumulating proxy group list.
		 * @param proxyGroupList proxy group list.
		 * @return this proxy group list.
		 */
		public ProxyGroupList accum(ProxyGroupList proxyGroupList) {
			return accum(proxyGroupList.groups);
		}

		/**
		 * Accumulating proxy group list.
		 * @param proxyGroupList proxy group list.
		 * @return this proxy group list.
		 */
		public ProxyGroupList accum(Matrix...proxyGroupList) {
			return accum(new MatrixStack(proxyGroupList));
		}
		
		/**
		 * Accumulating proxy group error.
		 * @param groupIndex group index.
		 * @param proxyGroupError proxy group error.
		 * @return this proxy group list.
		 */
		public ProxyGroupList accum(int groupIndex, Error proxyGroupError) {
			return accum(groupIndex, proxyGroupError.error());
		}
		
		/**
		 * Accumulating proxy group errors.
		 * @param proxyGroupErrors proxy group errors.
		 * @return this proxy group list.
		 */
		public ProxyGroupList accum(Error...proxyGroupErrors) {
			Matrix[] matrixErrors = new Matrix[proxyGroupErrors.length];
			for (int i = 0; i < matrixErrors.length; i++) matrixErrors[i] = proxyGroupErrors[i].error();
			return accum(matrixErrors);
		}

		/**
		 * Multiplying proxy group with specified value.
		 * @param groupIndex group index.
		 * @param value specified value.
		 * @return this proxy group list.
		 */
		public ProxyGroupList accumMultiply(int groupIndex, double value) {
			Matrix matrix = get(groupIndex).multiply0(value);
			this.groups.set(groupIndex, matrix);
			return this;
		}
		
		/**
		 * Multiplying proxy group list with specified value.
		 * @param value specified value.
		 * @return this proxy group list.
		 */
		public ProxyGroupList accumMultiply(double value) {
			this.groups = (MatrixStack)this.groups.multiply0(value);
			return this;
		}

		/**
		 * Divided proxy group by specified value.
		 * @param groupIndex group index.
		 * @param value specified value.
		 * @return this proxy group list.
		 */
		public ProxyGroupList accumDivide(int groupIndex, double value) {
			Matrix matrix = get(groupIndex).divide0(value);
			this.groups.set(groupIndex, matrix);
			return this;
		}
		
		/**
		 * Divided proxy group list by specified value.
		 * @param value specified value.
		 * @return this proxy group list.
		 */
		public ProxyGroupList accumDivide(double value) {
			this.groups = (MatrixStack)this.groups.divide0(value);
			return this;
		}

	}
	
	
	/**
	 * THis class represents accumulator of proxy group list.
	 * @author Loc Nguyen
	 * @version 1.0
	 *
	 */
	class ProxyGroupListAccumulator extends Accumulator.AccumulatorAbstract<ProxyGroupList> {
		
		/**
		 * Serial version UID for serializable class. 
		 */
		private static final long serialVersionUID = 1L;

		/**
		 * Constructor with proxy group list.
		 * @param value proxy group list.
		 */
		public ProxyGroupListAccumulator(ProxyGroupList proxyGroupList) {super(proxyGroupList);}

		/**
		 * Constructor with matrix stack as proxy group list.
		 * @param proxyGroupList matrix stack as proxy group list.
		 */
		public ProxyGroupListAccumulator(MatrixStack proxyGroupList) {super(new ProxyGroupList(proxyGroupList));}
		
		@Override
		public Accumulator<ProxyGroupList> accum(ProxyGroupList value) {
			if (value == null) return this;
			this.sum = this.sum != null ? this.sum.accum(value) : value;
			this.count++;
			return this;
		}

		/**
		 * Accumulating matrix stack.
		 * @param proxyGroupList matrix stack as proxy group list.
		 * @return this accumulator.
		 */
		public ProxyGroupListAccumulator accum(MatrixStack proxyGroupList) {
			if (proxyGroupList == null) return this;
			this.sum = this.sum != null ? this.sum.accum(proxyGroupList) : new ProxyGroupList(proxyGroupList);
			this.count++;
			return this;
		}

		@Override
		public int count() {return count;}

		@Override
		public ProxyGroupList mean() {
			return sum != null && count > 0 ? sum.accumDivide(count) : null;
		}
		
	}
	
	
	/**
	 * List of proxy groups (proxy group list) known as learnable parameter.
	 */
	protected ProxyGroupList proxyGroupList = null;
	
	
	/**
	 * Backward information: Accumulation of gradients of proxy group list.
	 */
	private ProxyGroupListAccumulator dProxyGroupListAccum = null;
	
	
	/**
	 * Constructor with neuron channel, activation function, convolutional activation function, and identifier reference.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 * @param idRef identifier reference.
	 */
	public ProxyNCA(int neuronChannel, Function activateRef, Function convActivateRef, Id idRef) {
		super(neuronChannel, activateRef, convActivateRef, idRef);
		config.put(IS_PROXY_FIELD, IS_PROXY_DEFAULT);
		config.put(AUGMENTED_FIELD, AUGMENTED_DEFAULT);
		config.put(PIECE_SIZE_FIELD, PIECE_SIZE_DEFAULT);
	}

	
	/**
	 * Constructor with neuron channel, activation function, and convolutional activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 * @param convActivateRef convolutional activation function.
	 */
	public ProxyNCA(int neuronChannel, Function activateRef, Function convActivateRef) {
		this(neuronChannel, activateRef, convActivateRef, null);
	}

	
	/**
	 * Constructor with neuron channel and activation function.
	 * @param neuronChannel neuron channel.
	 * @param activateRef activation function.
	 */
	public ProxyNCA(int neuronChannel, Function activateRef) {this(neuronChannel, activateRef, null, null);}

	
	/**
	 * Constructor with neuron channel.
	 * @param neuronChannel neuron channel.
	 */
	public ProxyNCA(int neuronChannel) {this(neuronChannel, null, null, null);}


	/**
	 * Creating similarity utility.
	 * @return similarity utility.
	 */
	protected Sim.MatrixSim sim() {return new Sim.DistanceMatrixSim();}
	
	
	/**
	 * Checking whether proxy is vectorized.
	 * @return whether proxy is vectorized.
	 */
	protected boolean isProxyVectorized() {return false;}

	
	/**
	 * Extracting proxy at specified row.
	 * @param proxies proxies matrix.
	 * @param row specified row.
	 * @return proxy at specified row.
	 */
	private Matrix extractProxy(Matrix proxies, int row) {
		return isProxyVectorized() ? proxies.getRow(row).transpose() : proxies.getRow(row);
	}
	
	
	/**
	 * Getting count of proxy groups.
	 * @return count of proxy groups.
	 */
	public int proxyGroupCount() {return proxyGroupList != null ? proxyGroupList.size() : 0;}
	
	
	/**
	 * Getting proxy group.
	 * @param groupIndex group index.
	 * @return proxy group.
	 */
	public Matrix proxyGroup(int groupIndex) {return proxyGroupList != null ? proxyGroupList.get(groupIndex) : null;}
	
	
	/**
	 * Getting count of proxies.
	 * @param groupIndex group index.
	 * @return count of proxies.
	 */
	public int proxyCount(int groupIndex) {return proxyGroupList != null ? proxyGroupList.get(groupIndex).rows() : 0;}
	
	
	/**
	 * Getting proxy.
	 * @param groupIndex group index.
	 * @param proxyIndex proxy index.
	 * @return proxy.
	 */
	public Matrix proxy(int groupIndex, int proxyIndex) {return proxyGroupList != null ? proxyGroupList.proxy(groupIndex, proxyIndex) : null;}

	
	@Override
	public void reset() {
		super.reset();
		reset0();
	}

	
	/**
	 * Internal resetting.
	 */
	private void reset0() {
		this.proxyGroupList = null;
		resetBackwardInfo0();
	}

	
	@Override
	protected boolean initialize(Size inputSize, Size middleSize, Size outputSize) {
		reset0();
		return super.initialize(inputSize, middleSize, outputSize);
	}


	@Override
	public boolean initialize(Size inputSize, Size middleSize, Size outputSize, Size nCoreClasses) {
		if (!super.initialize(inputSize, middleSize, outputSize, nCoreClasses)) return false;
		if (getNumberOfGroups() == 0 || !paramIsProxy()) return true;
		
		Matrix output = getOutput();
		int columns = output.rows()*output.columns();
		Matrix[] proxyGroups = new Matrix[getNumberOfGroups()]; 
		for (int groupIndex = 0; groupIndex < proxyGroups.length; groupIndex++) {
			proxyGroups[groupIndex] = output.create(new Size(columns, getNumberOfClasses(groupIndex)));
			MatrixUtil.fill(proxyGroups[groupIndex], new Random());
		}
		this.proxyGroupList = new ProxyGroupList(proxyGroups);
		return true;
	}


	/**
	 * Creating triplet.
	 * @param anchor anchor.
	 * @param positives positives which can be empty.
	 * @param negatives negatives which can be empty.
	 * @return triplet.
	 */
	private Triplet create(Matrix anchor, List<Matrix> positives, List<Matrix> negatives) {
		if (anchor == null) return null;
		Triplet triplet = new Triplet();
		triplet.anchor = anchor;
		triplet.positives.addAll(positives);
		triplet.negatives.addAll(negatives);
		return triplet.validate() ? triplet : null;
	}
	
	
	/**
	 * Creating quartet.
	 * @param anchor anchor.
	 * @param positives positives which can be empty.
	 * @param negatives negatives which can be empty.
	 * @param proxyGroup proxy group which can be null.
	 * @return quartet.
	 */
	private Quartet create(Matrix anchor, List<Matrix> positives, List<Matrix> negatives, Matrix proxyGroup) {
		Triplet triplet = create(anchor, positives, negatives);
		if (triplet == null) return null;
		Quartet quartet = new Quartet(triplet, proxyGroup);
		return quartet.validate() ? quartet : null;
	}
	
	
	/**
	 * Creating quartets.
	 * @param origins original rasters.
	 * @param augments augmented rasters.
	 * @param groupIndex group index.
	 * @return quartets.
	 */
	private List<Quartet> createQuartets(List<Raster> origins, List<Raster> augments, int groupIndex) {
		List<Quartet> quartets = Util.newList(0);
		for (int i = 0; i < origins.size(); i++) {
			List<Matrix> positives = Util.newList(0);
			List<Matrix> negatives = Util.newList(0);
			Map<Matrix, Matrix> prevs = Util.newMap(0);

			if (i < augments.size()) addOutput(augments.get(i), positives, prevs);
			
			for (int j = 0; j < origins.size() && j != i; j++) {
				if (isLabeled()) {
					int isSame = isSameClasses(origins.get(i), origins.get(j), groupIndex);
					if (isSame == 0)
						addOutput(origins.get(j), positives, prevs);
					else if (isSame > 0)
						addOutput(origins.get(j), negatives, prevs);
				}
				else
					addOutput(origins.get(j), negatives, prevs);
			}
			for (int j = 0; j < augments.size() && j != i; j++) {
				if (isLabeled()) {
					int isSame = isSameClasses(origins.get(i), augments.get(j), groupIndex);
					if (isSame == 0)
						addOutput(augments.get(j), positives, prevs);
					else if (isSame > 0)
						addOutput(augments.get(j), negatives, prevs);
				}
				else
					addOutput(augments.get(j), negatives, prevs);
			}
			
//			if (this.proxyGroupList == null && positives.size() == 0) continue;
//			if (positives.size() == 0 && negatives.size() > 0) continue;
			
			Quartet quartet = null;
			Matrix anchor = addOutput(origins.get(i), null, prevs);
			quartet = create(anchor, positives, negatives, this.proxyGroupList != null ? this.proxyGroupList.get(groupIndex) : null);
			if (quartet != null) {
				quartet.triplet.prevs.putAll(prevs);
				quartet.triplet.anchorInputRaster = origins.get(i);
				quartets.add(quartet);
			}
		}
		
		return quartets;
	}
	
	
	/**
	 * Adding output.
	 * @param input raster input.
	 * @param outputs outputs which can be null.
	 * @param outputPrevs map of previous outputs which can be null.
	 */
	private Matrix addOutput(Raster input, List<Matrix> outputs, Map<Matrix, Matrix> outputPrevs) {
		if (input == null) return null;
		Matrix output = evaluate(input);
		output = output != null ? (isProxyVectorized() ? output.vec() : output) : null;
		Matrix outputPrev = getOutputLayer().getInput();
		outputPrev = outputPrev != null ? (isProxyVectorized() ? outputPrev.vec() : outputPrev) : null;
		
		if (outputs != null && output != null) outputs.add(output);
		if (outputPrevs != null && outputPrev != null) outputPrevs.put(output, outputPrev);
		return output;
	}
	
	
	/**
	 * Calculating the last bias.
	 * @param output triplet output.
	 * @param realOutputProbs real output probabilities.
	 * @param outputLayer output layer.
	 * @param params additional parameters.
	 * @return last bias.
	 */
	protected Matrix[] calcOutputError(Quartet output, NeuronValue[] realOutputProbs, MatrixLayerAbstract outputLayer, int groupIndex, Object... params) {
		NeuronValue[][] grads = output.positiveEntropyGradient(realOutputProbs);
		if (grads == null) return null;
		
		Matrix[] dProxyGroup = null; //This is proxy group because proxy group is matrix and every proxy is also a 1-row matrix (row vector).
		Matrix dThetaAccum = null;
		if (this.proxyGroupList == null) {
			if (grads.length != 1 || grads[0].length != output.positiveSize()) throw new IllegalArgumentException();
			for (int positiveIndex = 0; positiveIndex < grads[0].length; positiveIndex++) {
				NeuronValue grad = grads[0][positiveIndex];
				
				Function f = outputLayer != null ? outputLayer.getOutputActivateRef() : null;
				Matrix positive = positiveIndex < output.positiveSize() ? output.positive(positiveIndex) : null;
				Matrix positivePrev = positive != null ? output.positivePrev(positiveIndex) : null;
				Matrix dsx2 = positive != null ? sim().dsim(positive, output.anchor(), f, positivePrev, output.anchorPrev()) : null;
				//
				Matrix dTheta = dsx2 != null ? dsx2.multiply0(grad) : null;
				
				if (dTheta != null) dThetaAccum = dThetaAccum != null ? dThetaAccum.add(dTheta) : dTheta;
			}
		}
		else if (output.positiveSize() == 0) {
			if (grads.length != 1 || grads[0].length != this.proxyCount(groupIndex)) throw new IllegalArgumentException();
			dProxyGroup = new Matrix[grads[0].length];
			for (int proxyIndex = 0; proxyIndex < grads[0].length; proxyIndex++) {
				Matrix proxy = this.proxyGroupList.proxy(groupIndex, proxyIndex);
				NeuronValue grad = grads[0][proxyIndex];

				Matrix dsw = sim().dsim(output.anchor(), proxy, false).multiply0(grad);
				
				dProxyGroup[proxyIndex] = dProxyGroup[proxyIndex] != null ? dProxyGroup[proxyIndex].add(dsw) : dsw;
			}
		}
		else {
			if (grads.length != this.proxyCount(groupIndex)) throw new IllegalArgumentException();
			dProxyGroup = new Matrix[grads.length];
			for (int proxyIndex = 0; proxyIndex < grads.length; proxyIndex++) {
				Matrix proxy = this.proxyGroupList.proxy(groupIndex, proxyIndex);
				for (int positiveIndex = 0; positiveIndex < grads[proxyIndex].length; positiveIndex++) {
					NeuronValue grad = grads[proxyIndex][positiveIndex];
					
					Matrix dsw = sim().dsim(output.anchor(), proxy, false).multiply0(grad);
					
					Function f = outputLayer != null ? outputLayer.getOutputActivateRef() : null;
					Matrix dsx1 = sim().dsim(output.anchor(), proxy, f, output.anchorPrev(), null);
					//
					Matrix positive = positiveIndex < output.positiveSize() ? output.positive(positiveIndex) : null;
					Matrix positivePrev = positive != null ? output.positivePrev(positiveIndex) : null;
					Matrix dsx2 = positive != null ? sim().dsim(positive, output.anchor(), f, positivePrev, output.anchorPrev()) : null;
					//
					Matrix dTheta = dsx2 != null ? dsx1.add(dsx2).multiply0(grad) : dsx1.multiply0(grad);
					
					dProxyGroup[proxyIndex] = dProxyGroup[proxyIndex] != null ? dProxyGroup[proxyIndex].add(dsw) : dsw;
					dThetaAccum = dThetaAccum != null ? dThetaAccum.add(dTheta) : dTheta;
				}
				if (dProxyGroup[proxyIndex].rows() != 1) dProxyGroup[proxyIndex] = dProxyGroup[proxyIndex].transpose();
			}
		}
		if (dProxyGroup == null && dThetaAccum == null) return null;
		
		if (dThetaAccum != null) {
			Matrix OUTPUT = getOutput();
			if (dThetaAccum.rows() == OUTPUT.rows()*OUTPUT.columns() && dThetaAccum.columns() == 1) dThetaAccum = dThetaAccum.vecInverse(OUTPUT.rows());
		}
		return new Matrix[] {dProxyGroup != null ? dProxyGroup[0].concatHorizontal(dProxyGroup) : null, dThetaAccum};
	}
	
	
	/**
	 * Calculating the last bias.
	 * @param piece piece of sample.
	 * @return the last bias.
	 */
	private Matrix[] calcOutputError(Iterable<Raster> piece) {
		List<Raster> origins = Util.newList(0);
		List<Raster> augments = Util.newList(0);
		for (Raster origin : piece) {
			if (origin == null) continue;
			if (paramIsAugmented() || !isLabeled()) {
				Raster augment = new Augmentor(origin).augmentRandom();
				if (augment != null) {
					origins.add(origin);
					augments.add(augment);
				}
			}
			else
				origins.add(origin);
		}
		
		if (origins.size() != augments.size()) augments.clear(); //Please pay attention to this line.
		
		int groupCount = this.proxyGroupList != null ? this.proxyGroupList.size() : 1;
		List<Matrix> dProxyGroupList = Util.newList(0);
		Matrix dThetaAccum = null;
		for (int groupIndex = 0; groupIndex < groupCount; groupIndex++) {
			List<Quartet> quartets = Util.newList(0);
			quartets.addAll(createQuartets(origins, augments, groupIndex));
			quartets.addAll(createQuartets(augments, origins, groupIndex));
			if (quartets.size() == 0) continue;
			
			Matrix dProxyGroup = null;
			for (Quartet quartet : quartets) {
				NeuronValue[] realOutputProbs = null;
				if (quartet.triplet.anchorInputRaster != null && isLabeled() && this.proxyGroupList != null) {
					int classIndex = classOf(groupIndex, quartet.triplet.anchorInputRaster);
					if (classIndex >= 0) realOutputProbs = createOutputByClass(groupIndex, classIndex);
				}
				
				Matrix[] errors = calcOutputError(quartet, realOutputProbs, getOutputLayer(), groupIndex);
				if (errors == null) continue;
				if (errors[0] != null) dProxyGroup = dProxyGroup != null ? dProxyGroup.add(errors[0]) : errors[0];
				if (errors[1] != null) dThetaAccum = dThetaAccum != null ? dThetaAccum.add(errors[1]) : errors[1];
			}
			if (dProxyGroup != null) dProxyGroupList.add(dProxyGroup);
		}
		if (dProxyGroupList.size() == 0 && dThetaAccum == null) return null;
		
		MatrixStack dProxyGroupStack = dProxyGroupList.size() > 0 ? new MatrixStack(dProxyGroupList.toArray(new Matrix[] {})) : null;
		return new Matrix[] {dProxyGroupStack, dThetaAccum};
	}

	
	@Override
	protected Error[] learnRaster(Iterable<Raster> sample, MatrixLayer focus, boolean learning, double learningRate) {
		if (this.proxyGroupList == null) return super.learnRaster(sample, focus, learning, learningRate);
		
		List<Error> outputErrorList = Util.newList(0);
		List<Raster> piece = Util.newList(0);
		boolean trained = false;
		for (Raster raster : sample) {
			if ((paramGetPieceSize() > 0 && piece.size() < paramGetPieceSize()) || (paramGetPieceSize() <= 0)) {
				piece.add(raster);
				continue;
			}

			if (piece.size() >= 2) {
				Matrix[] errors = calcOutputError(piece);
				piece.clear();
				if (errors == null) continue;
				
				if (errors[0] != null) {
					MatrixStack errorStack = (MatrixStack)errors[0];
					this.dProxyGroupListAccum = this.dProxyGroupListAccum != null ? this.dProxyGroupListAccum.accum(errorStack) : new ProxyGroupListAccumulator(errorStack);
				}
				if (errors[1] != null) outputErrorList.add(new Error(errors[1]));
				trained = true;
			}
		}
		
		if ((piece.size() >= 2) || (piece.size() == 1 && !trained)){
			Matrix[] errors = calcOutputError(piece);
			piece.clear();
			
			if (errors != null && errors[0] != null) {
				MatrixStack errorStack = (MatrixStack)errors[0];
				this.dProxyGroupListAccum = this.dProxyGroupListAccum != null ? this.dProxyGroupListAccum.accum(errorStack) : new ProxyGroupListAccumulator(errorStack);
			}
			if (errors != null && errors[1] != null) outputErrorList.add(new Error(errors[1]));
			trained = true;
		}
		
//		if (learning && this.dProxyGroupListAccum != null) updateParametersFromBackwardInfo0(this.dProxyGroupListAccum.count(), learningRate); //This code line is redundant. 
		
		for (Error error : outputErrorList) error.addLayerOInput(this); //Please pay attention to this code line for tracking errors.
		
		Error[]  outputErrors = outputErrorList.size() > 0 ? backward(outputErrorList.toArray(new Error[] {}), focus, learning, learningRate) : null;
		if (outputErrors != null) learnRasterVerify(sample);
		return outputErrors;
	}

	
	/**
	 * Checking whether two rasters have same classes.
	 * @param raster1 raster 1.
	 * @param raster2 raster 2.
	 * @param groupIndex group index.
	 * @return 0: equal, 1: different, -1: unknown.
	 */
	private int isSameClasses(Raster raster1, Raster raster2, int groupIndex) {
		int classIndex1 = classOf(groupIndex, raster1);
		int classIndex2 = classOf(groupIndex, raster2);
		if (classIndex1 == classIndex2 && classIndex1 >= 0)
			return 0;
		else if (classIndex1 < 0 && classIndex2 < 0)
			return -1;
		else
			return 1;
	}

	
	@Override
	protected Matrix calcOutputError(Matrix output, Matrix realOutput, MatrixLayerAbstract outputLayer, Object... params) {
		if (this.proxyGroupList == null) return super.calcOutputError(output, realOutput, outputLayer, params);
		
		//Calculating entropy likelihoods.
		Matrix likelihood = LikelihoodGradient.entropyGradientByColumn(output, realOutput); //Real output is matrix of probabilities. 
		
		int groupCount = this.proxyGroupList.size();
		Matrix[] proxyErrors = new Matrix[groupCount];
		Matrix lastErrorSum = null;
		for (int groupIndex = 0; groupIndex < groupCount; groupIndex++) {
			Matrix proxyGroup = this.proxyGroupList.get(groupIndex);
			Matrix proxyGroupError = proxyGroup.create(new Size(proxyGroup.columns(), proxyGroup.rows()));
			if (proxyGroupError.rows() != likelihood.rows()) throw new IllegalArgumentException();
			for (int row = 0; row < proxyGroup.rows(); row++) {
				Matrix proxy = extractProxy(proxyGroup, row);
	
				//Calculating proxy error.
				proxyErrors[groupIndex] = sim().dsim(output, proxy, false).multiply0(likelihood.get(row, groupIndex));
				for (int column = 0; column < proxyGroupError.columns(); column++)
					proxyGroupError.set(row, column, proxyErrors[groupIndex].get(column, 0));
				
				//Calculating lass error.
				Matrix lastError = sim().dsim(output, proxy, true);
				lastErrorSum = lastErrorSum != null ? lastErrorSum.add(lastError) : lastError;
			}
		}
		MatrixStack proxyErrorStack = new MatrixStack(proxyErrors);
		this.dProxyGroupListAccum = this.dProxyGroupListAccum != null ? this.dProxyGroupListAccum.accum(proxyErrorStack) : new ProxyGroupListAccumulator(proxyErrorStack);
		
		if (outputLayer == null) return lastErrorSum;
		Matrix input = outputLayer.getInput();
		Matrix derivative = input != null ? input.derivativeWise(outputLayer.getOutputActivateRef()) : null;
		return derivative != null ? derivative.multiplyWise(lastErrorSum) : lastErrorSum;
	}


	@Override
	public Error[] backward(Error[] outputErrors, MatrixLayer focus, boolean learning, double learningRate) {
		outputErrors = super.backward(outputErrors, focus, learning, learningRate);
		if (learning) updateParametersFromBackwardInfo0(outputErrors.length, learningRate);
		return outputErrors;
	}


	@Override
	public void updateParametersFromBackwardInfo(int recordCount, double learningRate) {
		super.updateParametersFromBackwardInfo(recordCount, learningRate);
		updateParametersFromBackwardInfo0(recordCount, learningRate);
	}

	
	/**
	 * Updating parameters from backward information.
	 * @param recordCount record count.
	 * @param learningRate learning rate.
	 */
	private void updateParametersFromBackwardInfo0(int recordCount, double learningRate) {
		if (this.proxyGroupList != null && this.dProxyGroupListAccum != null) {
			assert(this.dProxyGroupListAccum.count() == recordCount);
			ProxyGroupList mean = this.dProxyGroupListAccum.meanAndClear();
			if (mean != null) mean = mean.accumMultiply(learningRate);
			if (mean != null) this.proxyGroupList = this.proxyGroupList.accum(mean);
		}
		this.dProxyGroupListAccum = null;
	}
	
	
	@Override
	public void resetBackwardInfo() {
		super.resetBackwardInfo();
		resetBackwardInfo0();
	}

	
	/**
	 * Resetting backward information.
	 */
	private void resetBackwardInfo0() {this.dProxyGroupListAccum = null;}
	
	
	/**
	 * Getting proxy flag.
	 * @return proxy flag.
	 */
	boolean paramIsProxy() {
		return config.containsKey(IS_PROXY_FIELD) ? config.getAsBoolean(IS_PROXY_FIELD) : IS_PROXY_DEFAULT;
	}
	
	
	/**
	 * Setting proxy flag.
	 * @param isProxy proxy flag.
	 * @return this Proxy-NCA.
	 */
	ProxyNCA paramSetProxy(boolean isProxy) {
		config.put(IS_PROXY_FIELD, isProxy);
		return this;
	}

	
	/**
	 * Getting augmentation flag.
	 * @return augmentation flag.
	 */
	boolean paramIsAugmented() {
		return config.containsKey(AUGMENTED_FIELD) ? config.getAsBoolean(AUGMENTED_FIELD) : AUGMENTED_DEFAULT;
	}
	
	
	/**
	 * Setting augmentation flag.
	 * @param augmented augmentation flag.
	 * @return this Proxy-NCA.
	 */
	ProxyNCA paramSetAugmented(boolean augmented) {
		config.put(AUGMENTED_FIELD, augmented);
		return this;
	}


	/**
	 * Getting piece size.
	 * @return piece size.
	 */
	int paramGetPieceSize() {
		int pieceSize = config.getAsInt(PIECE_SIZE_FIELD);
		return pieceSize < 2 ? PIECE_SIZE_DEFAULT : pieceSize;
	}
	
	
	/**
	 * Setting piece size.
	 * @param pieceSize piece size.
	 * @return this model.
	 */
	ProxyNCA paramSetPieceSize(int pieceSize) {
		pieceSize = pieceSize < 2 ? PIECE_SIZE_DEFAULT : pieceSize;
		config.put(PIECE_SIZE_FIELD, pieceSize);
		return this;
	}


}
