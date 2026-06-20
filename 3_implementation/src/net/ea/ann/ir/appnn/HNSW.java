/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.ir.appnn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

import net.ea.ann.ir.AppNNAbstract.MatrixFeatureAppNNAbstract;

/**
 * This class is a default and simple implementation of Hierarchical Navigable Small World (HNSW) algorithm.
 * @author Gemini 2026
 * @version 1.0
 *
 */
public abstract class HNSW extends MatrixFeatureAppNNAbstract {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	private final int M;              // Max number of connections per node
	private final int maxLevel;      // Maximum layers
	private final double levelLambda; // Controls level distribution
	private final Random random = new Random();
	
	// Map to store neighbors at each level: NodeID -> Level -> List of NeighborIDs
	private final Map<Integer, Map<Integer, List<Integer>>> graph = new HashMap<>();
	// Map to store the actual vector data
	private final Map<Integer, float[]> vectors = new HashMap<>();
	
	private int entryPointId = -1;
	private int currentMaxLevel = -1;
	
	
	public HNSW(int M) {
		this.M = M;
		this.levelLambda = 1.0 / Math.log(M);
		this.maxLevel = 10; // Cap for safety
	}
	
	
	/**
	 * Calculates the insertion level for a new node using an exponential distribution.
	 */
	private int getRandomLevel() {
		double r = -Math.log(random.nextDouble()) * levelLambda;
		return Math.min((int) r, maxLevel);
	}
	
	
	/**
	 * Euclidean distance between two vectors.
	 */
	private float getDistance(float[] v1, float[] v2) {
		float sum = 0;
		for (int i = 0; i < v1.length; i++) {
			float diff = v1[i] - v2[i];
			sum += diff * diff;
		}
		return (float) Math.sqrt(sum);
	}
	
	
	/**
	 * Adds a new vector to the HNSW index.
	 */
	public void add(int id, float[] vector) {
		vectors.put(id, vector);
		int nodeLevel = getRandomLevel();
		
		if (entryPointId == -1) {
			entryPointId = id;
			currentMaxLevel = nodeLevel;
			initializeNodeInGraph(id, nodeLevel);
			return;
		}
		
		int currObj = entryPointId;
		@SuppressWarnings("unused")
		float currDist = getDistance(vector, vectors.get(currObj));
		
		// 1. Greedy search through upper layers to find a close entry point for the target level
		for (int i = currentMaxLevel; i > nodeLevel; i--) {
			currObj = findNearestNeighborGreedy(vector, currObj, i);
		}
		
		// 2. Insert node and link neighbors from nodeLevel down to 0
		initializeNodeInGraph(id, nodeLevel);
		for (int i = Math.min(nodeLevel, currentMaxLevel); i >= 0; i--) {
			// Find candidates using the logic discussed in our previous conversations
			PriorityQueue<NodeDistance> candidates = searchLayer(vector, currObj, M, i);
			
			// Link bi-directionally
			for (NodeDistance candidate : candidates) {
				link(id, candidate.id, i);
				link(candidate.id, id, i);
				// In a real implementation, you would prune connections here if they exceed M
			}
			currObj = candidates.peek().id;
		}
		
		if (nodeLevel > currentMaxLevel) {
			currentMaxLevel = nodeLevel;
			entryPointId = id;
		}
	}
	
	
	private void link(int u, int v, int level) {
		graph.get(u).get(level).add(v);
	}
	
	
	private void initializeNodeInGraph(int id, int level) {
		Map<Integer, List<Integer>> levels = new HashMap<>();
		for (int i = 0; i <= level; i++) {
			levels.put(i, new ArrayList<>());
		}
		graph.put(id, levels);
	}
	
	
	private int findNearestNeighborGreedy(float[] query, int entryId, int level) {
		int curr = entryId;
		float minDist = getDistance(query, vectors.get(curr));
		boolean changed = true;
		
		while (changed) {
			changed = false;
			for (int neighbor : graph.get(curr).get(level)) {
				float d = getDistance(query, vectors.get(neighbor));
				if (d < minDist) {
					minDist = d;
					curr = neighbor;
					changed = true;
				}
			}
		}
		return curr;
	}
	
	
	// Helper class for PriorityQueues
	private static class NodeDistance implements Comparable<NodeDistance> {
		int id;
		float distance;
		NodeDistance(int id, float distance) { this.id = id; this.distance = distance; }
		@Override public int compareTo(NodeDistance o) { return Float.compare(this.distance, o.distance); }
	}
	
	
	/**
	 * Implements the SearchLayer algorithm (Best-First Search with Candidate List C).
	 */
	private PriorityQueue<NodeDistance> searchLayer(float[] query, int entryId, int ef, int level) {
		Set<Integer> visited = new HashSet<>();
		PriorityQueue<NodeDistance> candidates = new PriorityQueue<>(); // Min-heap (C)
		PriorityQueue<NodeDistance> foundNeighbors = new PriorityQueue<>(Collections.reverseOrder()); // Max-heap (W)
		
		float dist = getDistance(query, vectors.get(entryId));
		NodeDistance entry = new NodeDistance(entryId, dist);
		
		candidates.add(entry);
		foundNeighbors.add(entry);
		visited.add(entryId);
		
		while (!candidates.isEmpty()) {
			NodeDistance c = candidates.poll();
			NodeDistance f = foundNeighbors.peek();
			
			if (c.distance > f.distance) break; // Early exit condition
			
			for (int e : graph.get(c.id).get(level)) {
				if (!visited.contains(e)) {
					visited.add(e);
					float d = getDistance(query, vectors.get(e));
					if (d < f.distance || foundNeighbors.size() < ef) {
						NodeDistance neighbor = new NodeDistance(e, d);
						candidates.add(neighbor);
						foundNeighbors.add(neighbor);
						if (foundNeighbors.size() > ef) foundNeighbors.poll();
					}
				}
			}
		}
		return foundNeighbors;
	}
	
	
}
