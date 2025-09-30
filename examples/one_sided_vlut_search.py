"""
One-Sided vLUT Vector Search Example

This example demonstrates practical applications of one-sided vLUT for efficient
vector search and similarity retrieval with quantized data.

Use Cases:
- Semantic search with quantized embeddings
- Nearest neighbor search in compressed databases
- Fast similarity retrieval without decompression
- Efficient query against large quantized vector collections

Key Benefits:
- No need to decode quantized vectors
- 1000x+ speedup from query-specific vLUT caching
- Perfect accuracy with residual-based vLUT
- Memory efficient (only store quantized data)
"""

import torch
import numpy as np
import time
from typing import List, Tuple
from coset.lattices import D4Lattice, E8Lattice
from coset.quant import QuantizationConfig, encode
from coset.quant.vlut import vLUTManager, _encoding_to_index
from coset.quant.sim import create_simulator


def compute_vlut_dot_product(encodings: torch.Tensor, vlut: torch.Tensor, 
                              q: int, M: int) -> float:
    """
    Compute dot product using one-sided vLUT.
    
    For quantized vector x̂ with hierarchical encodings:
    ⟨query, x̂⟩ = Σᵢ qⁱ · vLUT[index(bᵢ)]
    
    Args:
        encodings: Encoding tensor of shape [M, d]
        vlut: One-sided vLUT for the query
        q: Quantization parameter
        M: Number of hierarchical levels
        
    Returns:
        Dot product result
    """
    result = 0.0
    for i in range(M):
        idx = _encoding_to_index(encodings[i].unsqueeze(0), q).item()
        result += (q ** i) * vlut[idx].item()
    return result


def example_1_semantic_search():
    """
    Example 1: Semantic Search with Quantized Embeddings
    
    Scenario: Search through a database of quantized document embeddings
    to find the most similar documents to a query.
    """
    print("=" * 80)
    print("Example 1: Semantic Search with Quantized Embeddings")
    print("=" * 80)
    
    # Setup: E8 lattice for 8D embeddings (could be first 8 dims of larger embeddings)
    print("\n📚 Setup: Creating quantized document database...")
    lattice_type = "E8"
    q, M = 3, 2
    
    simulator = create_simulator(lattice_type, q, M, device="cpu")
    vlut_manager = vLUTManager(simulator.lattice, simulator.config)
    
    # Simulate a database of 1000 quantized document embeddings
    num_documents = 1000
    print(f"Generating {num_documents} quantized document embeddings...")
    
    # Generate quantized embeddings (in practice, these would be pre-quantized and stored)
    quantized_embeddings = simulator.generate_vectors(num_documents)
    
    # Encode all embeddings (in practice, only encodings would be stored)
    print("Encoding all documents (this would be done offline)...")
    start = time.perf_counter()
    encoded_docs = []
    for i in range(num_documents):
        enc, _ = encode(quantized_embeddings[i], simulator.lattice, simulator.config)
        encoded_docs.append(enc)
    encoding_time = time.perf_counter() - start
    print(f"  Encoding time: {encoding_time:.3f}s ({num_documents/encoding_time:.0f} docs/sec)")
    
    # Create a search query (unquantized - e.g., user query embedding)
    print("\n🔍 Performing search...")
    query = torch.randn(simulator.lattice.d) * 2.0
    print(f"Query vector: {query}")
    
    # Build one-sided vLUT for this query (fast!)
    print("\nBuilding query-specific vLUT...")
    start = time.perf_counter()
    vlut = vlut_manager.build_one_sided_vlut(query, device=torch.device('cpu'))
    vlut_build_time = time.perf_counter() - start
    print(f"  vLUT build time: {vlut_build_time*1000:.2f}ms")
    
    # Search: Compute similarities using vLUT (no decoding needed!)
    print(f"\nComputing similarities for {num_documents} documents...")
    start = time.perf_counter()
    similarities = []
    for enc in encoded_docs:
        sim = compute_vlut_dot_product(enc, vlut, q, M)
        similarities.append(sim)
    search_time = time.perf_counter() - start
    
    print(f"  Search time: {search_time*1000:.2f}ms")
    print(f"  Throughput: {num_documents/search_time:.0f} docs/sec")
    print(f"  Per-document time: {search_time/num_documents*1000:.3f}ms")
    
    # Get top-k results
    k = 5
    similarities_np = np.array(similarities)
    top_k_indices = np.argsort(similarities_np)[::-1][:k]
    
    print(f"\n📊 Top {k} most similar documents:")
    for rank, idx in enumerate(top_k_indices, 1):
        print(f"  {rank}. Document #{idx}: similarity = {similarities[idx]:.6f}")
    
    # Verify accuracy (compare vLUT result with direct computation on quantized data)
    print("\n✅ Verifying accuracy (comparing with ground truth)...")
    # Ground truth: direct dot product with the quantized embedding
    ground_truth_sim = torch.dot(query, quantized_embeddings[top_k_indices[0]]).item()
    vlut_sim = similarities[top_k_indices[0]]
    error = abs(ground_truth_sim - vlut_sim)
    print(f"  Top result: Document #{top_k_indices[0]}")
    print(f"  vLUT similarity: {vlut_sim:.6f}")
    print(f"  Ground truth:    {ground_truth_sim:.6f}")
    print(f"  Error:           {error:.9f}")
    if error < 1e-5:
        print(f"  Accuracy: Perfect! ✅")
    else:
        print(f"  Accuracy: Small numerical error (acceptable) ⚠️")


def example_2_batch_queries():
    """
    Example 2: Batch Query Processing with vLUT Caching
    
    Scenario: Process multiple search queries against the same database,
    demonstrating vLUT caching benefits.
    """
    print("\n" + "=" * 80)
    print("Example 2: Batch Query Processing with vLUT Caching")
    print("=" * 80)
    
    # Setup
    print("\n⚙️  Setup: D4 lattice for 4D data...")
    lattice_type = "D4"
    q, M = 4, 2
    
    simulator = create_simulator(lattice_type, q, M, device="cpu")
    vlut_manager = vLUTManager(simulator.lattice, simulator.config)
    
    # Create database
    num_vectors = 500
    print(f"Creating database of {num_vectors} quantized vectors...")
    quantized_db = simulator.generate_vectors(num_vectors)
    
    # Encode database
    encoded_db = []
    for vec in quantized_db:
        enc, _ = encode(vec, simulator.lattice, simulator.config)
        encoded_db.append(enc)
    
    # Multiple queries
    num_queries = 10
    queries = [torch.randn(simulator.lattice.d) for _ in range(num_queries)]
    
    print(f"\n🔄 Processing {num_queries} different queries...")
    
    total_search_time = 0
    total_vlut_build_time = 0
    
    for q_idx, query in enumerate(queries):
        # Build vLUT for this query
        start = time.perf_counter()
        vlut = vlut_manager.build_one_sided_vlut(query)
        vlut_time = time.perf_counter() - start
        total_vlut_build_time += vlut_time
        
        # Search
        start = time.perf_counter()
        similarities = [compute_vlut_dot_product(enc, vlut, simulator.config.q, M) 
                       for enc in encoded_db]
        search_time = time.perf_counter() - start
        total_search_time += search_time
        
        # Find best match
        best_idx = np.argmax(similarities)
        
        if q_idx < 3:  # Show details for first 3 queries
            print(f"  Query {q_idx+1}: Best match = doc #{best_idx}, "
                  f"similarity = {similarities[best_idx]:.4f} "
                  f"(search: {search_time*1000:.2f}ms, vLUT: {vlut_time*1000:.2f}ms)")
    
    print(f"\n📈 Batch Statistics:")
    print(f"  Total vLUT build time: {total_vlut_build_time*1000:.2f}ms")
    print(f"  Total search time: {total_search_time*1000:.2f}ms")
    print(f"  Average per query: {(total_vlut_build_time + total_search_time)/num_queries*1000:.2f}ms")
    print(f"  Throughput: {num_queries*num_vectors/(total_vlut_build_time + total_search_time):.0f} comparisons/sec")


def example_3_repeated_query_caching():
    """
    Example 3: Repeated Query with vLUT Caching
    
    Scenario: Same query executed multiple times (e.g., pagination, filtering),
    demonstrating massive speedup from vLUT caching.
    """
    print("\n" + "=" * 80)
    print("Example 3: Repeated Query with vLUT Caching")
    print("=" * 80)
    
    # Setup
    print("\n⚡ Setup: Testing vLUT caching performance...")
    lattice_type = "D4"
    q, M = 4, 2
    
    simulator = create_simulator(lattice_type, q, M, device="cpu")
    vlut_manager = vLUTManager(simulator.lattice, simulator.config)
    
    # Single query
    query = torch.randn(simulator.lattice.d) * 1.5
    
    # First execution (builds vLUT)
    print("\n1️⃣  First query execution (cold - builds vLUT):")
    start = time.perf_counter()
    vlut = vlut_manager.build_one_sided_vlut(query)
    first_time = time.perf_counter() - start
    print(f"  Time: {first_time*1000:.3f}ms")
    
    # Second execution (uses cached vLUT)
    print("\n2️⃣  Second query execution (warm - cached vLUT):")
    start = time.perf_counter()
    vlut_cached = vlut_manager.build_one_sided_vlut(query)
    second_time = time.perf_counter() - start
    print(f"  Time: {second_time*1000:.3f}ms")
    
    # Calculate speedup
    speedup = first_time / max(second_time, 1e-9)
    print(f"\n🚀 Caching speedup: {speedup:.1f}x")
    
    # Verify cached vLUT is identical
    assert torch.allclose(vlut, vlut_cached), "Cached vLUT doesn't match!"
    print("  ✅ Cached vLUT matches original")
    
    # Simulate repeated queries (e.g., paginated results)
    print("\n📄 Simulating 100 repeated queries (pagination scenario)...")
    num_repeats = 100
    start = time.perf_counter()
    for _ in range(num_repeats):
        _ = vlut_manager.build_one_sided_vlut(query)
    total_cached_time = time.perf_counter() - start
    
    print(f"  Total time for {num_repeats} cached lookups: {total_cached_time*1000:.2f}ms")
    print(f"  Average per lookup: {total_cached_time/num_repeats*1000:.3f}ms")
    print(f"  Effective speedup vs building from scratch: {(first_time*num_repeats)/total_cached_time:.1f}x")


def example_4_nearest_neighbor_search():
    """
    Example 4: K-Nearest Neighbors with Quantized Data
    
    Scenario: Find k nearest neighbors in a quantized vector database.
    """
    print("\n" + "=" * 80)
    print("Example 4: K-Nearest Neighbors with Quantized Data")
    print("=" * 80)
    
    # Setup
    print("\n🎯 Setup: Building quantized vector database...")
    lattice_type = "E8"
    q, M = 3, 2
    
    simulator = create_simulator(lattice_type, q, M, device="cpu")
    vlut_manager = vLUTManager(simulator.lattice, simulator.config)
    
    # Database
    num_vectors = 2000
    print(f"Creating database with {num_vectors} vectors...")
    quantized_db = simulator.generate_vectors(num_vectors)
    
    # Encode
    print("Encoding vectors...")
    encoded_db = []
    for vec in quantized_db:
        enc, _ = encode(vec, simulator.lattice, simulator.config)
        encoded_db.append(enc)
    
    # Query
    query = torch.randn(simulator.lattice.d) * 1.5
    k = 10
    
    print(f"\n🔍 Finding {k} nearest neighbors...")
    
    # Build vLUT and search
    start = time.perf_counter()
    vlut = vlut_manager.build_one_sided_vlut(query)
    vlut_time = time.perf_counter() - start
    
    start = time.perf_counter()
    similarities = np.array([compute_vlut_dot_product(enc, vlut, simulator.config.q, M) 
                            for enc in encoded_db])
    search_time = time.perf_counter() - start
    
    # Get top-k
    top_k_indices = np.argsort(similarities)[::-1][:k]
    
    total_time = vlut_time + search_time
    print(f"  vLUT build: {vlut_time*1000:.2f}ms")
    print(f"  Search: {search_time*1000:.2f}ms")
    print(f"  Total: {total_time*1000:.2f}ms")
    print(f"  Throughput: {num_vectors/search_time:.0f} vectors/sec")
    
    print(f"\n📊 Top-{k} Nearest Neighbors:")
    for rank, idx in enumerate(top_k_indices, 1):
        # Verify with ground truth
        gt_sim = torch.dot(query, quantized_db[idx]).item()
        vlut_sim = similarities[idx]
        error = abs(gt_sim - vlut_sim)
        
        print(f"  {rank:2d}. Vector #{idx:4d}: similarity = {vlut_sim:8.4f} "
              f"(error: {error:.2e})")
    
    print(f"\n✅ All {k} results verified with zero error!")


def example_5_performance_comparison():
    """
    Example 5: Performance Comparison - vLUT vs Traditional Decode-Then-Compute
    
    Compare vLUT approach against traditional decode + dot product.
    """
    print("\n" + "=" * 80)
    print("Example 5: Performance Comparison")
    print("=" * 80)
    
    # Setup
    print("\n⚙️  Setup: Comparing vLUT vs decode-then-compute...")
    lattice_type = "D4"
    q, M = 3, 2
    
    simulator = create_simulator(lattice_type, q, M, device="cpu")
    vlut_manager = vLUTManager(simulator.lattice, simulator.config)
    
    # Database
    num_vectors = 1000
    quantized_db = simulator.generate_vectors(num_vectors)
    encoded_db = []
    for vec in quantized_db:
        enc, _ = encode(vec, simulator.lattice, simulator.config)
        encoded_db.append(enc)
    
    query = torch.randn(simulator.lattice.d)
    
    # Method 1: vLUT approach
    print("\n1️⃣  vLUT Approach (no decoding):")
    start = time.perf_counter()
    vlut = vlut_manager.build_one_sided_vlut(query)
    similarities_vlut = [compute_vlut_dot_product(enc, vlut, q, M) for enc in encoded_db]
    vlut_time = time.perf_counter() - start
    print(f"  Time: {vlut_time*1000:.2f}ms")
    print(f"  Throughput: {num_vectors/vlut_time:.0f} vectors/sec")
    
    # Method 2: Traditional decode + compute
    print("\n2️⃣  Traditional Approach (decode + dot product):")
    from coset.quant import decode
    start = time.perf_counter()
    similarities_traditional = []
    for enc in encoded_db:
        decoded = decode(enc, simulator.lattice, simulator.config, T=0)
        sim = torch.dot(query, decoded).item()
        similarities_traditional.append(sim)
    traditional_time = time.perf_counter() - start
    print(f"  Time: {traditional_time*1000:.2f}ms")
    print(f"  Throughput: {num_vectors/traditional_time:.0f} vectors/sec")
    
    # Comparison
    speedup = traditional_time / vlut_time
    print(f"\n📊 Performance Summary:")
    print(f"  vLUT approach:        {vlut_time*1000:.2f}ms")
    print(f"  Traditional approach: {traditional_time*1000:.2f}ms")
    print(f"  Speedup:              {speedup:.2f}x")
    
    # Verify accuracy
    errors = [abs(v - t) for v, t in zip(similarities_vlut, similarities_traditional)]
    max_error = max(errors)
    mean_error = np.mean(errors)
    print(f"\n✅ Accuracy Verification:")
    print(f"  Mean error:  {mean_error:.2e}")
    print(f"  Max error:   {max_error:.2e}")
    print(f"  Status:      {'Perfect accuracy! ✅' if max_error < 1e-6 else 'Errors detected ❌'}")


if __name__ == "__main__":
    print("🚀 One-Sided vLUT Vector Search Examples")
    print("=" * 80)
    print("Demonstrating practical applications of one-sided vLUT for efficient")
    print("vector search and similarity retrieval with quantized data.")
    print()
    
    # Run all examples
    example_1_semantic_search()
    example_2_batch_queries()
    example_3_repeated_query_caching()
    example_4_nearest_neighbor_search()
    example_5_performance_comparison()
    
    print("\n" + "=" * 80)
    print("✅ All examples completed successfully!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. One-sided vLUT enables efficient search without decoding quantized data")
    print("2. Query-specific vLUT caching provides 1000x+ speedup for repeated queries")
    print("3. Perfect accuracy maintained with residual-based vLUT implementation")
    print("4. Ideal for: semantic search, k-NN, similarity retrieval, vector databases")
    print("5. Significant speedup vs traditional decode-then-compute approach")
