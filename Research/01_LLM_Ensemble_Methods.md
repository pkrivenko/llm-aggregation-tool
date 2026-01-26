# LLM Ensemble and Aggregation Methods: A Comprehensive Research Report

**Date:** January 2026
**Purpose:** Research overview for LLM aggregation pipeline development

---

## Executive Summary

This report surveys state-of-the-art techniques for combining outputs from multiple LLM calls to improve accuracy and reliability. These methods are particularly relevant for multi-stage pipelines (Doers, Judges, Final Judges) that aggregate LLM responses for improved decision-making.

Key findings:
- **Self-consistency** remains the foundational technique, offering 6-18% accuracy gains with simple majority voting
- **Universal Self-Consistency (USC)** extends this to free-form answers using LLM-based selection
- **Multi-agent debate** shows promise but current implementations often fail to outperform simpler baselines
- **LLM cascading** can reduce costs by 70-98% while maintaining accuracy
- **Test-time compute scaling** (as in o1/R1 models) represents the frontier of inference-time improvements
- **Best-of-N with reward models** provides principled selection but requires careful handling of reward hacking

---

## 1. Self-Consistency Decoding

### Overview

Self-consistency, introduced by [Wang et al. (2022)](https://arxiv.org/abs/2203.11171), is a decoding strategy that replaces greedy decoding in chain-of-thought prompting. The core intuition is that complex reasoning problems typically admit multiple different reasoning paths leading to the same correct answer.

### Algorithm

1. **Prompt with chain-of-thought examples** - Provide few-shot demonstrations showing reasoning steps
2. **Sample diverse reasoning paths** - Generate multiple completions using temperature sampling (typically T=0.7-1.0)
3. **Marginalize over paths** - Extract final answers and select the most frequent one via majority voting

### Key Results

Self-consistency achieves striking improvements across reasoning benchmarks:
- **GSM8K:** +17.9% over standard CoT
- **SVAMP:** +11.0%
- **AQuA:** +12.2%
- **StrategyQA:** +6.4%
- **ARC-challenge:** +3.9%

### Implementation Considerations

**Temperature Selection:**
- Higher temperatures (0.7-1.0) produce more diverse reasoning paths
- Too high (>1.2) introduces incoherent responses
- Too low (<0.5) produces near-identical paths, reducing ensemble benefit

**Number of Samples:**
- Diminishing returns typically after 10-40 samples
- More samples help difficult problems more than easy ones
- Cost scales linearly with sample count

**Answer Extraction:**
- Requires structured answer formats for reliable extraction
- Works best with numerical answers, multiple choice, or short phrases
- Struggles with free-form text outputs

### Practical Recommendations

- Use 5-20 samples for interactive applications, 20-40 for batch processing
- Set temperature between 0.7-0.9 for good diversity-coherence balance
- Implement robust answer parsing (regex, structured output)
- Consider weighted voting based on answer confidence when available

---

## 2. Universal Self-Consistency (USC)

### Motivation

Standard self-consistency requires extracting comparable answers for voting, which fails for:
- Open-ended generation tasks
- Long-form summaries
- Creative writing
- Responses with no single "answer"

[Universal Self-Consistency](https://arxiv.org/abs/2311.17311) addresses this limitation by using an LLM to select the most consistent response.

### Algorithm

1. **Sample multiple responses** - Generate N candidate responses to the same prompt
2. **Concatenate responses** - Combine all candidates into a single context
3. **LLM selection** - Prompt an LLM: "Given these N responses, which one is most consistent with the others?"

### Benchmarks

USC was evaluated across diverse tasks:
- **Mathematical reasoning:** Matches standard self-consistency without requiring answer format constraints
- **Code generation:** Matches execution-based voting performance without running code
- **Long-context summarization:** Consistent improvements over baselines
- **TruthfulQA:** Achieves highest truthfulness and informativeness scores

### Limitations

- **Position bias:** LLMs may favor responses in certain positions (first, last)
- **Long-context handling:** Performance degrades with many long responses
- **Performance saturation:** Some tasks (e.g., GSM8K) show diminishing returns beyond ~10 samples
- **Computational cost:** Requires an additional LLM call for selection

### Practical Recommendations

- Randomize response order to mitigate position bias
- Use concise response formats when possible
- Consider iterative selection for large candidate sets
- Pair with a capable judge model (GPT-4 class or better)

---

## 3. Multi-Agent Debate

### Overview

Multi-agent debate (MAD) frameworks have multiple LLM agents propose answers, critique each other's reasoning, and iteratively refine toward consensus. This "society of minds" approach aims to reduce hallucinations and improve reasoning.

### Key Frameworks

**1. Basic MAD ([Du et al., 2023](https://composable-models.github.io/llm_debate/))**
- Multiple agents generate initial responses
- Agents see others' responses and provide critiques
- Iterative refinement over 2-3 rounds
- Final answer via voting or designated judge

**2. Adaptive Heterogeneous MAD (A-HMAD)**
- Uses diverse specialized agents instead of homogeneous models
- Dynamic debate with adaptive stopping
- Addresses limitations of simple majority voting

**3. FREE-MAD ([Consensus-Free Multi-Agent Debate](https://arxiv.org/pdf/2509.11035))**
- Does not require multi-round consensus
- Evaluates entire debate trajectory, not just final round
- More efficient than iterative approaches

**4. [ReConcile](https://arxiv.org/abs/2309.13007) (Chen et al., 2024)**
- Round-table conference among diverse LLM agents
- Confidence-weighted voting mechanism
- Discussion prompts include grouped answers, confidence scores, and demonstrations
- Up to 11.4% improvement over baselines; outperforms GPT-4 on three datasets

### Critical Evaluation

Recent analysis reveals limitations of current MAD implementations:
- Current MAD methods often **fail to consistently outperform single-agent strategies**
- Many cases show **majority-driven convergence to incorrect answers**
- Performance can **degrade with more rounds** in some scenarios
- Computational cost scales with agents x rounds

However, debate shows clear benefits when:
- All models initially give wrong answers, then correct through discussion
- Problems require diverse perspectives or domain expertise
- Verification of factual claims is important

### Practical Recommendations

- Start with 3 agents, 2 rounds as baseline
- Use heterogeneous models when possible (e.g., GPT-4 + Claude + Gemini)
- Implement early stopping when consensus is reached
- Consider FREE-MAD for efficiency when iterative refinement isn't crucial
- Track debate trajectory, not just final answers

---

## 4. Mixture of Experts (MoE) Approaches

### Architecture Overview

[Mixture of Experts](https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/) is an architecture where specialized sub-networks (experts) are selectively activated based on input. A gating/routing network learns to direct inputs to appropriate experts.

### Key Components

**Experts:**
- Not domain-specialized (e.g., "Biology expert") but learned specializations
- Each expert is a subnetwork within the larger model
- Only a subset (typically 2 of 8) activated per token
- Enables massive model scaling without proportional compute increase

**Router/Gating Network:**
- Lightweight network that selects experts per token
- Trained jointly with experts via load-balancing loss
- Can use top-k selection or soft mixture

### Notable Implementations

**Mixtral 8x7B:**
- 8 experts per MoE layer, 2 activated per token
- Experts show specialization patterns by domain and syntax
- Competitive with much larger dense models at lower inference cost

**GPT-4 (rumored):**
- Reportedly uses MoE architecture
- Enables massive capability with manageable inference costs

### MoE for Ensemble Routing

Beyond internal MoE architectures, the concept applies to routing between separate models:

**LLM-based Routing:**
- Use an LLM to select which expert model handles each query
- Enables interpretable routing decisions
- Particularly useful for multi-domain applications

**Task-Specific Routing:**
- Route code questions to CodeLlama, math to specialized models
- Reduces cost by using smaller specialists when sufficient
- Requires accurate task classification

### Practical Recommendations

- For pipeline design: consider routing to specialized models by task type
- Implement confidence-based fallback to larger models
- Track which experts/models are most effective per task category
- Balance load across experts to prevent bottlenecks

---

## 5. LLM Cascading

### Overview

[LLM cascading](https://arxiv.org/abs/2405.15842) optimizes cost by using smaller, cheaper models for simple queries and escalating to larger models only when necessary. The key question: when does a problem truly require the full power of a large model?

### Architecture

```
Query -> Small Model -> [Confident?] -> Yes -> Return
                            |
                            No
                            v
                    Medium Model -> [Confident?] -> Yes -> Return
                            |
                            No
                            v
                    Large Model -> Return
```

### Key Research Results

**FrugalGPT:**
- Learns to route queries through model combinations
- Matches GPT-4 accuracy with **up to 98% cost savings**
- Adaptive routing based on query complexity

**[Cascade Routing](https://arxiv.org/html/2410.10347v1) (de Koninck et al., 2024):**
- Unified framework integrating routing and cascading
- Theoretically optimal strategy derived
- **Up to 14% improvement** over existing strategies

**Model Cascading for Code:**
- Reduces costs by **average 26%, up to 70%** in best cases
- Self-testing for verification at each stage
- Maintains or improves accuracy

**[Speculative Cascades](https://research.google/blog/speculative-cascades-a-hybrid-approach-for-smarter-faster-llm-inference/):**
- Combines cascading with speculative decoding
- Better output quality at lower cost than either alone
- Sometimes defers to smaller model for efficiency

### Deferral Mechanisms

**Confidence-Based:**
- Small model outputs confidence score
- Defer if below threshold
- Challenge: LLM confidence calibration is often poor

**Self-Verification:**
- Model attempts to verify its own answer
- Defer if verification fails
- More reliable than raw confidence

**Learned Routing:**
- Train classifier on (query, model, success) data
- Predict which model will succeed
- Requires training data collection

### Practical Recommendations

- Start with two-tier cascade (small + large)
- Use task-specific confidence thresholds
- Implement self-verification for critical applications
- Track routing decisions to tune thresholds
- Consider latency impact of cascade evaluation

---

## 6. Voting and Consensus Methods

### Basic Methods

**Majority Voting:**
- Most common aggregation method
- Each response gets equal weight
- Select answer with highest count
- Simple but ignores confidence/quality signals

**Weighted Voting:**
- Assign weights based on model capability, confidence, or past performance
- Better utilizes heterogeneous model ensembles
- Requires calibration of weights

**Plurality vs. Unanimous:**
- Plurality: most common answer wins
- Unanimous: require all models to agree (high precision, low recall)
- Supermajority: require k of N agreement

### Advanced Methods

**[Confidence-Weighted Majority Vote (CISC)](https://arxiv.org/pdf/2502.06233):**
- Extract self-assessed confidence for each response
- Normalize with softmax and temperature parameter
- Weight votes by normalized confidence
- Shown to improve over standard majority voting

**[Inverse-Entropy Weighted Voting](https://arxiv.org/html/2511.02309):**
- Weight by inverse of response entropy (higher confidence = higher weight)
- Achieves optimality in 97% of test cases
- Particularly effective for sequential reasoning chains

**[Optimal Weight and Inverse Surprising Popularity](https://arxiv.org/html/2510.01499v1):**
- Leverage first-order (direct answers) and second-order (predictions about others) information
- Provably mitigate limitations of majority voting
- More reliable for heterogeneous agent ensembles

**[Hybrid Majority Reward (HMR) and Weighted Reward-Frequency (WRF)](https://arxiv.org/html/2508.01773):**
- Combine voting frequency with reward model scores
- WRF: `score = w * normalized_reward + (1-w) * frequency`
- Balances consensus with quality signals

### Challenges

- **Dispersed answers:** Majority voting fails when no clear majority
- **Confident errors:** Models may consistently agree on wrong answers
- **Position bias:** Evaluation order can influence results
- **Calibration:** Confidence scores often poorly calibrated

### Practical Recommendations

- Default to majority voting as baseline
- Add confidence weighting when models provide calibrated confidence
- For heterogeneous ensembles, use model-specific weights
- Implement tie-breaking rules (e.g., defer to most capable model)
- Track agreement rates as quality signal

---

## 7. Speculative Execution / Draft-then-Verify

### Overview

[Speculative decoding](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/) accelerates LLM inference by having a small draft model propose tokens that a large target model verifies in parallel.

### How It Works

1. **Draft Phase:** Small model generates K candidate tokens quickly
2. **Verify Phase:** Large model evaluates all K tokens in single forward pass
3. **Accept/Reject:** Accept longest prefix matching large model's distribution
4. **Continue:** Generate one additional token, repeat

### Key Insight

- LLM inference is **memory-bound** - GPUs have excess compute capacity
- Many tokens are **easy to predict** - small model often sufficient
- Verification is **parallelizable** - check K tokens as fast as 1

### Performance

- **2-3x speedup** without any accuracy loss
- Higher acceptance rates = more speedup
- Works best when small model aligns well with large model

### [Self-Speculative Decoding](https://aclanthology.org/2024.acl-long.607/) (ACL 2024)

- Uses the same model for both draft and verify
- Draft by skipping intermediate layers
- No additional model required, no training needed
- **Up to 1.99x speedup** on LLaMA-2

### Advanced Techniques

**EAGLE-3:**
- Lightweight prediction head attached to target model
- Eliminates need for separate draft model
- Higher acceptance rates and throughput

**CTC-based Draft Models:**
- Train draft model with CTC loss for sequential reasonableness
- Higher acceptance rates than standard draft models

### Relevance to Aggregation Pipelines

While speculative decoding is primarily an inference optimization, the draft-verify pattern has broader applications:

- **Draft responses with fast model, verify with capable model**
- **Generate multiple drafts, select best via verification**
- **Use verification as quality filter for ensemble**

### Practical Recommendations

- Use speculative decoding for latency-critical single-model inference
- Apply draft-verify pattern for quality-critical aggregation
- Consider self-verification for cost-effective filtering
- Leverage frameworks with built-in support (vLLM, TensorRT-LLM)

---

## 8. Best-of-N Sampling with Reward Models

### Overview

[Best-of-N (BoN) sampling](https://arxiv.org/abs/2502.12668) generates N candidate responses and selects the best using a reward model. This is a foundational technique for test-time scaling.

### Algorithm

1. Generate N responses with temperature sampling
2. Score each response with reward model
3. Select response with highest reward score

### Challenges

**[Reward Hacking](https://icml.cc/virtual/2024/37706):**
- Reward models are imperfect proxies for true quality
- Optimizing reward score can degrade actual quality
- Becomes worse with larger N

**Reward Model Limitations:**
- Task-specific, often don't generalize
- Computationally expensive (similar size to generator)
- Require training data aligned with evaluation criteria

### Solutions

**[Regularized Best-of-N (RBoN)](https://arxiv.org/abs/2404.01054):**
- Add proximity term to prevent reward hacking
- RBoN_KL: KL divergence from reference
- RBoN_WD: Wasserstein distance regularization

**[MBR-BoN](https://aclanthology.org/2025.naacl-long.472.pdf) (Minimum Bayes Risk):**
- Incorporate MBR objective as proximity regularizer
- Outperforms both standard BoN and MBR decoding
- Balances reward optimization with consistency

**[Pairwise Reward Model](https://medium.com/@techsachin/pairwise-rm-test-time-scaling-of-llm-via-best-of-n-sampling-with-knockout-tournament-0d36287f95f5):**
- Instead of absolute scores, compare pairs
- Knockout tournament structure
- **+6.7% on MATH-500, +3.9% on Olympiad Bench**

**[Self-Certainty Selection](https://arxiv.org/html/2502.18581v1):**
- Use LLM's own token probabilities as reward signal
- No external reward model needed
- Efficient and scalable

**[SWIFT](https://arxiv.org/abs/2505.12225):**
- Learn reward from LLM hidden states
- **+12.7% over EurusRM-7B on MATH**
- Uses <0.005% parameters of standard reward models

### Practical Recommendations

- Start with N=4-8 for cost-quality balance
- Use regularization to prevent reward hacking
- Consider pairwise comparison for more robust selection
- Self-certainty is a good baseline without external models
- Track reward vs. actual quality correlation

---

## 9. LLM-as-Judge Techniques

### Overview

[LLM-as-a-Judge](https://www.evidentlyai.com/llm-guide/llm-as-a-judge) uses a capable LLM to evaluate outputs from other models. This is central to Doer-Judge architectures.

### Approaches

**Direct Scoring:**
- LLM assigns numerical score (1-10, 1-100)
- Can use custom rubrics
- Subject to scale interpretation variance

**Pairwise Comparison:**
- LLM chooses better of two responses
- More reliable than absolute scoring
- Scales as O(N^2) for N candidates

**Reference-Based:**
- Compare against gold standard or reference answer
- More objective when reference available
- Limited to tasks with clear correct answers

### [Advanced Techniques](https://arxiv.org/html/2412.05579v2)

**Critique-out-Loud (CLoud):**
- Generate natural language critique first
- Then score based on critique
- **Significantly outperforms** traditional reward models on reasoning tasks

**Synthetic Critiques:**
- Generate critiques synthetically for training
- Cost-effective and scalable
- Improves reward model quality

**Self-Rewarding LLMs:**
- Model scores its own outputs
- Enables iterative self-improvement
- Risk of self-reinforcing errors

### [Known Biases](https://alopatenko.github.io/LLMEvaluation/)

**Position Bias:** Preference for responses in certain positions
**Length Bias:** Preference for longer (or shorter) responses
**Self-Preference:** Models prefer their own outputs
**Verbosity Bias:** Conflating detail with quality

### Mitigations

- **Multiple evaluations** with voting/averaging
- **Randomize presentation order**
- **Cross-examination** to detect inconsistencies
- **Calibration** against human judgments

### Practical Recommendations

- Use pairwise comparison for ranking tasks
- Implement position randomization
- Combine multiple judge calls with voting
- Validate against human labels periodically
- Use most capable available model as judge

---

## 10. Test-Time Compute Scaling

### The New Paradigm

[Test-time compute scaling](https://arxiv.org/abs/2408.03314) represents a paradigm shift: instead of only scaling training compute, invest compute at inference time for better results.

### Key Research

**[Scaling LLM Test-Time Compute](https://openreview.net/forum?id=4FWAwZtd2n) (August 2024):**
- "Enabling LLMs to improve their outputs by using more test-time computation is a critical step towards building generally self-improving agents"
- Test-time scaling can be more effective than scaling model parameters
- Implies new tradeoffs between training and inference compute

**OpenAI o1 and o3:**
- Generate long internal chains of thought before answering
- RL-trained to perform complex reasoning
- Performance improves with more test-time compute

**[DeepSeek R1](https://magazine.sebastianraschka.com/p/state-of-llms-2025) (January 2025):**
- Demonstrated reasoning behavior emerges from RL
- Open replication of o1-style capabilities
- Significant impact on the field

**[s1: Simple Test-Time Scaling](https://arxiv.org/pdf/2501.19393):**
- SFT on only 1,000 examples builds competitive reasoning model
- "Budget forcing" reproduces OpenAI's test-time scaling curves
- Reasoning ability present from pretraining, fine-tuning activates it

### Techniques

**Chain-of-Thought:**
- Generate intermediate reasoning steps
- [Foundational technique](https://arxiv.org/abs/2201.11903) for reasoning improvement

**[Tree of Thoughts](https://proceedings.neurips.cc/paper_files/paper/2023/file/271db9922b8d1f4dd7aaef84ed5ac703-Paper-Conference.pdf):**
- Explore multiple reasoning branches
- Search algorithms (BFS, DFS) for path selection
- Enables backtracking from dead ends

**[Graph of Thoughts](https://arxiv.org/html/2401.14295v5):**
- Arbitrary reasoning dependencies
- Aggregation of multiple thoughts
- Dynamic programming-like subproblem decomposition

**[Atom of Thoughts](https://arxiv.org/pdf/2502.12018):**
- Decompose into atomic reasoning units
- Integrates with tree search and reflection
- Consistently outperforms baselines as compute increases

### 2026 Outlook

- More focus on inference-time scaling expected
- Trade-off between latency, cost, and accuracy
- Extreme inference scaling valuable when accuracy matters most
- Performance improvements from tooling, not just models

### Practical Recommendations

- Implement chain-of-thought prompting as baseline
- Use tree/graph structures for complex multi-step problems
- Consider compute budget vs. accuracy requirements
- Track reasoning quality, not just final answers
- Explore "thinking" tokens for difficult queries

---

## Summary Comparison Table

| Method | Accuracy Gain | Cost Impact | Latency Impact | Best Use Case |
|--------|--------------|-------------|----------------|---------------|
| **Self-Consistency** | +6-18% | 5-40x generation | ~linear with samples | Structured reasoning |
| **Universal Self-Consistency** | Similar to SC | +1 LLM call | Moderate | Free-form outputs |
| **Multi-Agent Debate** | Variable (0-11%) | 3-9x+ generation | High (multiple rounds) | Fact verification |
| **MoE Routing** | Task-dependent | Can reduce | Can reduce | Multi-domain tasks |
| **LLM Cascading** | Maintained | -26 to -98% | Variable | Cost optimization |
| **Majority Voting** | Baseline | N generations | Parallel | Simple aggregation |
| **Weighted Voting** | +2-5% over MV | Same as MV | Same as MV | Heterogeneous ensembles |
| **Best-of-N + RM** | +3-13% | N gen + N scores | Moderate | Quality selection |
| **LLM-as-Judge** | N/A (evaluation) | 1 judge call | Low | Quality filtering |
| **Test-Time Compute** | Up to 2x+ | Scales with compute | High | Complex reasoning |
| **Speculative Decoding** | None (speed) | Same | 2-3x faster | Latency optimization |

---

## Recommendations for Multi-Stage Pipeline Design

Based on this research, here are recommendations for a Doers-Judges-Final Judges architecture:

### Doer Stage
1. **Use diverse models** - Heterogeneous ensembles perform better
2. **Sample with temperature** - 0.7-0.9 for good diversity
3. **Generate 3-5 responses** - Diminishing returns beyond this
4. **Implement chain-of-thought** - Improves reasoning quality

### Judge Stage
1. **Use pairwise comparison** - More reliable than absolute scoring
2. **Randomize presentation order** - Mitigates position bias
3. **Use capable judge model** - GPT-4 class or better
4. **Request critiques** - CLoud pattern improves evaluation

### Final Judge / Aggregation
1. **Confidence-weighted voting** - Better than simple majority
2. **Consider USC** - For free-form final answers
3. **Implement early stopping** - When strong consensus emerges
4. **Track uncertainty** - High disagreement = uncertain answer

### Cost Optimization
1. **Cascade where possible** - Route simple queries to smaller models
2. **Budget allocation** - Spend more compute on difficult questions
3. **Cache intermediate results** - Avoid redundant computation

---

## References

Key papers referenced in this report:

1. Wang et al. (2022). [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)
2. Chen et al. (2024). [Universal Self-Consistency for Large Language Models](https://arxiv.org/abs/2311.17311)
3. Du et al. (2023). [Improving Factuality and Reasoning through Multiagent Debate](https://composable-models.github.io/llm_debate/)
4. Chen et al. (2024). [ReConcile: Round-Table Conference Improves Reasoning](https://arxiv.org/abs/2309.13007)
5. Yao et al. (2023). [Tree of Thoughts: Deliberate Problem Solving](https://proceedings.neurips.cc/paper_files/paper/2023/file/271db9922b8d1f4dd7aaef84ed5ac703-Paper-Conference.pdf)
6. Besta et al. (2024). [Demystifying Chains, Trees, and Graphs of Thoughts](https://arxiv.org/html/2401.14295v5)
7. Zhang et al. (2024). [Draft & Verify: Self-Speculative Decoding](https://aclanthology.org/2024.acl-long.607/)
8. de Koninck et al. (2024). [A Unified Approach to Routing and Cascading](https://arxiv.org/html/2410.10347v1)
9. Snell et al. (2024). [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314)
10. Yang et al. (2025). [Ensemble Large Language Models: A Survey](https://www.mdpi.com/2078-2489/16/8/688)

---

*Report generated for LLM Aggregation Tool research. Last updated: January 2026.*
