# AGENTS.md - Autonomous Local-Learning Rule R&D

## Mission

Discover compact, interpretable **local learning rules** that cause a fixed hierarchical neural network to recover the hierarchy exposed by a fixed dataset provider. The selected provider, neural topology, training loop, and benchmark are not targets for optimization. The learning rule is.

The scientific objective is to identify mechanisms that may later transfer to hierarchical representation and invariant learning in vision, language, temporal, and sensorimotor domains.

Success on one provider demonstrates hierarchy recovery for that benchmark, not general concept formation or cross-dataset transfer.

## Benchmark architecture

The benchmark has three immutable layers during Auto-R&D:

1. The selected dataset-provider notebook defines cases, seeds, features, hierarchy labels, and parent maps. The default is `nb/Dataset.ipynb`.
2. `nb/Prepare.ipynb` validates provider output and defines topology, training, numerical guards, and metrics.
3. `nb/Eval.ipynb` loads both layers, imports `nb/Rules.py`, tunes parameters, evaluates panels, and records results.

An operator may select another provider before a benchmark run with `AUTORD_DATASET_NOTEBOOK`. A rule-search agent must not set, change, or condition behavior on that value.

## Non-negotiable file permissions

During Auto-R&D, you MAY:

- Edit `nb/Rules.py` only.
- Copy `nb/Rules.py` into `out/rules/` to create a new, append-only snapshot before mutation.
- Allow the fixed evaluator to create or append its designated results file.
- Append narrative experiment notes only to a file explicitly designated by the operator or evaluator.

You MUST NOT edit:

- the selected dataset-provider notebook, including `nb/Dataset.ipynb`
- `nb/Prepare.ipynb`
- `nb/Eval.ipynb`
- `nb/Rules.ipynb`
- `README.md`
- `src/util/plot_util.py`
- `docs/papers/learning_rules_autord.tex` or its PDF
- dataset cases, splits, seeds, labels, parent maps, or provider selection
- metric, topology, training, or dataset-validation code
- evaluation budgets
- numerical guards or any future locality checks

Do not modify notebooks to improve a score. Do not bypass imports, monkey-patch SciPy/NumPy, inspect hierarchy labels from `nb/Rules.py`, read provider/evaluator globals, read experiment answers from disk, or condition behavior on provider, case, seed, split, or dataset identity.

Provider cells tagged `skip-on-provider-import` are interactive documentation or visualization only. The evaluator must skip them, and their output must not enter fitness calculations or rule state.

## Dataset-provider boundary

The provider contract is documented in `nb/Dataset.ipynb` and `README.md`. Provider output is privileged evaluator input. It may include labels, parent maps, and metadata because fixed metrics need ground truth; none of those fields are available to `nb/Rules.py`.

Treat each provider and its declared development/promotion panels as a separate benchmark. Never compare scores from different providers as though they came from the same fitness distribution. Claims of transfer require independently reported results across providers.

## Fundamental invariance

Neuron numbering has no semantic meaning. Never optimize toward neuron `i` matching hierarchy node `i` directly. Evaluation is permutation invariant.

For correspondence matrix `C`, the evaluator solves an optimal assignment, rewards high assigned correspondence, and penalizes unmatched cross-talk. The same principle applies to topology comparisons after independently aligning neighboring layers.

## Public API and allowed information

Keep this `nb/Rules.py` API unchanged:

- `RuleConfig` and `PARAM_BOUNDS`
- `init_state(n_pre, n_post, rng)`
- `activate(x, w, state, cfg)`
- `update(x, y, w, state, cfg)`
- `complexity_score()`

The runner does not provide a `LocalContext` object. In `activate` and `update`, a candidate may use only information passed through this API:

- `x`: presynaptic activities for the current layer
- `y`: postsynaptic activities for the current layer, when passed to `update`
- `w`: current synaptic weights for the current layer
- `state`: mutable state created independently for the current layer
- `cfg`: scalar hyperparameters declared by the rule

Layer-local traces, neuron statistics, competition state, and homeostatic variables may be stored in `state`. The initialization RNG may be used only to initialize that layer's local state.

A rule must not use:

- dataset hierarchy identities or labels
- dataset parent assignments or provider metadata
- global loss or backpropagated gradients
- activities not passed through the public layer API
- evaluation correspondence matrices or optimal assignments
- provider, preparation, or evaluator globals
- provider, case, split, seed, or dataset identity
- experiment history or answers read from disk

The locality contract is a protocol and code-review boundary; the current Python evaluator is not a security sandbox and does not automatically detect every violation.

## Research cycle

Repeat the following cycle until the experiment budget is exhausted.

### 1. Inspect evidence

Read the selected provider's documented characteristics, the current champion rule, and recent available experiment evidence. Identify a concrete failure mode, for example:

- duplicate neurons or insufficient specialization
- dead neurons
- excessive weight growth
- weak hierarchy specificity
- correct clustering at one layer but poor higher-level abstraction
- unstable performance across cases or seeds
- excessive sensitivity to overlap or variance
- poor topology consistency

Do not inspect individual labels, parent maps, generated samples, or provider internals from `nb/Rules.py`.

### 2. State one hypothesis

Before editing, state a short hypothesis in the working experiment context. Persist it only if a narrative notes file has been explicitly designated. Examples:

- "A sliding postsynaptic threshold may prevent frequent clusters from monopolizing neurons."
- "Anti-Hebbian lateral decorrelation may improve one-to-one prototype specialization."
- "Slower plasticity at higher levels may reduce the moving-target problem."

Prefer one mechanistic change per trial unless combining two previously validated mechanisms.

### 3. Snapshot before mutation

Copy the current `nb/Rules.py` to `out/rules/` with a monotonically increasing trial id or UTC timestamp. Never overwrite a prior snapshot.

### 4. Modify only `nb/Rules.py`

Keep the public API unchanged. Prefer small, interpretable rules. A useful rule is not merely code that scores well; it should correspond to a plausible local mechanism.

### 5. Tune continuous parameters

Do not spend coding iterations hand-tuning scalar constants. Use the provided SciPy differential-evolution path to optimize parameters within the bounds declared by the rule. Do not alter the evaluator's optimization budget.

The checked-in evaluator call uses the smoke-test budget `maxiter=2, popsize=3`; treat its result as preliminary. Promotion-quality optimization requires an operator-designated fixed budget.

### 6. Evaluate across the configured panel

Use the evaluator and selected provider exactly as configured. The provider declares development and promotion cases and seeds. Record the provider name with every result. Visible promotion cases do not constitute a hidden test set.

### 7. Apply rejection criteria

Provider output is validated before training. The evaluator automatically raises on non-finite weights and caps weight-row norms at `5.0`. Reject any candidate that fails those checks.

Also reject a candidate on protocol or review evidence that it:

- produces non-finite activities
- collapses most units to silence
- collapses most activity to one unit
- violates locality
- accesses forbidden state
- changes the public rule API
- is materially more complex without compensating improvement

Do not claim that activity-range, collapse, locality, or API criteria were automatically tested; the current evaluator does not implement dedicated checks for all of them.

### 8. Make the promotion decision

Use the fixed composite score and examine:

- permutation-invariant within-level hierarchy recovery
- assigned-versus-unmatched correspondence dominance
- cross-level specificity
- co-activity-derived parent-child/topology agreement
- performance on promotion-only cases, if the provider declares any
- mean score across the configured panel
- minimum panel score
- across-panel variance

Complexity is penalized. If scores are statistically indistinguishable, prefer the simpler rule. Do not promote on one lucky seed or case.

### 9. Keep or revert

If the promotion criteria are met, keep the new `nb/Rules.py` as champion. Otherwise restore the previous champion byte-for-byte from its snapshot.

### 10. Record scientific interpretation

The evaluator's compact CSV record contains the dataset-provider name, timestamp, fitness, mean/standard-deviation/minimum overall scores, mean hierarchy/specificity/topology scores, and optimized parameters.

In the trial report—and in a narrative notes file only when one has been explicitly designated—also record:

- provider name
- trial id and parent rule id
- hypothesis
- code or mechanism change
- per-level matching scores
- constraint or protocol failures
- complexity measure
- keep/reject decision
- interpretation and next hypothesis

Do not manually edit the evaluator's numeric CSV.

## Search strategy

Use a mixture of:

- mutation of the current champion
- revisiting older Pareto-strong rules
- recombination of mechanisms that succeeded independently
- simplification or ablation of complex champions

Candidate mechanisms include Hebbian covariance terms, Oja-style normalization, BCM-like sliding thresholds, anti-Hebbian decorrelation, soft or hard winner-take-all competition, activity homeostasis, eligibility traces, multiple local timescales, local prediction terms, adaptive plasticity, and layer-dependent timescales generated by a shared rule.

Do not assume these mechanisms are correct. Treat them as hypotheses.

## Scientific guardrails

The default provider intentionally uses nested Gaussian worlds and gives each neural layer the corresponding hierarchy level's cardinality. A rule that succeeds there has demonstrated hierarchical clustering or recovery under substantial structural prior knowledge, not general concept formation. Do not overclaim.

New providers should increase difficulty with anisotropic covariance, unequal priors, nonlinear manifolds, transformation-defined equivalence classes, temporal continuity, active sensorimotor perturbations, over-provisioned neural layers, or independently sourced datasets. Provider changes must be versioned and evaluated as new benchmarks rather than introduced during a rule search.

The final target is a reusable learning principle, not provider-specific code.
