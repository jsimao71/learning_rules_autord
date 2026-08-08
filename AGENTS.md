# AGENTS.md - Autonomous Local-Learning Rule R&D

## Mission

Discover compact, interpretable **local learning rules** that cause a fixed hierarchical neural network to recover the latent hierarchy of fixed synthetic generative worlds. The neural topology and benchmark are not targets for optimization. The learning rule is.

The scientific objective is to identify mechanisms that may later transfer to hierarchical representation and invariant learning in vision, language, temporal, and sensorimotor domains.

Success on this benchmark demonstrates hierarchy recovery in nested Gaussian data, not general concept formation.

## Non-negotiable file permissions

During Auto-R&D, you MAY:

- Edit `nb/Rules.py` only.
- Copy `nb/Rules.py` into `out/rules/` to create a new, append-only snapshot before mutation.
- Allow the fixed evaluator to create or append its designated results file.
- Append narrative experiment notes only to a file explicitly designated by the operator or evaluator.

You MUST NOT edit:

- `nb/Prepare.ipynb`
- `nb/Eval.ipynb`
- `nb/Rules.ipynb`
- `README.md`
- `docs/papers/learning_rules_autord.tex` or its PDF
- metric code
- dataset/generator code
- topology code
- seed selection
- development/promotion world definitions
- evaluation budgets
- numerical guards or any future locality checks

Do not modify notebooks to improve a score. Do not bypass imports, monkey-patch SciPy/NumPy, inspect latent labels from `nb/Rules.py`, read evaluator globals, read experiment answers from disk, or condition behavior on seed/world identity.

## Fundamental invariance

Neuron numbering has no semantic meaning. Never optimize toward neuron `i` matching latent prototype `i` directly. Evaluation is permutation invariant.

For correspondence matrix `C`, the evaluator solves an optimal assignment, rewards high assigned correspondence, and penalizes unmatched cross-talk. The same principle applies to topology comparisons after independently aligning neighboring layers.

## Public API and allowed information

Keep this `nb/Rules.py` API unchanged:

- `RuleConfig` and `PARAM_BOUNDS`
- `init_state(n_pre, n_post, rng)`
- `activate(x, w, state, cfg)`
- `update(x, y, w, state, cfg)`
- `complexity_score()`

The current runner does not provide a `LocalContext` object. In `activate` and `update`, a candidate may use only information passed through this API:

- `x`: presynaptic activities for the current layer
- `y`: postsynaptic activities for the current layer, when passed to `update`
- `w`: current synaptic weights for the current layer
- `state`: mutable state created independently for the current layer
- `cfg`: scalar hyperparameters declared by the rule

Layer-local traces, neuron statistics, competition state, and homeostatic variables may be stored in `state`. The initialization RNG may be used only to initialize that layer's local state.

A rule must not use:

- ground-truth latent identities or labels
- generator parent assignments
- global loss or backpropagated gradients
- activities not passed through the public layer API
- evaluation correspondence matrices or optimal assignments
- evaluator globals
- development or promotion world identity
- experiment history or answers read from disk

The locality contract is a protocol and code-review boundary; the current Python evaluator is not a security sandbox and does not automatically detect every violation.

## Research cycle

Repeat the following cycle until the experiment budget is exhausted.

### 1. Inspect evidence

Read the current champion rule and recent available experiment evidence. Identify a concrete failure mode, for example:

- duplicate neurons or insufficient specialization
- dead neurons
- excessive weight growth
- weak hierarchy specificity
- correct clustering at one layer but poor higher-level abstraction
- unstable performance across seeds or worlds
- excessive sensitivity to overlap or variance
- poor topology consistency

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

Use the evaluator exactly as provided. Development optimization currently uses two world specifications and three seeds. The promotion panel uses those specifications plus one additional specification across five seeds. The specifications are visible and therefore do not constitute a hidden test set.

### 7. Apply rejection criteria

The evaluator automatically raises on non-finite weights and caps weight-row norms at `5.0`. Reject any candidate that fails those checks.

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
- performance on the additional promotion world
- mean score across the configured panel
- minimum panel score
- across-panel variance

Complexity is penalized. If scores are statistically indistinguishable, prefer the simpler rule. Do not promote on one lucky seed or world.

### 9. Keep or revert

If the promotion criteria are met, keep the new `nb/Rules.py` as champion. Otherwise restore the previous champion byte-for-byte from its snapshot.

### 10. Record scientific interpretation

The current evaluator's compact CSV record contains the timestamp, fitness, mean/standard-deviation/minimum overall scores, mean hierarchy/specificity/topology scores, and optimized parameters.

In the trial report—and in a narrative notes file only when one has been explicitly designated—also record:

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

The first benchmark intentionally uses nested Gaussian worlds and gives each neural layer the corresponding latent level's cardinality. A rule that succeeds here has demonstrated hierarchical clustering or recovery under substantial structural prior knowledge, not general concept formation. Do not overclaim.

After a robust rule emerges, future benchmark stages should increase difficulty with anisotropic covariance, unequal cluster priors, nonlinear manifolds, transformation-defined equivalence classes, temporal continuity, active sensorimotor perturbations, and over-provisioned neural layers.

The final target is a reusable learning principle, not benchmark-specific code.
