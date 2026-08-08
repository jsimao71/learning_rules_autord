# Learning Rules Auto-R&D for Hierarchy Recovery

This project is a controlled research harness for discovering **local neural learning rules** that recover hierarchical representations without backpropagation.

The harness separates three concerns:

1. A **dataset provider** supplies features, hierarchy labels, and parent-child structure.
2. A fixed **training and metric layer** trains candidate local rules and measures hierarchy recovery.
3. An **evaluator** selects a provider, tunes continuous rule parameters, and compares candidates across a fixed panel.

This separation lets later benchmarks use other generated or open-source datasets without changing the learning-rule API, training loop, or metrics. Candidate rules are scored with fixed, permutation-invariant hierarchy-recovery metrics. Locality and the broader scientific protocol are enforced by the agent instructions and code review; the current evaluator is not a security sandbox.

## Core hypothesis

A useful local plasticity rule should allow successive neural layers to recover increasingly coarse latent structure using only locally available variables. Neuron identities are arbitrary, so recovery is evaluated **up to permutation**. A good layer has high matched correspondence and low unmatched cross-talk after neurons are optimally assigned to hierarchy nodes.

The default provider uses nested Gaussian hierarchies because they provide exact ground truth. Later providers can test richer distributions and invariances, including anisotropic clusters, unequal priors, nonlinear manifolds, transformation-defined equivalence classes, temporal continuity, sensorimotor trajectories, or suitable open-source datasets with an explicit hierarchy.

## Project layout

- `nb/Dataset.ipynb` - default **dataset provider**. Defines the nested-Gaussian cases, development/promotion panels, seeds, and data-generation function.
- `nb/Prepare.ipynb` - **immutable training and metric code**. Validates provider output and defines the fixed topology, training loop, reproducibility behavior, and hierarchy-recovery metrics. It contains no dataset generation.
- `nb/Rules.ipynb` - human-readable notebook form of the initial learning rule and its explanation. The coding agent does not edit this notebook directly.
- `nb/Rules.py` - **the only learning-rule source the coding agent may modify** during the main search loop.
- `nb/Eval.ipynb` - loads the selected dataset provider and `nb/Prepare.ipynb`, imports `nb/Rules.py`, materializes datasets, tunes rule hyperparameters, evaluates configured panels, plots cross-level correspondence scores, and records compact trial results.
- `src/util/plot_util.py` - reusable NumPy/Matplotlib helpers for PCA projections, variance spectra, feature correlations, and hierarchy-tree plots.
- `AGENTS.md` - autonomous experiment protocol for the coding agent.
- `docs/papers/learning_rules_autord.tex`, `docs/papers/learning_rules_autord.pdf` - research paper draft describing the method and motivation.
- `sbin/run_eval.sh`, `sbin/run_eval.bat` - launch `nb/Eval.ipynb` from the correct working directory, optionally with another dataset-provider notebook.
- `sbin/ipynb_to_py.sh`, `sbin/ipynb_to_py.bat` - notebook-to-Python utilities. Their default input is `nb/Rules.ipynb` relative to the repository root.
- `sbin/py_to_ipynb.sh`, `sbin/py_to_ipynb.bat` - Python-to-notebook utilities. Their defaults refresh `nb/Rules.ipynb` from `nb/Rules.py`.
- `requirements.txt` - Python packages required by the notebooks and conversion utilities.
- `out/rules/` - append-only snapshots created by the agent before rule mutations.
- `nb/experiments.csv` - compact numeric results created or extended by `nb/Eval.ipynb`.

## Dataset-provider contract

`nb/Eval.ipynb` loads `nb/Dataset.ipynb` by default. A provider notebook must export:

- `DATASET_PROVIDER_NAME`
- `DEV_CASES` and `PROMOTION_CASES`
- `DEV_SEEDS` and `PROMOTION_SEEDS`
- `case_name(case)`
- `load_dataset(case, seed)`

`load_dataset` returns a mapping containing:

- `name` - unique human-readable dataset instance name
- `x` - finite floating-point features with shape `[samples, features]`
- `labels_by_level` - one integer label vector per hierarchy level, ordered leaf/fine to top/coarse
- `k_levels` - node cardinality at each level
- `parent_maps` - one child-to-parent integer map per adjacent level pair
- `metadata` - optional provider-specific information ignored by the fixed evaluator

`nb/Prepare.ipynb` validates this boundary before training. A fixed open-source split may ignore the provider's data seed, but the evaluator still uses the configured seed for neural initialization and training order.

Select another provider without modifying evaluator code:

```text
AUTORD_DATASET_NOTEBOOK=/absolute/path/to/OpenSourceDataset.ipynb
```

Relative provider paths are resolved against `nb/`. Selecting or changing a provider defines a different benchmark and is an operator action, not an Auto-R&D rule mutation.

## Dataset visualization

Run `nb/Dataset.ipynb` interactively to visualize a generated Gaussian hierarchy. Its example section uses `src/util/plot_util.py` to show:

- one shared two-dimensional PCA projection colored separately at each hierarchy level;
- individual and cumulative PCA explained variance;
- correlations among the highest-variance input features; and
- the ground-truth child-to-parent hierarchy.

The plotting cell is tagged `skip-on-provider-import`. The evaluator ignores cells with this tag when loading a provider, keeping exploratory visualization outside automated fitness evaluation. Plotting helpers return Matplotlib figures and axes without calling `show()`, so other provider notebooks can reuse and compose them.

## Default generative hierarchy

The default provider samples top-level centers from a uniform box. Every lower-level prototype is sampled around one parent:

```text
mu_child ~ Normal(mu_parent, sigma_level^2 I)
```

Cardinality increases toward the leaves. Observations are sampled around leaf prototypes. The provider records the exact hierarchy node associated with every sample at every level, so the true hierarchy is known.

## Permutation-invariant evaluation

A learned neural layer is compared with a dataset hierarchy level even though neuron indices may be arbitrarily permuted. For correspondence matrix `C`, `C[i,j]` is the row-normalized mean activity of learned neuron `i` for samples belonging to hierarchy node `j`. The evaluator uses the Hungarian algorithm to find the maximum-correspondence assignment. Its score rewards high assigned entries, penalizes mean unmatched correspondence, and includes an assignment-coverage factor for rectangular comparisons.

The current metrics are:

1. **Within-level recovery** - the assignment score for neural layer `l` against hierarchy level `l`.
2. **Level specificity** - the intended-level score minus the best score against another hierarchy level.
3. **Hierarchy/topology consistency** - agreement between the provider's parent maps and parent-child relations inferred from adjacent-layer co-activity after independently aligning both layers.
4. **Numerical stability** - updates with non-finite weights fail, and weight-row norms are capped at `5.0`. Softmax activations are bounded when their inputs are finite.
5. **Panel robustness** - fitness combines results across the development or promotion cases and seeds declared by the selected provider.

Activity-range, neuron-collapse, locality, forbidden-state-access, and public-API checks remain protocol or review requirements; the current evaluator does not enforce them all automatically.

## Learning-rule API and locality contract

The runner calls the following public API in `nb/Rules.py`:

- `RuleConfig` and `PARAM_BOUNDS`
- `init_state(n_pre, n_post, rng)`
- `activate(x, w, state, cfg)`
- `update(x, y, w, state, cfg)`
- `complexity_score()`

Here `x` and `y` are the current layer's presynaptic and postsynaptic activities, `w` contains its synaptic weights, and `state` is mutable state created separately for that layer. A rule must not access hierarchy labels, parent assignments, provider metadata, evaluation metrics, optimal assignments, gradients, downstream losses, evaluator globals, seed/case identity, or benchmark results. This locality contract is enforced procedurally rather than through a runtime sandbox.

## Auto-R&D loop

Each coding-agent iteration should:

1. Read `AGENTS.md`, the selected dataset provider, the current `nb/Rules.py`, and recent experiment evidence.
2. State one explicit mechanistic hypothesis.
3. Snapshot the current rule under `out/rules/` without overwriting an earlier snapshot.
4. Modify only `nb/Rules.py`.
5. Tune continuous hyperparameters using the evaluator's SciPy optimizer and its checked-in budget.
6. Evaluate on the selected provider's configured development and promotion panels.
7. Reject violations of numerical checks or the broader protocol requirements.
8. Compare mean, across-panel variance, minimum performance, hierarchy alignment, and complexity.
9. Keep the candidate only if it meets the promotion criterion; otherwise restore the previous champion byte-for-byte.
10. Report the hypothesis, mechanism, provider, results, decision, interpretation, and next hypothesis. Persist these details only in a notes file explicitly designated for that purpose.

The checked-in evaluation call uses a short smoke-test optimizer budget (`maxiter=2`, `popsize=3`). Its output is preliminary; a promotion-quality optimization requires an operator-designated fixed budget. A coding agent must not change evaluation budgets, dataset providers, cases, or seeds itself.

The long-term objective is not merely to find a high-scoring opaque rule, but to discover compact, interpretable plasticity principles that can later be tested for transfer across independently defined dataset providers.

## Getting started

Install the dependencies:

```text
python -m pip install -r requirements.txt
```

Launch the default benchmark from the repository root:

```text
bash sbin/run_eval.sh
```

On Windows:

```text
sbin\run_eval.bat
```

Pass an alternative provider notebook as the first argument:

```text
bash sbin/run_eval.sh /absolute/path/to/OpenSourceDataset.ipynb
```

The initial `nb/Rules.py` implements a simple competitive Hebbian/Oja-like baseline. The checked-in evaluator performs only the short smoke-test optimization described above.

To regenerate the human-readable rule notebook from the repository root, run `sbin/py_to_ipynb.sh` or `sbin\py_to_ipynb.bat` with no arguments. Explicit input and output paths remain supported.

The code requires Python 3.10 or newer.
