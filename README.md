# Learning Rules Auto-R&D for Hierarchy Recovery

This project is a controlled research harness for discovering **local neural learning rules** that recover hierarchical representations without backpropagation.

The central idea is to generate data from a known multidimensional hierarchical process, build a fixed neural hierarchy whose layer widths match the latent-level cardinalities, and let an automated coding agent (for example, Codex or Claude Code) iteratively modify only the learning rule. Candidate rules are scored with fixed, permutation-invariant hierarchy-recovery metrics. Locality and the broader scientific protocol are enforced by the agent instructions and code review; the current evaluator is not a security sandbox.

## Core hypothesis

A useful local plasticity rule should allow successive neural layers to recover increasingly coarse latent structure using only locally available variables. Neuron identities are arbitrary, so recovery is evaluated **up to permutation**. A good layer has high matched correspondence and low unmatched cross-talk after neurons are optimally assigned to latent nodes.

The initial benchmark uses nested Gaussian hierarchies because they provide exact ground truth. Later stages can test richer distributions and invariances, including anisotropic clusters, unequal priors, nonlinear manifolds, transformation-defined equivalence classes, temporal continuity, and sensorimotor trajectories.

## Project layout

- `nb/Prepare.ipynb` - **immutable benchmark definition**. Contains the hierarchical Gaussian generator, fixed neural topology, reproducibility utilities, and metric implementations. The coding agent must never edit this file during Auto-R&D.
- `nb/Rules.ipynb` - human-readable notebook form of the initial learning rule and its explanation. The coding agent does not edit this notebook directly.
- `nb/Rules.py` - **the only learning-rule source the coding agent may modify** during the main search loop.
- `nb/Eval.ipynb` - loads the fixed benchmark and `nb/Rules.py`, tunes rule hyperparameters, evaluates configured seed/world panels, plots a cross-level correspondence score matrix, and records compact trial results.
- `AGENTS.md` - autonomous experiment protocol for the coding agent.
- `docs/papers/learning_rules_autord.tex`, `docs/papers/learning_rules_autord.pdf` - research paper draft describing the method and motivation.
- `sbin/ipynb_to_py.sh`, `sbin/ipynb_to_py.bat` - notebook-to-Python utilities.
- `sbin/py_to_ipynb.sh`, `sbin/py_to_ipynb.bat` - Python-to-notebook utilities for refreshing `nb/Rules.ipynb` from an accepted `nb/Rules.py`. Pass these paths explicitly when invoking the scripts from the repository root.
- `requirements.txt` - Python packages required by the current notebooks and conversion utilities.
- `out/rules/` - append-only snapshots created by the agent before rule mutations.
- `nb/experiments.csv` - compact numeric results created or extended by `nb/Eval.ipynb` when its kernel working directory is `nb/`.

## Generative hierarchy

At level `L`, latent prototypes live in `R^d`. Top-level centers are sampled from a uniform box. Every lower-level prototype is sampled around one parent:

```text
mu_child ~ Normal(mu_parent, sigma_level^2 I)
```

Cardinality increases toward the leaves. Observations are sampled around leaf prototypes. The generator records the exact latent node associated with every sample at every level, so the true hierarchy is known.

## Permutation-invariant evaluation

A learned neural layer is compared with a latent level even though neuron indices may be arbitrarily permuted. For correspondence matrix `C`, `C[i,j]` is the row-normalized mean activity of learned neuron `i` for samples belonging to latent node `j`. The evaluator uses the Hungarian algorithm to find the maximum-correspondence assignment. Its score rewards high assigned entries, penalizes mean unmatched correspondence, and includes an assignment-coverage factor for rectangular comparisons.

The current metrics are:

1. **Within-level recovery** - the assignment score for neural layer `l` against latent level `l`.
2. **Level specificity** - the intended-level score minus the best score against another latent level.
3. **Hierarchy/topology consistency** - agreement between the true generative tree and parent-child relations inferred from adjacent-layer co-activity after independently aligning both layers.
4. **Numerical stability** - updates with non-finite weights fail, and weight-row norms are capped at `5.0`. Softmax activations are bounded when their inputs are finite.
5. **Panel robustness** - parameter optimization uses two development world specifications and three seeds; the promotion panel evaluates those specifications plus one additional specification across five seeds. These worlds are visible in the evaluator and are not a hidden test set.

Activity-range, neuron-collapse, locality, forbidden-state-access, and public-API checks remain protocol or review requirements; the current evaluator does not enforce them all automatically.

## Learning-rule API and locality contract

The runner calls the following public API in `nb/Rules.py`:

- `RuleConfig` and `PARAM_BOUNDS`
- `init_state(n_pre, n_post, rng)`
- `activate(x, w, state, cfg)`
- `update(x, y, w, state, cfg)`
- `complexity_score()`

Here `x` and `y` are the current layer's presynaptic and postsynaptic activities, `w` contains its synaptic weights, and `state` is mutable state created separately for that layer. A rule must not access latent labels, parent assignments, evaluation metrics, optimal assignments, gradients, downstream losses, evaluator globals, seed/world identity, or benchmark results. This locality contract is enforced procedurally rather than through a runtime sandbox.

## Auto-R&D loop

Each coding-agent iteration should:

1. Read `AGENTS.md`, the current `nb/Rules.py`, and recent experiment evidence.
2. State one explicit mechanistic hypothesis.
3. Snapshot the current rule under `out/rules/` without overwriting an earlier snapshot.
4. Modify only `nb/Rules.py`.
5. Tune continuous hyperparameters using the evaluator's SciPy optimizer and its checked-in budget.
6. Evaluate on the configured development and promotion panels.
7. Reject violations of numerical checks or the broader protocol requirements.
8. Compare mean, across-panel variance, minimum performance, hierarchy alignment, and complexity.
9. Keep the candidate only if it meets the promotion criterion; otherwise restore the previous champion byte-for-byte.
10. Report the hypothesis, mechanism, results, decision, interpretation, and next hypothesis. Persist these details only in a notes file explicitly designated for that purpose.

The checked-in evaluation call uses a short smoke-test optimizer budget (`maxiter=2`, `popsize=3`). Its output is preliminary; a promotion-quality optimization requires an operator-designated fixed budget. A coding agent must not change evaluation budgets itself.

The long-term objective is not merely to find a high-scoring opaque rule, but to discover compact, interpretable plasticity principles that can later be tested for transfer beyond synthetic Gaussian hierarchies.

## Getting started

Install the current dependencies:

```text
python -m pip install -r requirements.txt
```

Inspect `nb/Prepare.ipynb`, then start or run the evaluator with `nb/` as the kernel working directory. For example:

```text
cd nb
jupyter notebook Eval.ipynb
```

The initial `nb/Rules.py` implements a simple competitive Hebbian/Oja-like baseline. The checked-in evaluator performs only the short smoke-test optimization described above.

To regenerate the human-readable rule notebook from the repository root, use explicit paths:

```text
bash sbin/py_to_ipynb.sh nb/Rules.py nb/Rules.ipynb
```

The code requires Python 3.10 or newer.
