# Getting Started with RAGAS eval for RAG

RAGAS (Retrieval-Augmented Generation Assessment) is an open-source framework designed to quantitatively evaluate RAG pipelines by measuring faithfulness, answer relevancy, context precision, and context recall.  Unlike traditional metrics, it uses an "LLM-as-a-Judge" approach to assess factual accuracy and context grounding, often requiring no human-annotated ground truth labels. This allows developers to systematically diagnose retrieval and generation bottlenecks without relying on subjective "vibe checks."

Core concepts - experiments, datasets and metrics

---


### 1. Experiments



An experiment in RAGAS is a deliberate, isolated change to your application, run against a fixed test dataset and scored by metrics, so you can measure its impact.

Think of it as a scientific loop: hypothesize → change one thing → run evals → observe → iterate. The classic example: swap the embedding model in your RAG pipeline and see if retrieval quality (and downstream answer quality) improves.



Three components every experiment has:



|Component|Role|
|-|-|
|Test dataset|Fixed set of inputs (queries + expected outputs) you evaluate against|
|Application endpoint|The code/model being tested (your RAG app, a single retriever, a prompt, etc.)|
|Metrics|Quantitative scores that let you compare "before" vs. "after"|



Principles that make an experiment useful:

* Isolate one change at a time  -  if you change the chunking strategy and the LLM simultaneously, you can't attribute the score delta.
* Use measurable metrics  -  not "it feels better," but a numeric delta on faithfulness, context recall, etc.
* Store results systematically  -  RAGAS auto-saves each run to an `experiments/` directory so you can compare across versions.



In code, you define an experiment with the `@experiment` decorator:

```python
from ragas import experiment

@experiment
async def my\\\_experiment(row, model):
    response = my\\\_app(row.query, model=model)
    score = my\\\_metric.score(response, row.ground\\\_truth)
    return {row, "response": response, "accuracy": score.value}

 Run it  -  results are stored automatically
await my\\\_experiment.arun(dataset, model="gpt-4o")


```



The decorator handles iteration over every row in the dataset, collects the returned dicts, and persists them. You can then re-run the same function with a different `model` argument to get a second run you can diff against the first.

The mental model: an experiment is not "evaluating your app" in a one-shot sense. It's a reproducible unit of comparison  -  the same dataset, the same metrics, but one variable changed. That's what makes the score delta meaningful.





---



### 2. Datasets

Building on the experiment loop from before  -  datasets are the fixed input side, and experiment results are the output side. RAGAS treats them as two distinct artifacts with different lifecycles:



||Evaluation Dataset|Experiment Results|
|-|-|-|
|Stored in|`datasets/`|`experiments/`|
|Contains|Inputs (queries), optional expected outputs, optional metadata|All dataset attributes + system response + metric scores + optional metadata (e.g. trace URI)|
|Lifecycle|You author it once, version it, reuse across many experiments|Generated fresh on every experiment run|



The key mental shift: an experiment result is a superset of the dataset row. The `@experiment` function receives a row, and its return dict is what gets persisted  -  so `{row, "response": ..., "accuracy": ...}` means the saved CSV has the original query, the new response, and the score all in one place. That's what makes cross-run comparison trivial: same input columns, different response/score columns.

Storage is backend-agnostic. CSV is the default (simple, small, diffable), but RAGAS's `Dataset` and `Experiment` classes accept a `backend` parameter:



```python
from ragas import Dataset, Experiment

 Load a dataset
dataset = Dataset.load(name="my\\\_eval\\\_v1.2", backend="local/csv", root\\\_dir="./datasets")

 Load results from a previous run
results = Experiment.load(name="first\\\_experiment", backend="local/csv", root\\\_dir=".")
```



Versioning matters. The docs explicitly recommend encoding a version in the dataset name (`my\\\_eval\\\_v1.2`) so that when you compare two experiment results, you can confirm they were run against the same input set. Without this, a score delta is ambiguous  -  did the app change, or did the test data change?

In the workflow you already saw, the flow is:



```
datasets/my\\\_eval\\\_v1.2.csv  ──►  @experiment function  ──►  experiments/20251007\\\_baseline.csv
         (fixed input)                                    (response + scores per row)
```



You can then load both result files with pandas and diff the score columns to see the impact of whatever you changed.





---



### 3. Metrics





The "Few Strong Signals" Principle

This is the most practically important principle in the whole metrics section, because it directly governs how you design experiments.

The core claim: a metric is only useful if it changes your decision. If you track 8 metrics and they all move in the same direction (or don't move at all), you've added 7 units of noise and cost without buying any additional signal.



The docs' example makes this concrete:



|Weak signal approach|Strong signal approach|
|-|-|
|Track "coherence" + "helpfulness" + "fluency" + "tone" + "length" + …|Track goal accuracy  -  was the user's objective met?|



The single "goal accuracy" metric subsumes the others. If the goal was met, coherence and tone were almost certainly fine. If it wasn't, you know something went wrong without needing to triangulate across six weak proxies.

Why this matters for experimentation specifically:

When you run an experiment (change one variable, measure the delta), you need the metric to discriminate between "it got better" and "it didn't." Weak metrics have high variance and low effect-size sensitivity  -  a 0.02 swing in "helpfulness" could be noise. A 0.15 swing in "goal accuracy" is a real signal you can act on.

The practical rule: start with 1 - 2 end-to-end metrics that map directly to what your user cares about. Add component-level metrics only when you need to diagnose why the end-to-end score moved.



The Four Design Principles (quick reference)

These are the internal consistency rules RAGAS follows when building its own metrics:



||Principle|What it means in practice|
|-|-|-|
|1|Single-aspect focus|One metric = one question. Don't build a "quality score" that blends faithfulness + relevance + tone.|
|2|Intuitive \& interpretable|A teammate should be able to explain what the score means without reading source code.|
|3|Objective over subjective|Prefer metrics with a verifiable criterion (claim is entailed by context) over vibes ("the answer feels good").|
|4|Consistent range \[0, 1]|Every metric is normalized so you can compare and combine them without unit conversion.|





Available Metrics  -  Mental Model (landscape view)

You don't need to memorize these. What you need is the shape of the space so you can navigate it when a problem arises.

By pipeline stage (the most useful axis):



|Stage|Metrics|Question they answer|
|-|-|-|
|Retrieval|Context Precision, Context Recall, Context Entity Recall, Context Relevance|Did the retriever find the right material?|
|Generation|Faithfulness, Answer Relevancy|Did the LLM use that material correctly and stay on-topic?|
|End-to-end|Answer Correctness, Answer Similarity|Is the final answer actually right?|
|Agent / multi-turn|Topic Adherence, Tool Call Accuracy, Agent Goal Accuracy|Did the agent do the right thing across turns?|
|Traditional (non-LLM)|BLEU, ROUGE, Exact Match, String Presence|Cheap sanity checks, no LLM judge needed|



By whether you need ground truth:

* Reference-free (works in production, no labels): Faithfulness, Answer Relevancy, Context Precision (without reference)
* Reference-based (needs a labeled `ground\\\_truth`): Context Recall, Answer Correctness, Context Precision (with reference)



This distinction matters for your experimentation workflow: reference-free metrics let you evaluate any production traffic; reference-based metrics give you a stronger signal but require you to have authored a dataset with expected outputs (which loops back to the datasets and test data generation concepts).



The one-liner mental model: pick 1 - 2 end-to-end metrics as your headline, add 1 - 2 component metrics only for diagnosis, and resist the urge to add more unless a specific failure mode is invisible to your current set.

