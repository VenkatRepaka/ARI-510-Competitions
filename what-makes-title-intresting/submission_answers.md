# Submission form answers (temporary draft)

Plain text, ready to paste into the form. Dev score = mean MAE over 5-fold × 10-repeat cross-validation.

---

## Methodology

Approach: the task is to predict how interesting a title is (1–6) from its text and source. The key fact about the data is that each of the 120 training titles has one rater, but each test label is the mean of six raters. Training labels are whole numbers spread over the full scale, while test labels are averages close to the middle. So the plan was to keep the learned part of every model very small and to get the "average reader" signal from a model that was never trained on the noisy labels: a zero-shot LLM. Conventional models were also tried, to check that none of them clearly beats this idea.

Preprocessing: no text cleaning, because capitals, emoji, hashtags and punctuation are useful cues. The source (YouTube, Reddit, podcast, article) is one-hot encoded and used in every model. Sentence encoders read "<source>: <title>"; LLM prompts use a plain source name such as "YouTube video" or "news article". Predictions are clipped to 1–6.

Dev set and evaluation: a single train/dev split of 120 titles is too noisy (one fold of 24 titles varies by ±0.17 MAE), so every model is scored with RepeatedKFold (5 folds × 10 repeats, seed 42) on the training data, i.e. 50 fits per model. All fitting (scalers, ridge, SVR, fine-tuning) happens inside each fold. Dev score = mean MAE over the 50 folds; the out-of-fold correlation with the labels was also tracked because it is less noisy. Differences below about 0.02 were treated as noise. Dev MAE is measured against single-rater labels, so it is much higher than leaderboard MAE (constant: 1.495 dev vs 0.848 public) and is only used to compare models. The public board has only ~15 titles, so choices were made on dev MAE, not on the public score. LLM scores are computed zero-shot for all 150 titles (no label is ever shown), so they can be used inside the CV without leakage.

Models tried:
- CONST: training mean (3.133) for every title. Mean per source was also tried (1.478).
- SENTENCE-EMBEDDINGS-SVR: mpnet sentence embeddings + source, RBF SVR, predictions shrunk toward the mean. Ridge and kNN on the same embeddings were slightly worse; a small neural network overfit (1.72–1.79).
- HAND-RIDGE: 24 hand-crafted features (length, capitals, ?, !, digits, "you", emoji, hashtags) × source, ridge. Showed that cues mean different things per source: longer podcast titles are more interesting, Reddit questions less, hashtags and emoji hurt on YouTube. Dev 1.449. Not submitted.
- BERT-EMBEDDINGS-SVR: bert-base mean-pooled embeddings + source, SVR. Best non-LLM model, but picked from 27 tries, so slightly optimistic. Dev 1.433. Not submitted.
- OTHER-CLASSIC-MODELS: CLIP text encoder + kNN, TF-IDF ridge, random forest, gradient boosting, MLPs, and a fully fine-tuned mpnet (1.500; with ~96 noisy titles per fold it only learns the noise). Dev 1.44–1.52. Not submitted.
- CLICKBAIT-TRANSFER: the competition mentions clickbait traits (curiosity gap, exaggeration), so models were trained on the Webis Clickbait 2017 corpus (40k tweets, CC BY 4.0; Pearson 0.73 on its own test set) and their scores used as features. Clickbait scores did not predict interest. Dev 1.468. Not submitted.
- LLM-Q: local Qwen2.5-VL-7B (4-bit, MLX), zero-shot. No text is generated: one forward pass gives the next-token probabilities of "1"…"6", which are rescaled and turned into an expected rating (1·p1 + … + 6·p6). This keeps the model's uncertainty and is deterministic. A simple sanity check showed a problem: the prompt asks for one digit, yet on average only 63 % of the probability fell on "1"…"6", and only the first title in a run looked normal. The cause was traced (with help from an AI coding assistant) to the mlx-vlm library keeping stored state from the first prompt; clearing it before each call fixed this, and the same check then gave at least 99.9 % on the digits for every title. A ridge on [rating + source] maps the rating to the label scale with only 5 learned numbers (one slope, one shift per source).
- LLM-PAIR: the same Qwen model compares pairs of titles ("which would a typical reader find more interesting, A or B?"). Each title meets at least 12 random others (1,126 pairs), each pair in both orders to cancel the model's preference for "A". The soft win rate per title goes into the same ridge. A Bradley-Terry score was also tried (1.408).
- LLM-CLAUDE: Claude Opus 5.5 via the Anthropic API (effort "low", total cost about $0.46) with the same prompt. The API gives no token probabilities, so Claude returns p1–p6 as structured JSON, turned into an expected rating and fitted with the same ridge. Its rating correlates twice as strongly with the labels as Qwen's (Spearman 0.38 vs 0.19) and is positive in every source. Scores are saved to a CSV, since the API has no seed.

Hyperparameter selection: small grid searches on the same repeated CV, picking the lowest dev MAE and preferring the simpler setting when results were within noise. SVR C from {0.3, 1, 3} → 0.3; shrinkage toward the mean → s = 0.4; kNN k from {10, 20, 40} → 40; hand-crafted ridge α from {30, 100, 300, 1000} → 300; LLM ridge α from {1, 10, 30} → 1 (results nearly flat).

LLM prompt selection and what else was tried:
The main prompt was designed from what the data showed, and an LLM assistant (Claude) was used to review and refine the wording before the final version was fixed. The readout uses the probability of the first answer token, so the prompt has to be short and force a single digit 1–6. Final prompt:

System: "You are an average adult internet user judging content titles. Answer only with what is asked."
User: "How interesting would a typical reader find this {YouTube video / Reddit post / podcast episode / news article} title?
Scale: 1 = not interesting at all, 2 = slightly, 3 = somewhat, 4 = fairly, 5 = very, 6 = extremely interesting.
Title: {title}
Answer with a single digit from 1 to 6."

Why each part is there: the "average adult internet user" role and the "typical reader" question match the test label, which is the mean of six ordinary readers; the source type is included because cues mean different things per source (long titles help podcasts, questions hurt on Reddit); the scale uses the same 1–6 range as the labels with a word for each level; and "Answer with a single digit" keeps the first token a digit. The digit mass check (≥ 99.9 % of the probability on "1"…"6") confirmed the format works.

Other prompts tried, each compared with the main prompt (dev 1.415) using the same model, readout and CV:
- Click: "You are scrolling past this {source}. How likely are you to click or open it, based only on its title? Scale: 1 = would never click … 6 = would definitely click." Adding it gave 1.428.
- Five Yes/No traits, read as P(Yes): "Does this {source} title deliberately hold back key information to make the reader curious (a curiosity gap)?", "… use emotional or emotionally charged language?", "… exaggerate or use hype (ALL CAPS, superlatives, sensational claims)?", "… directly address the reader ('you', 'your', a question)?", "… miss details a reader would need to understand what it is about?". Adding them gave 1.443–1.454.
- Six personas as system messages, e.g. "You are a 20-year-old university student who spends a lot of time on YouTube, Reddit and gaming." or "You are a 68-year-old retiree who reads newspapers daily and enjoys history, health and local news.", with the question "How interesting is this title to you personally?". The average gave 1.446; the personas agreed too much (correlation 0.83) to imitate six real raters.
- Pairwise: "Which of these two titles would a typical reader find more interesting? A ({source}): {title A} B ({source}): {title B} Answer with a single letter: A or B." This gave the best local Qwen dev score (1.393) but a worse public score (0.812).
- The same main prompt on a second local model (Qwen3-8B, 1.439) and on Claude Opus 5.5 (1.378, used for the best run). For Claude one sentence was added, because the API returns no token probabilities: "Different readers disagree, so give the probability that a randomly chosen typical reader would pick each score (p1..p6, each between 0 and 1, summing to 1)."

Wording check (prompt_variants.ipynb): one part of the main prompt was changed at a time. Removing the system role (1.433), removing the source type (1.431), asking "How interesting is this title to you?" instead of "to a typical reader" (1.431) and reversing the question to "How boring would a typical reader find this title?" (1.447) were all worse; removing the words on the scale gave 1.410, a tie within the ~0.02 CV tolerance. An editor variant ("You are a professional editor who reviews headlines. Rate the quality of this {source} title from 1 (very poor) to 6 (excellent).") scored 1.399.

Conclusion: the typical-reader prompt was retained for single-title scoring, with both Qwen2.5 and Claude. None of the other questions (click, traits, personas) improved on it. The pairwise prompt is a different kind of task (comparing two titles), and despite its better dev score it did worse on the public board. In the wording check, only reversing the question ("how boring") was clearly worse; dropping the role, the source type or the "typical reader" view made the score slightly worse (by 0.016–0.018, just inside the ~0.02 tolerance), and the scale wording tied, so the labelled scale was kept because it gives each digit a clear meaning. The editor variant scored slightly better (1.399, also within the tolerance), but that check was run after the submissions, so it is reported here and was not used in any submitted run.

---

## Runs

run1 | CONST: training mean | 1.495 | 0.84793
run2 | SENTENCE-EMBEDDINGS-SVR: mpnet + source, SVR | 1.463 | 0.82826
run3 | LLM-Q: Qwen2.5-7B rating + source, ridge | 1.415 | 0.72446
run8 | LLM-PAIR: Qwen2.5-7B pairwise win rate + source, ridge | 1.393 | 0.81173
run9 | LLM-CLAUDE: Claude Opus 5.5 rating + source, ridge | 1.378 | 0.62883

---

## Reflection

LLM-CLAUDE had the lowest MAE in repeated cross-validation (1.378) and on the public leaderboard (0.629). My main takeaway was that how the labels were collected should shape the modeling approach. Each training title had one rating, while each test label averaged six ratings. Asking what a typical reader would find interesting matched the kind of judgment the test labels represented. A small ridge model then used the 120 training titles to calibrate the LLM scores.

The prompt experiments showed the value of asking directly about interest. Adding click-likelihood or headline-trait scores increased dev MAE, and the six personas agreed too closely to capture much variation across readers. The editor prompt scored 1.399 in a check after submissions, a small improvement over 1.415 for the typical-reader Qwen prompt and within the roughly 0.02 tolerance used for dev comparisons.

LLM-PAIR scored better than LLM-Q on dev (1.393 vs. 1.415 MAE), while LLM-Q did better on the public leaderboard (0.724 vs. 0.812 MAE). Repeated cross-validation reduced dependence on any one split, but dev scores still reflected single-rater labels. With only about 15 titles on the public leaderboard, I treated the reversal as a reason to be cautious about small differences on either measure.

Difficulties: the problem that took the most time was with Qwen. Its scores looked normal, but only 63 % of the probability was on the digits 1–6. The cause was stored data in the mlx-vlm library. After clearing it before each title, the check showed at least 99.9 % on the digits for every title. Pairwise scoring also had to use each pair in both orders, because the model preferred one position (2,252 calls). Claude had to return its answer as JSON, because its API does not give token probabilities.
