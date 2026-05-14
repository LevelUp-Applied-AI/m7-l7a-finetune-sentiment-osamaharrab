# Adversarial Evaluation Analysis

## Per-hypothesis accuracy

| Hypothesis category | Correct | Incorrect | Total | Accuracy |
|---|---:|---:|---:|---:|
| negation | 2 | 4 | 6 | 33.3% |
| lexical_trigger | 1 | 5 | 6 | 16.7% |
| domain_shift | 3 | 3 | 6 | 50.0% |
| length_extreme | 4 | 2 | 6 | 66.7% |
| sarcasm | 0 | 6 | 6 | 0.0% |
| other | 0 | 3 | 3 | 0.0% |

Overall accuracy was 10/33, or 30.3%.

## Confirmed hypotheses

Sarcasm was the clearest confirmed failure mode. Rows 25-30 were all expected negative but were predicted positive; row 25 was predicted positive with 0.927 probability for "Fantastic work turning a two tap payment into a ten minute chore," and row 29 was predicted positive with 0.913 probability for a "Perfect update" that forgets the password every morning.

Lexical triggers also behaved as predicted. Row 10 was expected negative but predicted positive with 0.839 probability, apparently overweighting "amazing" even though the sentence says the app charges a fee before anything works. Row 11 flipped the other way: it was expected positive but predicted negative with 0.877 probability because the word "bad" appears inside "A bad review would be unfair."

Negation was partly confirmed. The model got explicit negative recommendations right, but it missed negated negative cues that imply improvement: row 4 expected positive was predicted negative with 0.932 probability, row 5 expected positive was predicted negative, and row 8 expected positive was predicted neutral.

The custom "other" category also exposed a weakness with mixed outcomes. Rows 31-33 were all predicted neutral even though the human label should be driven by the decisive outcome: export failure in row 31, narrow timer accuracy in row 32, and unrecoverable data loss in row 33.

## Refuted hypotheses

Length extremes were less damaging than I expected. The model correctly labeled the very short row 19 as positive and row 21 as neutral, and it also handled the long negative row 22 and long neutral row 24. The only length failures were row 20, where "Still broken today" was predicted neutral, and row 23, where a long positive migration story was predicted neutral.

Domain shift was also only partly problematic. The model correctly treated row 3, row 15, and row 17 as neutral, including a sports sentence and a recipe instruction. That refutes the strongest version of the hypothesis that any non-app domain would automatically be forced into positive or negative sentiment.

Negation was not uniformly broken. Row 1, "did not improve battery life," and row 7, "would not recommend," were both predicted negative correctly. The model seems better at negation when the final meaning remains negative than when negation turns a negative cue into a positive one.

## What the results reveal about the decision boundary

The decision boundary appears to rely heavily on local sentiment cues and less on pragmatic reversal. Sarcasm rows 25-30 all crossed into positive because surface words like "Fantastic," "Love," "Best," "Thanks," "Perfect," and "delightful" outweighed the negative situation described later in the sentence.

The model is also asymmetric on negation. It can use negation when it reinforces a negative app-review pattern, as in rows 1 and 7, but it struggles when negation turns a negative cue into praise, as in rows 4, 5, 6, and 8. That suggests the learned boundary treats words like "hate," "crashes," "bad," and "broken" as strong anchors unless the surrounding construction is very common in training.

Finally, neutral seems to be used as a fallback for mixed or unfamiliar framing. Rows 9, 12, 13, 31, 32, and 33 were predicted neutral even when the expected label was positive or negative. The boundary is cautious when the sentence includes competing evidence or a domain frame that does not look like a plain app-store review.
