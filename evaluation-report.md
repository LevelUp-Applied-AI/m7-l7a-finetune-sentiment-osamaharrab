# Module 7 Week A — Lab Evaluation Report

## Dataset
AARSynth app reviews Sentences-50Agree subset with 7,472 examples across negative, neutral, and positive labels. Label distribution: negative=2,519, neutral=2,442, positive=2,511; split sizes: 5,977 train and 1,495 test.

## Model and hyperparameters
- Backbone: distilbert-base-uncased
- Number of labels: 3
- Learning rate: 5e-05
- Epochs: 2
- Batch size: 8
- max_length: 128
- Seed: 42
- Training time: 1,137.0 seconds wall-clock on this machine

## Metrics on the test split

Aggregate:

| Metric | Value |
| --- | --- |
| Accuracy | 0.6234 |
| Macro-F1 | 0.6200 |

Per class:

| Class | F1 | Precision | Recall |
| --- | --- | --- | --- |
| Negative | 0.7058 | 0.7002 | 0.7114 |
| Neutral | 0.4669 | 0.4549 | 0.4795 |
| Positive | 0.6873 | 0.7100 | 0.6660 |

## Confusion matrix

| True \ Pred | negative | neutral | positive |
| --- | --- | --- | --- |
| negative | 355 | 124 | 20 |
| neutral | 116 | 222 | 125 |
| positive | 36 | 142 | 355 |

## Three qualitative error examples (one per class)

### Negative
- Original sentence: slow, laggy performance. thinkfree office opens my pdfs and allows smooth transitions and zooming.
- Gold label: negative
- Predicted label: neutral
- Predicted probability for the gold label: 0.0600
- Analysis: The sentence has clear negative cues in "slow" and "laggy," but the second clause describes a competitor working well. The model appears to soften the complaint into neutral because the sentence is partly comparative rather than a direct app failure.

### Neutral
- Original sentence: great <url> ruined with ads
- Gold label: neutral
- Predicted label: negative
- Predicted probability for the gold label: 0.0327
- Analysis: This looks like annotation ambiguity or label noise: "ruined with ads" is a strong negative phrase even though the gold label is neutral. The model's negative prediction is understandable and shows that ad-related complaint language dominates the opening positive word.

### Positive
- Original sentence: addicted to phone? this is the app
- Gold label: positive
- Predicted label: negative
- Predicted probability for the gold label: 0.0094
- Analysis: The short review is probably praising the app as useful for phone addiction, but the cue word "addicted" is strongly negative out of context. With little surrounding text, the model overweights that cue and misses the implied recommendation.

## Hugging Face Hub model URL
https://huggingface.co/osamaharrab/m7-app-review-sentiment
