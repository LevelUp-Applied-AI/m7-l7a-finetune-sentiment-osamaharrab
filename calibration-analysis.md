# Calibration Analysis

## Reliability diagram interpretation

The saved reliability diagram (`figures/reliability-diagram.png`) shows that most predictions are high-confidence: 1,460 of 1,867 holdout examples fall in the 0.6-1.0 confidence buckets. The model is mostly over-confident. The 0.6-0.7 bucket has mean confidence 0.649 but accuracy 0.503, and the 0.7-0.8 bucket has mean confidence 0.752 but accuracy 0.585. Even the strongest 0.9-1.0 bucket is over-confident: mean confidence 0.937 versus accuracy 0.840. The only under-confident bucket is 0.3-0.4, where accuracy is 0.600 versus mean confidence 0.382, but that bucket contains only 5 examples, so I would not put much weight on it.

## Expected Calibration Error

The expected calibration error is 0.1143 on 1,867 held-out reviews. This means that, after weighting by bucket size, the model's predicted confidence differs from observed accuracy by about 11.4 percentage points. For production, I would treat the class label as useful but not treat the raw softmax probability as a trustworthy probability of correctness. A prediction shown as 0.75 confidence is not actually correct about 75% of the time in this evaluation.

## A specific calibration pattern

The strongest pattern is over-confidence near the neutral decision boundary. When the model predicted class 1 (neutral), its precision was only 0.484 while its mean confidence was 0.648. Negative and positive predictions were also over-confident, but less severely: negative precision was 0.739 at mean confidence 0.824, and positive precision was 0.695 at mean confidence 0.782. This likely comes from the label construction and task shape. Neutral reviews are mapped from 3-star ratings, which often mix praise and complaints, so they sit between the clearer negative and positive classes. Fine-tuning with ordinary cross-entropy encourages a single confident class even when the text is borderline.

## A proposed engineering action

I would add temperature scaling on a held-out calibration set before exposing probabilities to downstream systems. I would also use threshold-based abstention for neutral predictions: for example, route neutral predictions below 0.80 calibrated confidence to human review or a second model, because the current neutral predictions are much less reliable than their confidence suggests. Finally, I would collect more boundary examples where 3-star reviews contain mixed sentiment, since that is the area most likely to improve both accuracy and calibration.
