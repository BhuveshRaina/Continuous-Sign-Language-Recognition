# Preprocessing Pipeline Analysis Report
**Status:** SUCCESS
**Date:** 2026-04-19

## 1. Executive Summary
The preprocessing pipeline successfully identified and consolidated redundant label variations caused by capitalization, punctuation, and linguistic inconsistencies (contractions). This significantly improves model training stability by reducing the number of sparse classes.

### Key Metrics
- **Total Video Samples:** 2410
- **Initial Unique Variations:** 197
- **Final Canonical Classes:** 108
- **Labels Eliminated:** 89
- **Redundancy Reduction:** 45.18%

## 2. Visualization (Impact)
```text
Original Variations: [########################################] 197
Canonical Classes:   [#####################] 108
```

## 3. High-Density Canonical Classes
Classes with the most variation in the original dataset:

| Canonical Class | Variations Merged | List of Variations |
| :--- | :---: | :--- |
| i dont agree | 6 | `I do not agree`, `I dont agree`, `I_don't_agree`, `i do not agree`, `i dont agree`, `i_do_not_agree` |
| You are bad | 4 | ` you are bad`, `You are bad`, `You are bad `, `you are bad` |
| I am afraid of that | 3 | `I am afraid of that`, `i am afraid of that`, `i am afraid of that ` |
| I am feeling bored  | 3 | `I am feeling bored`, `I am feeling bored `, `i am feeling bored` |
| i am feeling cold | 3 | `I am feeling cold`, `I am feeling cold `, `i am feeling cold` |
| i do not mean it | 3 | `I do not mean it`, `i do not mean it`, `i_don't_mean_it` |
| i am fine thank you sir | 3 | `Iam fine. Thank you sir`, `i am fine thank you sir`, `i am fine. thank you sir` |
| why are you crying | 3 | `Why are you crying`, `Why are you crying?`, `why are you crying` |
| You are welcome | 3 | `You are welcome`, `You are welcome `, `you are welcome` |
| can i help you | 2 | `Can I help you`, `can i help you` |
| can you repeat that please | 2 | `Can you repeat that please`, `can you repeat that please` |
| comb your hair | 2 | `Comb your hair`, `comb your hair` |
| congratulations | 2 | `Congratulations`, `congratulations` |
| could you please talk slower | 2 | `Could you please talk slower`, `could you please talk slower` |
| do me a favour | 2 | `Do me a favour`, `do me a favour` |

## 4. Normalization Rules Applied
The following transformations were automated in the pipeline:
1. **Case Normalization:** Convering all text to lowercase.
2. **Punctuation Removal:** Removing commas, periods, and special characters.
3. **Contraction Expansion:** Normalizing `don't` -> `do not`, `iam` -> `i am`, etc.
4. **Underscore Handling:** Replacing `_` with spaces for filesystem consistency.
5. **Whitespace Trimming:** Removing leading/trailing and duplicate internal spaces.

## 5. File Inventory
- `results/preprocessing_mapping.csv`: Full line-by-line mapping of all variations.
- `results/preprocessing_report.md`: This summary document.
