# Fashion Recommendation & Classification System

**BSDC-DEVDP-28A · Introduction to Artificial Intelligence · SRH Berlin University of Applied Sciences**

## Problem

Online fashion retailers typically list tens of thousands of products — far more than any shopper can browse manually. A shopper who knows roughly what they want (e.g. a black jacket for winter, something casual) still has to filter manually through categories that don't map onto how people actually describe clothing. This project builds an AI-based system that takes simple user preferences (gender, colour, season, usage) and returns matching items or complete outfits from a real product catalogue.

## Data

- **Dataset:** [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) (Kaggle, `paramaggarwal`)
- **Size:** 44,424 products, 10 attributes (`gender`, `masterCategory`, `subCategory`, `articleType`, `baseColour`, `season`, `year`, `usage`, `productDisplayName`, `id`)
- **Cleaning:** rows with missing `baseColour`, `season`, `year`, `usage`, or `productDisplayName` are dropped (~0.9% of rows removed)

## Approach

Two complementary components, built on the same underlying feature set:

**1. Classification** — three supervised models predict a product's `articleType` from its other attributes:
| Model | Accuracy |
|---|---|
| Decision Tree | 76.50% |
| K-Nearest Neighbors (k=5) | 73.74% |
| Random Forest (100 trees) | 76.72% |

**2. Recommendation** — a `NearestNeighbors` similarity search (not a classifier) returns the *k* most similar real products to a user's stated preferences. An **Outfit Builder** extends this to assemble a complete outfit (top + bottom + shoes) that share the same colour/season/usage profile, filtering by season first to avoid weather-inappropriate combinations.

## Key Findings

- **The three classifiers converge within ~3 points of each other (73.7–76.7%) because `subCategory` alone carries most of the predictive signal.** Isolating features showed `subCategory` alone reaches 56.22% accuracy; all other features combined (without it) reach only 54.56%. This reveals a hierarchy (`masterCategory → subCategory → articleType`) that sets a hard ceiling no algorithm can exceed — most confusion is concentrated between visually/functionally similar pairs like Tshirts and Shirts, which are often identical across every other feature we have.
- **Season labels in the source data are unreliable.** Manual inspection found 145 of 913 Flip Flops products (16%) labeled `season = Winter`, despite being warm-weather footwear — a labeling inconsistency inherited from the original catalogue, not introduced by our models.
- **We identified and rejected a data-leakage shortcut.** Including word features from `productDisplayName` pushed classification accuracy to 94.71%, but this only works because product titles often contain the answer verbatim (e.g. "...Black Shirt" → `Shirts`). We excluded this feature as it would not generalize to real-world product intake.

## Repository Structure

```
├── fashion_ai_project.py    # Full analysis: EDA, cleaning, classification, evaluation, recommender
├── app.py                   # Streamlit demo app (Item Recommender + Outfit Builder)
├── requirements.txt
├── styles.csv                # Dataset (place here; see Setup)
└── images/                   # Product photos (not included; see Setup)
```

## Setup

```bash
pip install -r requirements.txt
```

1. Download `styles.csv` from the [Kaggle dataset above](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) and place it in this folder.
2. Download the accompanying `images/` folder from the same dataset page and place it in this folder (not tracked in this repo due to size).
3. Run the analysis: open `fashion_ai_project.py` in Colab/Jupyter, or run cell-by-cell.
4. Run the demo app:
   ```bash
   streamlit run app.py
   ```

## Limitations & Ethics

See `ETHICS.md` for a full reflection on potential harms, sources of bias, and what a user of this system should be aware of before relying on its output.

## Team

Mara, Rayan, [add name] — Summer 2026
