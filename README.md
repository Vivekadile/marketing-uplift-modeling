
#  Marketing Uplift Modeling

Predicting **which customers will buy *because of* a marketing campaign** — not just who will buy — so companies spend their marketing budget only where it actually changes behaviour.

---

## 🚧 Status

**In progress.** Results will be added once experiments are complete.

---

## 📌 Problem

Most marketing campaigns send discounts, emails, or ads to every customer, or to customers a model predicts are "likely to buy."

That approach wastes money, because customers fall into four groups:

| Segment | Buys without campaign? | Buys with campaign? | Should we target? |
|---|---|---|---|
| **Persuadables** | No | Yes | ✅ Yes — the campaign changes their decision |
| **Sure Things** | Yes | Yes | ❌ No — money wasted, they would buy anyway |
| **Lost Causes** | No | No | ❌ No — money wasted, nothing changes |
| **Sleeping Dogs** | Yes | No | ❌ Never — the campaign drives them away |

A standard "likelihood to buy" model mostly finds **Sure Things**. This project instead estimates the **uplift**: the change in a customer's probability of converting *caused by* the campaign.

```
Uplift = P(conversion | treated) − P(conversion | not treated)
```

---

## 💼 Why It Matters

- **Lower cost per conversion:** budget goes only to customers whose behaviour the campaign actually changes.
- **Fewer wasted discounts:** customers who would pay full price are not given unnecessary offers.
- **Less customer annoyance:** Sleeping Dogs who react negatively to contact can be excluded.
- **Measurable ROI:** the model's value is expressed as *incremental* conversions, which is what marketing teams report.

Uplift modeling is used in e-commerce, telecom, banking, insurance, and food-delivery marketing.

---

## 📊 Dataset

### Primary: Criteo Uplift Prediction Dataset
- **Source:** Criteo AI Lab (publicly available)
- **Size:** ~14 million rows
- **Features:** 12 anonymized numerical features (`f0`–`f11`)
- **Treatment flag:** whether the user was shown the advertising campaign
- **Outcomes:** `visit` and `conversion`
- **Collected from a randomized experiment**, which is what makes causal (uplift) estimation valid

### Secondary: Hillstrom Email Marketing Dataset
- **Size:** 64,000 customers
- **Groups:** Men's email, Women's email, No email (control)
- **Outcomes:** visit, conversion, and spend over two weeks
- **Features are interpretable** (recency, purchase history, channel, zip code type), which makes it useful for explaining *who* the persuadable customers are

### Data Challenges
- **Treatment imbalance:** the large majority of Criteo users are in the treatment group, and control is much smaller.
- **Very low conversion rate:** conversions are rare, so the uplift signal is small and noisy.
- **No individual ground truth:** a customer is either treated or not, never both, so true uplift for one person can never be observed. Evaluation must work at the group level.
- **Scale:** 14 million rows requires efficient processing (Polars / sampling for experimentation).
- **Anonymized features:** Criteo features cannot be interpreted, which is why Hillstrom is used for explainability.

---

## 🔧 Approach

```
Raw Data
   ↓
Randomization Check  →  confirm treatment and control groups are comparable
   ↓
Preprocessing        →  cleaning, scaling, encoding (Hillstrom)
   ↓
Train/Test Split     →  stratified by both treatment and outcome
   ↓
Uplift Models        →  baseline → meta-learners → tree-based uplift models
   ↓
Evaluation           →  Qini, AUUC, uplift@k, budget simulation
   ↓
Segmentation         →  Persuadables / Sure Things / Lost Causes / Sleeping Dogs
   ↓
Deployment           →  FastAPI scoring + Streamlit campaign dashboard
```

### Key Steps
1. **Randomization check:** compare feature distributions between treatment and control. If the groups differ, uplift estimates are biased.
2. **Stratified splitting:** keep the treatment ratio and conversion rate the same in train and test sets.
3. **Model training:** train and compare several uplift approaches (see below).
4. **Targeting simulation:** given a fixed budget (for example, contacting only 20% of customers), measure how many *extra* conversions each model produces compared with random targeting.

---

## 🤖 Models

| Model | Type | Why it is included |
|---|---|---|
| **Random targeting** | Baseline | The minimum bar every model must beat |
| **Response model** | Baseline | Standard "likely to buy" model — shows why predicting conversion is not the same as predicting uplift |
| **S-Learner** | Meta-learner | One model with treatment as a feature; simple starting point |
| **T-Learner** | Meta-learner | Separate models for treated and control groups; uplift = difference |
| **X-Learner** | Meta-learner | Designed for unbalanced treatment/control groups, which matches this data |
| **Class Transformation** | Direct uplift | Converts the problem into a single classification task |
| **Uplift Random Forest / Causal Forest** | Tree-based | Splits data directly to maximize differences in treatment effect |

Base learners inside meta-learners: **LightGBM** (fast on large tabular data) and **Logistic Regression** (interpretable reference).

**Libraries:** `scikit-uplift`, `CausalML`, `EconML`, `LightGBM`, `scikit-learn`

---

## 📏 Evaluation Metrics

Accuracy and ROC-AUC are **not** used as primary metrics: they measure how well a model predicts conversion, not how well it identifies customers the campaign actually influences.

| Metric | What it measures |
|---|---|
| **Qini Coefficient** | How much better the model ranks customers by uplift compared with random targeting |
| **AUUC** (Area Under the Uplift Curve) | Overall uplift gained as more customers are targeted |
| **Uplift@k** | Actual uplift among the top k% of customers ranked by the model (e.g., top 10%, 30%) |
| **Incremental conversions** | Extra conversions gained versus random targeting for the same budget |
| **Campaign ROI simulation** | Profit after campaign cost, at different targeting budgets |

---

## 📈 Results

*To be updated after experiments.*

| Model | Qini Coefficient | AUUC | Uplift@30% |
|---|---|---|---|
| Random targeting | – | – | – |
| Response model | – | – | – |
| S-Learner | – | – | – |
| T-Learner | – | – | – |
| X-Learner | – | – | – |
| Uplift Random Forest | – | – | – |

---

## 📁 Project Structure

```
marketing-uplift-modeling/
├── data/
│   ├── raw/                 # Original datasets (not committed)
│   └── processed/           # Cleaned data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_randomization_check.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data/                # Loading and preprocessing
│   ├── models/              # Uplift model training
│   ├── evaluation/          # Qini, AUUC, uplift@k
│   └── api/                 # FastAPI app
├── dashboard/               # Streamlit app
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/coder-zen/marketing-uplift-modeling.git
cd marketing-uplift-modeling

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train models
python src/models/train.py

# Start the API
uvicorn src.api.main:app --reload

# Start the dashboard
streamlit run dashboard/app.py
```

---

## 🚀 Deployment

- **FastAPI:** a `/predict` endpoint that takes customer features and returns an uplift score and customer segment. A batch endpoint scores a full customer list, since campaigns are planned in batches rather than in real time.
- **Streamlit dashboard:** a budget slider ("contact top X% of customers") that shows expected incremental conversions and ROI for that budget, plus the size of each customer segment.
- **MLflow:** experiment tracking and model comparison.
- **Docker:** consistent environment for the API and dashboard.
- **GitHub Actions:** automated tests on every push.

---

## 🧗 Challenges & Learnings

*To be updated during development.*

Expected areas:
- Evaluating a model when individual uplift can never be directly observed
- Handling a highly unbalanced treatment/control split
- Extracting a reliable signal from very rare conversion events
- Processing ~14 million rows efficiently

---

## 🔮 Future Work

- Multi-treatment uplift (choosing *which* offer to send, not just whether to send one)
- Cost-sensitive targeting where each customer has a different contact cost
- Uplift modeling on observational (non-randomized) data using propensity scores

---

## 👤 Author

**Zen** — [GitHub: coder-zen](https://github.com/coder-zen)