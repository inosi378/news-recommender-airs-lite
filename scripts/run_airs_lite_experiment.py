import argparse
import json
from pathlib import Path
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


NEWS_COLS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

BEHAVIOR_COLS = [
    "impression_id",
    "user_id",
    "time",
    "history",
    "impressions",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_rows", type=int, default=5000)
    parser.add_argument("--dev_rows", type=int, default=5000)
    parser.add_argument("--exp_name", type=str, default="small")

    parser.add_argument("--max_history", type=int, default=50)
    parser.add_argument("--max_clicks_per_user", type=int, default=50)
    parser.add_argument("--min_pair_count", type=int, default=2)
    parser.add_argument("--max_features", type=int, default=50000)

    parser.add_argument("--run_ablation", action="store_true")

    return parser.parse_args()


def resolve_base_dir():
    current = Path.cwd()

    if (current / "data").exists() and (current / "scripts").exists():
        return current

    if current.name == "scripts":
        return current.parent

    raise FileNotFoundError(
        "Project root not found. Run this script from the project root directory."
    )


def load_mind_data(base_dir):
    raw_dir = base_dir / "data" / "raw"
    train_dir = raw_dir / "MINDsmall_train"
    dev_dir = raw_dir / "MINDsmall_dev"

    train_news = pd.read_csv(train_dir / "news.tsv", sep="\t", names=NEWS_COLS)
    dev_news = pd.read_csv(dev_dir / "news.tsv", sep="\t", names=NEWS_COLS)

    train_behaviors = pd.read_csv(train_dir / "behaviors.tsv", sep="\t", names=BEHAVIOR_COLS)
    dev_behaviors = pd.read_csv(dev_dir / "behaviors.tsv", sep="\t", names=BEHAVIOR_COLS)

    return train_news, dev_news, train_behaviors, dev_behaviors


def parse_impressions(impression_str):
    parsed = []

    if pd.isna(impression_str):
        return parsed

    for item in str(impression_str).split():
        news_id, label = item.rsplit("-", 1)
        parsed.append((news_id, int(label)))

    return parsed


def build_samples(behaviors_df, n_rows=None):
    rows = []

    if n_rows is None or n_rows < 0:
        target_df = behaviors_df
    else:
        target_df = behaviors_df.head(n_rows)

    for _, row in tqdm(target_df.iterrows(), total=len(target_df), desc="building samples"):
        history = row["history"]
        if pd.isna(history):
            history = ""

        for news_id, label in parse_impressions(row["impressions"]):
            rows.append({
                "impression_id": row["impression_id"],
                "user_id": row["user_id"],
                "history": history,
                "candidate_news": news_id,
                "label": label,
            })

    return pd.DataFrame(rows)


def build_news_all(train_news, dev_news):
    news_all = pd.concat([train_news, dev_news], ignore_index=True)
    news_all = news_all.drop_duplicates("news_id").reset_index(drop=True)

    news_all["title"] = news_all["title"].fillna("")
    news_all["abstract"] = news_all["abstract"].fillna("")
    news_all["category"] = news_all["category"].fillna("")
    news_all["subcategory"] = news_all["subcategory"].fillna("")
    news_all["text"] = news_all["title"] + " " + news_all["abstract"]

    return news_all[[
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "text",
    ]]


def parse_history(history, max_history=50):
    if pd.isna(history) or history == "":
        return []

    return str(history).split()[-max_history:]


def build_user_click_sets(samples, max_clicks_per_user=50):
    user_clicks = {}

    for user_id, group in tqdm(samples.groupby("user_id"), desc="building user click sets"):
        clicks = []

        for hist in group["history"].dropna().unique():
            if hist != "":
                clicks.extend(str(hist).split())

        positive_clicks = group.loc[group["label"] == 1, "candidate_news"].tolist()
        clicks.extend(positive_clicks)

        seen = set()
        unique_clicks = []

        for news_id in clicks:
            if news_id not in seen:
                unique_clicks.append(news_id)
                seen.add(news_id)

        user_clicks[user_id] = set(unique_clicks[-max_clicks_per_user:])

    return user_clicks


def build_npmi_dict(user_clicks, min_pair_count=2):
    item_counts = Counter()
    pair_counts = Counter()

    users = list(user_clicks.keys())
    n_users = len(users)

    for user in tqdm(users, desc="counting item pairs"):
        items = sorted(list(user_clicks[user]))

        if len(items) < 2:
            item_counts.update(items)
            continue

        item_counts.update(items)

        for a, b in combinations(items, 2):
            pair_counts[(a, b)] += 1

    npmi_dict = {}

    for (a, b), count_ab in tqdm(pair_counts.items(), desc="computing NPMI"):
        if count_ab < min_pair_count:
            continue

        count_a = item_counts[a]
        count_b = item_counts[b]

        p_a = count_a / n_users
        p_b = count_b / n_users
        p_ab = count_ab / n_users

        if p_a <= 0 or p_b <= 0 or p_ab <= 0 or p_ab >= 1:
            continue

        pmi = np.log(p_ab / (p_a * p_b))
        npmi = pmi / (-np.log(p_ab))

        # 추천 feature에서는 음수 관계를 0으로 처리
        npmi = max(0.0, float(npmi))

        npmi_dict[(a, b)] = npmi
        npmi_dict[(b, a)] = npmi

    return npmi_dict, item_counts


def max_npmi_score(history_ids, candidate_news, npmi_dict):
    if not history_ids:
        return 0.0

    return max(
        npmi_dict.get((hist_news, candidate_news), 0.0)
        for hist_news in history_ids
    )


def build_popularity_score(train_samples):
    click_counts = train_samples[train_samples["label"] == 1].groupby("candidate_news").size()

    if len(click_counts) == 0:
        return {}

    max_click = click_counts.max()

    return {
        news_id: np.log1p(count) / np.log1p(max_click)
        for news_id, count in click_counts.items()
    }


def make_feature_table(
    samples,
    name,
    news_all,
    news_tfidf,
    news_id_to_idx,
    npmi_dict,
    popularity_score,
    max_history=50,
):
    news_id_to_category = dict(zip(news_all["news_id"], news_all["category"]))
    news_id_to_subcategory = dict(zip(news_all["news_id"], news_all["subcategory"]))

    rows = []

    for impression_id, group in tqdm(samples.groupby("impression_id"), desc=f"making features: {name}"):
        history = group["history"].iloc[0]
        history_ids = parse_history(history, max_history=max_history)

        hist_categories = [
            news_id_to_category.get(news_id)
            for news_id in history_ids
            if news_id in news_id_to_category
        ]
        hist_categories = [x for x in hist_categories if pd.notna(x) and x != ""]
        category_counter = Counter(hist_categories)
        category_total = len(hist_categories)

        hist_subcategories = [
            news_id_to_subcategory.get(news_id)
            for news_id in history_ids
            if news_id in news_id_to_subcategory
        ]
        hist_subcategories = [x for x in hist_subcategories if pd.notna(x) and x != ""]
        subcategory_counter = Counter(hist_subcategories)
        subcategory_total = len(hist_subcategories)

        hist_indices = [
            news_id_to_idx[news_id]
            for news_id in history_ids
            if news_id in news_id_to_idx
        ]

        if len(hist_indices) > 0:
            user_vec = news_tfidf[hist_indices].mean(axis=0)
        else:
            user_vec = None

        for _, row in group.iterrows():
            candidate = row["candidate_news"]

            npmi_score = max_npmi_score(history_ids, candidate, npmi_dict)

            candidate_category = news_id_to_category.get(candidate, "")
            if category_total > 0 and candidate_category != "":
                category_pref = category_counter.get(candidate_category, 0) / category_total
            else:
                category_pref = 0.0

            candidate_subcategory = news_id_to_subcategory.get(candidate, "")
            if subcategory_total > 0 and candidate_subcategory != "":
                subcategory_pref = subcategory_counter.get(candidate_subcategory, 0) / subcategory_total
            else:
                subcategory_pref = 0.0

            if user_vec is not None and candidate in news_id_to_idx:
                candidate_vec = news_tfidf[news_id_to_idx[candidate]]
                tfidf_score = float(user_vec @ candidate_vec.T)
            else:
                tfidf_score = 0.0

            pop_score = popularity_score.get(candidate, 0.0)

            rows.append({
                "impression_id": row["impression_id"],
                "user_id": row["user_id"],
                "history": row["history"],
                "candidate_news": candidate,
                "label": int(row["label"]),
                "npmi_score": npmi_score,
                "category_pref": category_pref,
                "subcategory_pref": subcategory_pref,
                "tfidf_score": tfidf_score,
                "popularity_score": pop_score,
            })

    return pd.DataFrame(rows)


def mrr_score(labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)

    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]

    positive_indices = np.where(sorted_labels == 1)[0]

    if len(positive_indices) == 0:
        return np.nan

    return 1.0 / (positive_indices[0] + 1)


def dcg_at_k(labels, scores, k):
    labels = np.array(labels)
    scores = np.array(scores)

    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order][:k]

    gains = sorted_labels
    discounts = np.log2(np.arange(2, len(sorted_labels) + 2))

    return np.sum(gains / discounts)


def ndcg_at_k(labels, scores, k):
    dcg = dcg_at_k(labels, scores, k)

    ideal_scores = np.array(labels)
    idcg = dcg_at_k(labels, ideal_scores, k)

    if idcg == 0:
        return np.nan

    return dcg / idcg


def evaluate_by_impression(scored_df):
    metrics = []

    for impression_id, group in scored_df.groupby("impression_id"):
        labels = group["label"].values
        scores = group["score"].values

        if len(np.unique(labels)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(labels, scores)

        metrics.append({
            "impression_id": impression_id,
            "AUC": auc,
            "MRR": mrr_score(labels, scores),
            "nDCG@5": ndcg_at_k(labels, scores, k=5),
            "nDCG@10": ndcg_at_k(labels, scores, k=10),
        })

    metrics_df = pd.DataFrame(metrics)
    result = metrics_df[["AUC", "MRR", "nDCG@5", "nDCG@10"]].mean(skipna=True)

    return result, metrics_df


def train_and_evaluate(train_features, dev_features, feature_cols, model_name):
    X_train = train_features[feature_cols].fillna(0)
    y_train = train_features["label"].astype(int)
    X_dev = dev_features[feature_cols].fillna(0)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    scored_dev = dev_features.copy()
    scored_dev["score"] = model.predict_proba(X_dev)[:, 1]

    result, metrics_df = evaluate_by_impression(scored_dev)

    result_row = {
        "model": model_name,
        "AUC": result["AUC"],
        "MRR": result["MRR"],
        "nDCG@5": result["nDCG@5"],
        "nDCG@10": result["nDCG@10"],
    }

    coef_table = pd.DataFrame({
        "feature": feature_cols,
        "coef": model.named_steps["lr"].coef_[0],
        "model": model_name,
    }).sort_values("coef", ascending=False)

    return result_row, metrics_df, scored_dev, coef_table


def main():
    args = parse_args()
    base_dir = resolve_base_dir()

    processed_dir = base_dir / "data" / "processed"
    output_dir = base_dir / "outputs"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("BASE_DIR:", base_dir)
    print("Experiment:", args.exp_name)

    config = {
        "exp_name": args.exp_name,
        "train_rows": args.train_rows,
        "dev_rows": args.dev_rows,
        "max_history": args.max_history,
        "max_clicks_per_user": args.max_clicks_per_user,
        "min_pair_count": args.min_pair_count,
        "max_features": args.max_features,
        "run_ablation": args.run_ablation,
    }

    with open(output_dir / f"run_config_{args.exp_name}.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n[1] Loading MIND data")
    train_news, dev_news, train_behaviors, dev_behaviors = load_mind_data(base_dir)

    print("train_news:", train_news.shape)
    print("dev_news:", dev_news.shape)
    print("train_behaviors:", train_behaviors.shape)
    print("dev_behaviors:", dev_behaviors.shape)

    print("\n[2] Building samples")
    train_n_rows = None if args.train_rows < 0 else args.train_rows
    dev_n_rows = None if args.dev_rows < 0 else args.dev_rows

    train_samples = build_samples(train_behaviors, train_n_rows)
    dev_samples = build_samples(dev_behaviors, dev_n_rows)

    train_samples["history"] = train_samples["history"].fillna("")
    dev_samples["history"] = dev_samples["history"].fillna("")

    print("train_samples:", train_samples.shape)
    print("dev_samples:", dev_samples.shape)

    train_samples.to_csv(processed_dir / f"train_samples_{args.exp_name}.csv", index=False, encoding="utf-8-sig")
    dev_samples.to_csv(processed_dir / f"dev_samples_{args.exp_name}.csv", index=False, encoding="utf-8-sig")

    print("\n[3] Building news table")
    news_all = build_news_all(train_news, dev_news)
    news_all.to_csv(processed_dir / f"news_all_{args.exp_name}.csv", index=False, encoding="utf-8-sig")

    print("news_all:", news_all.shape)

    print("\n[4] Building TF-IDF vectors")
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        stop_words="english",
    )

    news_tfidf = vectorizer.fit_transform(news_all["text"].fillna(""))
    news_id_to_idx = {
        news_id: idx
        for idx, news_id in enumerate(news_all["news_id"])
    }

    print("news_tfidf:", news_tfidf.shape)

    print("\n[5] Building NPMI dictionary")
    user_clicks = build_user_click_sets(
        train_samples,
        max_clicks_per_user=args.max_clicks_per_user,
    )

    npmi_dict, item_counts = build_npmi_dict(
        user_clicks,
        min_pair_count=args.min_pair_count,
    )

    print("users:", len(user_clicks))
    print("items:", len(item_counts))
    print("NPMI pairs:", len(npmi_dict))

    print("\n[6] Building popularity proxy")
    popularity_score = build_popularity_score(train_samples)
    print("popular items:", len(popularity_score))

    print("\n[7] Building feature tables")
    train_features = make_feature_table(
        train_samples,
        name="train",
        news_all=news_all,
        news_tfidf=news_tfidf,
        news_id_to_idx=news_id_to_idx,
        npmi_dict=npmi_dict,
        popularity_score=popularity_score,
        max_history=args.max_history,
    )

    dev_features = make_feature_table(
        dev_samples,
        name="dev",
        news_all=news_all,
        news_tfidf=news_tfidf,
        news_id_to_idx=news_id_to_idx,
        npmi_dict=npmi_dict,
        popularity_score=popularity_score,
        max_history=args.max_history,
    )

    train_features.to_csv(processed_dir / f"train_airs_lite_features_{args.exp_name}.csv", index=False, encoding="utf-8-sig")
    dev_features.to_csv(processed_dir / f"dev_airs_lite_features_{args.exp_name}.csv", index=False, encoding="utf-8-sig")

    print("train_features:", train_features.shape)
    print("dev_features:", dev_features.shape)

    print("\n[8] Training and evaluating models")

    model_settings = {
        "TF-IDF baseline": ["tfidf_score"],
        "AiRS-lite full": [
            "npmi_score",
            "category_pref",
            "subcategory_pref",
            "tfidf_score",
            "popularity_score",
        ],
        "AiRS-lite revised": [
            "npmi_score",
            "category_pref",
            "subcategory_pref",
            "tfidf_score",
        ],
    }

    if args.run_ablation:
        model_settings.update({
            "w/o NPMI": [
                "category_pref",
                "subcategory_pref",
                "tfidf_score",
                "popularity_score",
            ],
            "w/o popularity": [
                "npmi_score",
                "category_pref",
                "subcategory_pref",
                "tfidf_score",
            ],
            "w/o category features": [
                "npmi_score",
                "tfidf_score",
                "popularity_score",
            ],
            "w/o TF-IDF": [
                "npmi_score",
                "category_pref",
                "subcategory_pref",
                "popularity_score",
            ],
            "only NPMI": ["npmi_score"],
            "only popularity": ["popularity_score"],
        })

    result_rows = []
    all_coef_tables = []

    for model_name, feature_cols in model_settings.items():
        print("running model:", model_name, feature_cols)

        result_row, metrics_df, scored_dev, coef_table = train_and_evaluate(
            train_features,
            dev_features,
            feature_cols,
            model_name,
        )

        result_rows.append(result_row)
        all_coef_tables.append(coef_table)

        safe_name = model_name.replace(" ", "_").replace("/", "_")

        metrics_df.to_csv(
            output_dir / f"metrics_{safe_name}_{args.exp_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        scored_dev.to_csv(
            output_dir / f"scored_dev_{safe_name}_{args.exp_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    results_df = pd.DataFrame(result_rows)
    coef_df = pd.concat(all_coef_tables, ignore_index=True)

    results_df.to_csv(output_dir / f"results_{args.exp_name}.csv", index=False, encoding="utf-8-sig")
    coef_df.to_csv(output_dir / f"coef_{args.exp_name}.csv", index=False, encoding="utf-8-sig")

    print("\n=== RESULTS ===")
    print(results_df)

    print("\nSaved:")
    print(output_dir / f"results_{args.exp_name}.csv")
    print(output_dir / f"coef_{args.exp_name}.csv")


if __name__ == "__main__":
    main()