"""
Assignment 1: Market Basket Analysis using Apriori Algorithm
Author: Jit Sarkar
Repository: DWBDA-Assignment-1-Market-Basket-Analysis
Description: Discovers frequent itemsets and strong association rules in
             computer accessories retail transaction data.
"""

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def main():
    print("=" * 70)
    print("DWBDA Assignment 1: Market Basket Analysis (Apriori Algorithm)")
    print("=" * 70)
    
    # 1. Load dataset
    csv_file = "computer_accessories_transactions.csv"
    df_raw = pd.read_csv(csv_file)
    print(f"\nLoaded {len(df_raw)} transactions from '{csv_file}'.")
    print("\nSample raw transactions:")
    print(df_raw.head())

    # 2. Extract transaction item lists
    transactions = [items.split(";") for items in df_raw["Items"]]

    # 3. One-hot encoding
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    print(f"\nOne-hot encoded dataset shape: {df_encoded.shape} (Transactions x Unique Items)")
    print("Item list:", list(te.columns_))

    # 4. Apriori Frequent Itemsets (min_support = 20%)
    min_support = 0.20
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))
    frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False).reset_index(drop=True)
    
    print(f"\n--- Frequent Itemsets (min_support >= {min_support*100:.0f}%) ---")
    print(frequent_itemsets)

    # 5. Association Rules Generation (min_confidence = 50%)
    min_confidence = 0.50
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(by="lift", ascending=False).reset_index(drop=True)

    # Convert frozensets to readable strings for display
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

    display_cols = ["antecedents_str", "consequents_str", "support", "confidence", "lift", "leverage"]
    print(f"\n--- Association Rules (min_confidence >= {min_confidence*100:.0f}%, sorted by Lift) ---")
    print(rules[display_cols].to_string(index=False))

    # Save rules to CSV
    rules.to_csv("mba_rules_summary.csv", index=False)
    print("\n[SUCCESS] Rules saved to 'mba_rules_summary.csv'.")

    # 6. Visualization
    if len(rules) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Subplot 1: Support vs Confidence colored by Lift
        scatter = axes[0].scatter(
            rules["support"],
            rules["confidence"],
            c=rules["lift"],
            cmap="viridis",
            s=120,
            alpha=0.85,
            edgecolors="black"
        )
        cbar = fig.colorbar(scatter, ax=axes[0])
        cbar.set_label("Lift Metric", fontsize=11)
        axes[0].set_title("Support vs. Confidence (Colored by Lift)", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Support", fontsize=11)
        axes[0].set_ylabel("Confidence", fontsize=11)
        axes[0].grid(True, linestyle="--", alpha=0.6)

        # Subplot 2: Top Rules by Lift Bar Chart
        top_rules = rules.head(8).copy()
        rule_labels = [f"{row['antecedents_str']} -> {row['consequents_str']}" for _, row in top_rules.iterrows()]
        axes[1].barh(rule_labels, top_rules["lift"], color="#2b5c8f", edgecolor="black")
        axes[1].set_xlabel("Lift Value", fontsize=11)
        axes[1].set_title("Top Association Rules Ranked by Lift", fontsize=13, fontweight="bold")
        axes[1].invert_yaxis()
        axes[1].grid(True, axis="x", linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig("mba_visualization.png", dpi=300)
        print("[SUCCESS] Visualization saved to 'mba_visualization.png'.")
        plt.close()

if __name__ == "__main__":
    main()
