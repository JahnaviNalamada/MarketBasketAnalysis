import pandas as pd
from apyori import apriori

def generate_rules(file_path, min_support, min_confidence, min_lift):
    # Load the dataset
    st_df = pd.read_csv(file_path, header=None, encoding='utf-8', dtype=str)

    # Convert DataFrame into a list of transactions
    transactions = []
    for i in range(len(st_df)):
        transactions.append([str(st_df.values[i, j]) for j in range(st_df.shape[1]) if str(st_df.values[i, j]) != 'nan'])

    # Apply Apriori algorithm
    association_rules = apriori(
        transactions, 
        min_support=min_support, 
        min_confidence=min_confidence, 
        min_lift=min_lift, 
        min_length=2
    )
    association_results = list(association_rules)

    # Extract and format rules
    recommendations = []
    for item in association_results:
        pair = item[0]
        items = [x for x in pair]
        if len(items) >= 2:  # Only consider rules with exactly two items
            recommendations.append({
                "Rule": f"{items[0]} -> {items[1]}",
                "Antecedent": items[0],
                "Consequent": items[1],
                "Support": item[1],
                "Confidence": item[2][0][2],
                "Lift": item[2][0][3]
            })

    return pd.DataFrame(recommendations)

