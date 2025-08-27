import numpy as np


class Topsis:
    def __init__(self, dataset, weights, impact):
      
        self.dataset = dataset.copy()
        self.weights = weights
        self.impact = impact
        self.nCol = self.dataset.shape[1]  # number of columns
        self.normalized_dataset = None
        self.p_sln = None
        self.n_sln = None
        self.scores = None
        self.rank = None

    def normalize(self):
        #Normalize the dataset and apply weights.
        df = self.dataset.copy()
        for i in range(1, self.nCol):  # skip first column (alternatives)
            col_sum_sq = np.sqrt(np.sum(df.iloc[:, i] ** 2))
            df.iloc[:, i] = (df.iloc[:, i] / col_sum_sq) * self.weights[i-1]
        self.normalized_dataset = df
        return df

    def calculate_ideal_solutions(self):
        #Calculate positive and negative ideal solutions based on impact.
        df = self.normalized_dataset
        p_sln = df.iloc[:, 1:].max().values  # skip first column
        n_sln = df.iloc[:, 1:].min().values
        for i, imp in enumerate(self.impact):
            if imp == '-':
                p_sln[i], n_sln[i] = n_sln[i], p_sln[i]
        self.p_sln = p_sln
        self.n_sln = n_sln
        return p_sln, n_sln

    def calculate_scores(self):
        """Compute distances and TOPSIS scores for each alternative."""
        df = self.normalized_dataset
        scores = []
        distance_positive = []
        distance_negative = []

        for i in range(len(df)):
            d_pos = np.sqrt(np.sum((self.p_sln - df.iloc[i, 1:].values) ** 2))
            d_neg = np.sqrt(np.sum((self.n_sln - df.iloc[i, 1:].values) ** 2))
            score = d_neg / (d_pos + d_neg)
            scores.append(score)
            distance_positive.append(d_pos)
            distance_negative.append(d_neg)

        self.normalized_dataset['distance_positive'] = distance_positive
        self.normalized_dataset['distance_negative'] = distance_negative
        self.normalized_dataset['Topsis_score'] = scores
        self.normalized_dataset['Rank'] = self.normalized_dataset['Topsis_score'].rank(
            method='max', ascending=False).astype(int)
        self.scores = scores
        self.rank = self.normalized_dataset['Rank'].values
        return self.normalized_dataset

    def evaluate(self):
        #Run the full TOPSIS process and return ranked DataFrame.
        self.normalize()
        self.calculate_ideal_solutions()
        result_df = self.calculate_scores()

        result_df = result_df.sort_values('Rank').reset_index(drop=True)
        return result_df

