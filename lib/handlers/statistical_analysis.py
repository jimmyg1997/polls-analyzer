import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LogisticRegression
from statsmodels.multivariate.manova import MANOVA

class DescriptiveStatisticsAnalyzer:
    """Performs basic descriptive statistics."""
    @staticmethod
    def analyze(data):
        return data.describe().T

class ChiSquareAnalyzer:
    """Performs Chi-Square test for independence."""
    @staticmethod
    def analyze(data, col1, col2):
        contingency_table = pd.crosstab(data[col1], data[col2])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        return {"Chi2": chi2, "p-value": p}

class ANOVAAnalyzer:
    """Performs One-Way ANOVA test."""
    @staticmethod
    def analyze(data, dependent_var, group_var):
        groups = [group[dependent_var].dropna() for name, group in data.groupby(group_var)]
        f_stat, p_value = stats.f_oneway(*groups)
        return {"F-statistic": f_stat, "p-value": p_value}

class NonParametricTests:
    """Performs Mann-Whitney U Test and Kruskal-Wallis Test."""
    @staticmethod
    def mann_whitney(data, col, group):
        group1 = data[data[group] == data[group].unique()[0]][col]
        group2 = data[data[group] == data[group].unique()[1]][col]
        u_stat, p_value = stats.mannwhitneyu(group1, group2)
        return {"U-statistic": u_stat, "p-value": p_value}

    @staticmethod
    def kruskal_wallis(data, col, group):
        groups = [group[col] for name, group in data.groupby(group)]
        h_stat, p_value = stats.kruskal(*groups)
        return {"H-statistic": h_stat, "p-value": p_value}

class CorrelationAnalyzer:
    """Performs Spearman's Rank Correlation test."""
    @staticmethod
    def analyze(data, col1, col2):
        corr, p_value = stats.spearmanr(data[col1], data[col2])
        return {"Correlation": corr, "p-value": p_value}

class LogisticRegressionAnalyzer:
    """Performs Logistic Regression analysis."""
    @staticmethod
    def analyze(data, dependent_var, independent_vars):
        model = LogisticRegression()
        model.fit(data[independent_vars], data[dependent_var])
        return {"Coefficients": model.coef_, "Intercept": model.intercept_}

class FactorAnalysisAnalyzer:
    """Performs Factor Analysis to identify latent variables."""
    @staticmethod
    def analyze(data, n_factors):
        fa = FactorAnalysis(n_components=n_factors)
        fa.fit(data)
        return {"Components": fa.components_}

class ClusterAnalysis:
    """Performs Cluster Analysis using K-Means."""
    @staticmethod
    def analyze(data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data)
        return {"Labels": kmeans.labels_}

class MANOVAAnalyzer:
    """Performs Multivariate Analysis of Variance (MANOVA)."""
    @staticmethod
    def analyze(data, dependent_vars, independent_var):
        formula = " + ".join(dependent_vars) + " ~ " + independent_var
        manova = MANOVA.from_formula(formula, data)
        return manova.mv_test()
