from typing import Literal, cast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from scipy.optimize import curve_fit


### ALL UTILS FUNCTIONS ###

def approx_hsgp_hyperparams(
    x,
    x_center,
    lengthscale_range: tuple[float, float],
    cov_func: Literal["expquad", "matern32", "matern52"],
) -> tuple[int, float]:
    """Use heuristics for minimum `m` and `c` values.

    Based on recommendations from Ruitort-Mayol et. al.

    In practice, you need to choose `c` large enough to handle the largest
    lengthscales, and `m` large enough to accommodate the smallest lengthscales.

    NOTE: These recommendations are based on a one-dimensional GP.

    Parameters
    ----------
    x : tensor_like
        The x values the HSGP will be evaluated over.
    x_center : tensor_like
        The center of the data.
    lengthscale_range : tuple[float, float]
        The range of the lengthscales. Should be a list with two elements [lengthscale_min, lengthscale_max].
    cov_func : Literal["expquad", "matern32", "matern52"]
        The covariance function to use. Supported options are "expquad", "matern52", and "matern32".

    Returns
    -------
    - `m` : int
        Number of basis vectors. Increasing it helps approximate smaller lengthscales, but increases computational cost.
    - `c` : float
        Scaling factor such that L = c * S, where L is the boundary of the approximation.
        Increasing it helps approximate larger lengthscales, but may require increasing m.

    Raises
    ------
    ValueError
        If either `x_range` or `lengthscale_range` is not in the correct order.

    References
    ----------
    .. [1] Ruitort-Mayol, G., Anderson, M., Solin, A., Vehtari, A. (2022).
    Practical Hilbert Space Approximate Bayesian Gaussian Processes for Probabilistic Programming
    """
    lengthscale_min, lengthscale_max = lengthscale_range
    if lengthscale_min >= lengthscale_max:
        raise ValueError(
            "The boundaries are out of order. {lengthscale_min} should be less than {lengthscale_max}"
        )

    Xs = x - x_center
    S = np.max(np.abs(Xs), axis=0)

    if cov_func.lower() == "expquad":
        a1, a2 = 3.2, 1.75

    elif cov_func.lower() == "matern52":
        a1, a2 = 4.1, 2.65

    elif cov_func.lower() == "matern32":
        a1, a2 = 4.5, 3.42

    else:
        raise ValueError(
            "Unsupported covariance function. Supported options are 'expquad', 'matern52', and 'matern32'."
        )

    c = max(a1 * (lengthscale_max / S), 1.2)
    m = int(a2 * c / (lengthscale_min / S))

    return m, c

def nrmse(y_true, y_pred):
    """Calculate Normalized Root Mean Square Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2)) / (np.max(y_true) - np.min(y_true))

def rssd(effect_share, spend_share):
    """
    Calculate RSSD decomposition metric

    Decomposition distance eliminates the majority of "bad models"
    (larger prediction error and/or unrealistic media effect where
    the smallest channel gets the most effect)
    """
    return np.sqrt(np.sum((effect_share - spend_share) ** 2))

def calculate_spend_effect_share(df_shap_values, media_channels, df_original):
    """Calculate spend and effect share for each media channel"""
    # Calculate effect share based on absolute SHAP values
    responses = pd.DataFrame(df_shap_values[media_channels].abs().sum(axis=0), columns=["effect_share"])
    response_percentages = responses / responses.sum()

    # Calculate spend share
    spends_percentages = pd.DataFrame(
        df_original[media_channels].sum(axis=0) / df_original[media_channels].sum(axis=0).sum(),
        columns=["spend_share"]
    )

    # Combine results
    spend_effect_share = pd.merge(response_percentages, spends_percentages, left_index=True, right_index=True)
    spend_effect_share = spend_effect_share.reset_index().rename(columns={"index": "media"})

    return spend_effect_share

def plot_spend_vs_effect_share(spend_effect_share, figure_size=(15, 8)):
    """Plot share of spend vs share of effect for media channels"""
    # Prepare data for plotting
    plot_data = spend_effect_share.melt(
        id_vars=["media"],
        value_vars=["spend_share", "effect_share"],
        var_name="Type",
        value_name="Share"
    )

    # Calculate RSSD
    rssd_value = rssd(
        effect_share=spend_effect_share.effect_share.values,
        spend_share=spend_effect_share.spend_share.values
    )

    # Create plot
    plt.figure(figsize=figure_size)
    ax = sns.barplot(x="media", y="Share", hue="Type", data=plot_data)

    # Add percentage labels
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.annotate(f"{height*100:.1f}%",
                   (p.get_x() + p.get_width() / 2., height),
                   ha='center', va='bottom',
                   xytext=(0, 5),
                   textcoords='offset points')

    # Add RSSD value to plot
    plt.text(
        0.05, 0.95,
        f"RSSD: {rssd_value:.3f}",
        transform=ax.transAxes,
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    )

    # Customize plot
    plt.title('Share of Spend VS Share of Effect', fontsize=16)
    plt.ylabel('Share')
    plt.xlabel('')
    plt.xticks(rotation=45)
    plt.tight_layout()

    return plt

def logistic_saturation_offset_scale(x, lam, alpha, beta):
    """
    Extended logistic saturation that can shift and scale outputs:
      g(x) = alpha + beta * ( (1 - exp(-lam*x)) / (1 + exp(-lam*x)) )
    """
    return alpha + beta * ((1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x)))


def plot_shap_vs_spend(
    df_shap_values,
    x_input_original,
    x_input_transformed,
    features,
    media_channels,
    figsize=(15, 7),
    fit_lambda=False
):
    """
    Plot SHAP values vs spend for each media channel.

    If fit_lambda=True, fit the extended logistic saturation function
      g(x) = alpha + beta * [(1 - exp(-lam*x)) / (1 + exp(-lam*x))]
    which allows capturing negative SHAP values (due to alpha < 0, etc.).
    """
    for channel in media_channels:
        if channel not in x_input_transformed.columns:
            continue

        # Calculate mean spend for non-zero values
        mean_spend = x_input_original.loc[x_input_original[channel] > 0, channel].mean()

        fig, ax = plt.subplots(figsize=figsize)
        sns.regplot(
            x=x_input_transformed[channel],
            y=df_shap_values[channel],
            label=channel,
            scatter_kws={'alpha': 0.65},
            line_kws={'color': 'C2', 'linewidth': 3},
            lowess=True,
            ax=ax
        ).set(title=f'{channel}: Spend vs Shapley')

        # Reference lines
        ax.axhline(0, linestyle="--", color="black", alpha=0.5)
        ax.axvline(mean_spend, linestyle="--", color="red", alpha=0.5,
                   label=f"Average Spend: {mean_spend:.2f}")

        if fit_lambda:
            x_data = x_input_transformed[channel].values
            y_data = df_shap_values[channel].values

            # Initial guesses for lam, alpha, beta
            # lam ~ 0.5, alpha ~ mean of y, beta ~ half the range of y
            p0 = [0.5, np.mean(y_data), (np.max(y_data) - np.min(y_data)) / 2]
            try:
                popt, _ = curve_fit(logistic_saturation_offset_scale, x_data, y_data, p0=p0)
                lam_fit, alpha_fit, beta_fit = popt

                # Sort x for smooth plotting
                x_sorted = np.sort(x_data)
                fitted_curve = logistic_saturation_offset_scale(x_sorted, lam_fit, alpha_fit, beta_fit)
                ax.plot(
                    x_sorted,
                    fitted_curve,
                    color='red',
                    linewidth=3,
                    label=(f'Extended Logistic Fit:\n'
                           f'λ={lam_fit:.2f}, α={alpha_fit:.2f}, β={beta_fit:.2f}')
                )
            except Exception as e:
                print(f"Logistic fit failed for {channel}: {e}")

        ax.set_xlabel(f"{channel} spend")
        ax.set_ylabel(f"SHAP Value for {channel}")
        plt.legend()
        plt.tight_layout()
        plt.show()


def shap_feature_importance(shap_values, data, figsize=(15, 8)):
    """Plot SHAP feature importance"""
    feature_list = data.columns

    if isinstance(shap_values, pd.DataFrame) == False:
        shap_v = pd.DataFrame(shap_values)
        shap_v.columns = feature_list
    else:
        shap_v = shap_values

    df_v = data.copy().reset_index().drop('index', axis=1)

    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i], df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list), pd.Series(corr_list)], axis=1).fillna(0)

    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns = ['Variable', 'Corr']
    corr_df['Sign'] = np.where(corr_df['Corr'] > 0, 'red', 'blue')

    # Plot it
    shap_abs = np.abs(shap_v)
    k = pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable', 'SHAP_abs']
    k2 = k.merge(corr_df, left_on='Variable', right_on='Variable', how='inner')
    k2 = k2.sort_values(by='SHAP_abs', ascending=True)
    colorlist = k2['Sign']

    plt.figure(figsize=figsize)
    ax = k2.plot.barh(x='Variable', y='SHAP_abs', color=colorlist, figsize=figsize, legend=False)
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")
    plt.tight_layout()
    plt.show()
