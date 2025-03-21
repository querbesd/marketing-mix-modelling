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
    x_input_transformed,
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

        ax.set_xlabel(f"{channel} transformed spend")
        ax.set_ylabel(f"SHAP Value")

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

def visualize_roas(roas_results, countries=None, channels=None, plot_type='bar'):
        """
        Visualize ROAS results

        Parameters:
        -----------
        roas_results : dict
            Dictionary of ROAS results
        countries : list, optional
            List of countries to visualize, defaults to all
        channels : list, optional
            List of channels to visualize, defaults to all
        plot_type : str, optional
            Type of plot to create ('bar' or 'heatmap')

        Returns:
        --------
        matplotlib.figure.Figure
            The matplotlib figure object
        """

        # If no countries specified, use all
        if countries is None:
            countries = list(roas_results.keys())

        # Prepare data for plotting
        plot_data = []

        for country in countries:
            # If no channels specified, use all except 'overall'
            country_channels = channels if channels else [ch for ch in roas_results[country].keys() if ch != 'overall']

            for channel in country_channels:
                # Get ROAS value
                roas_value = roas_results[country][channel].get('roas',
                            roas_results[country][channel].get('incremental_roas', np.nan))

                # Add to plot data
                plot_data.append({
                    'Country': country,
                    'Channel': channel,
                    'ROAS': roas_value
                })

        # Convert to DataFrame
        df = pd.DataFrame(plot_data)

        # Create appropriate plot
        if plot_type == 'heatmap':
            # Create heatmap
            plt.figure(figsize=(10, 8))
            pivot_data = df.pivot(index='Country', columns='Channel', values='ROAS')
            ax = sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', center=1.0, fmt='.2f',
                            linewidths=0.5, cbar_kws={'label': 'ROAS'})
            plt.title('ROAS by Country and Channel', fontsize=16)
            plt.tight_layout()

        else:  # Default to bar plot
            # Create grouped bar plot
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(x='Channel', y='ROAS', hue='Country', data=df)



            # Add labels and title
            plt.xlabel('Channel', fontsize=14)
            plt.ylabel('ROAS', fontsize=14)
            plt.title('ROAS by Channel', fontsize=16)

            # Add value labels on bars
            for i, bar in enumerate(ax.patches):
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.05,
                        f'{height:.0f}',
                        ha='center', va='bottom'
                    )

            plt.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()

        return plt.gcf()

def calculate_incremental_roas(model, X, media_columns, data_indices, verbose=True):
    """Calculate incremental ROAS using model's built-in zero-out functionality"""

    # Get data for specified indices
    X_data = X.iloc[data_indices] if hasattr(data_indices, '__iter__') else X.loc[data_indices]


    try:
        baseline_prediction = model.predict(test_indices=data_indices)
        if verbose:
            print(f"Generated baseline predictions, shape: {baseline_prediction.shape}")
    except Exception as e:
        if verbose:
            print(f"Error using model's predict method: {e}")
        raise ValueError("Baseline prediction required but couldn't be generated")

    # Initialize results
    roas_results = {}

    # For each channel, zero it out and calculate ROAS
    for channel in media_columns:
        if verbose:
            print(f"  Processing channel: {channel}")

        # Store original spend data
        original_spend = X_data[channel].values
        total_spend = np.sum(original_spend)

        if total_spend <= 0:
            if verbose:
                print(f"  Skipping {channel} - no spend in selected timeframe")
            roas_results[channel] = {
                'incremental_contribution': 0.0,
                'total_spend': 0.0,
                'incremental_roas': np.nan
            }
            continue

        try:
            # Get prediction with channel zeroed out
            zeroed_prediction = model.predict(
                test_indices=data_indices,
                zero_media_columns=[channel]
            )

            # Calculate incremental contribution
            incremental_contribution = np.sum(baseline_prediction - zeroed_prediction)

            # Calculate ROAS
            if total_spend > 0:
                roas = incremental_contribution / total_spend
            else:
                roas = np.nan

            if verbose:
                print(f"    Baseline sum: {np.sum(baseline_prediction):.2f}")
                print(f"    Zeroed-out sum: {np.sum(zeroed_prediction):.2f}")
                print(f"    Incremental contribution: {incremental_contribution:.2f}")
                print(f"    Total spend: {total_spend:.2f}")
                print(f"    Incremental ROAS: {roas:.2f}")

            # Store results
            roas_results[channel] = {
                'incremental_contribution': float(incremental_contribution),
                'total_spend': float(total_spend),
                'incremental_roas': float(roas),
                'baseline_prediction_sum': float(np.sum(baseline_prediction)),
                'zeroed_prediction_sum': float(np.sum(zeroed_prediction))
            }

        except Exception as e:
            if verbose:
                print(f"  Error calculating ROAS for {channel}: {e}")
                import traceback
                traceback.print_exc()

            roas_results[channel] = {
                'incremental_contribution': np.nan,
                'total_spend': float(total_spend),
                'incremental_roas': np.nan
            }

    return roas_results


def compare_all_models_roas(results, X, media_columns, data_indices, data_type='test', verbose=True):
    """
    Compare incremental ROAS across all models.

    Parameters:
    -----------
    results : dict
        Dictionary of model results from run_mmm_analysis
    X : DataFrame
        Feature data
    media_columns : list
        List of media channel columns
    data_indices : list or array-like
        Indices of data to use for calculation
    data_type : str, optional
        Type of data being analyzed ('test' or 'train')
    verbose : bool, optional
        Whether to print detailed output

    Returns:
    --------
    dict
        Dictionary of ROAS results by model and channel
    """

    # Extract models from results
    models_dict = results['models']

    # Initialize dictionary to store ROAS results
    all_roas_results = {}

    # Calculate ROAS for each model
    for model_name, model in models_dict.items():
        if verbose:
            print(f"\nCalculating incremental ROAS for {model_name}...")

        try:
            # Calculate ROAS for this model
            model_roas = calculate_incremental_roas(
                model=model,
                X=X,
                media_columns=media_columns,
                data_indices=data_indices,
                verbose=verbose
            )

            # Store results
            all_roas_results[model_name] = model_roas

        except Exception as e:
            if verbose:
                print(f"Error calculating ROAS for {model_name}: {e}")
                import traceback
                traceback.print_exc()

            # Create empty results for this model
            all_roas_results[model_name] = {
                channel: {
                    'incremental_contribution': np.nan,
                    'total_spend': np.nan,
                    'incremental_roas': np.nan
                } for channel in media_columns
            }

    # Visualize the comparison
    try:
        if verbose:
            print("\nVisualizing ROAS comparison across models...")
        visualize_models_roas_comparison(all_roas_results, media_columns)
    except Exception as e:
        if verbose:
            print(f"Error visualizing ROAS comparison: {e}")

    return all_roas_results


def visualize_models_roas_comparison(all_roas_results, media_columns):
    """
    Visualize ROAS comparison across all models.

    Parameters:
    -----------
    all_roas_results : dict
        Dictionary with ROAS results by model and channel
    media_columns : list
        List of media channel columns
    """

    # Create DataFrame for visualization
    roas_data = []

    for model_name, model_results in all_roas_results.items():
        for channel, metrics in model_results.items():
            roas_data.append({
                'Model': model_name,
                'Channel': channel,
                'ROAS': metrics.get('incremental_roas', np.nan)
            })

    df = pd.DataFrame(roas_data)

    # Check if we have any valid ROAS values
    if df.empty or df['ROAS'].isna().all():
        print("No valid ROAS values to visualize!")
        return

    # Create plot
    plt.figure(figsize=(12, 8))

    # Set up positions for grouped bars
    channels = media_columns
    models = list(all_roas_results.keys())
    x = np.arange(len(channels))
    width = 0.8 / len(models)  # Width of bars

    # Colors for different models
    colors = plt.cm.tab10.colors

    # Plot bars for each model
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]
        roas_values = []

        # Ensure correct order of channels
        for channel in channels:
            channel_roas = model_data[model_data['Channel'] == channel]['ROAS'].values
            roas_values.append(channel_roas[0] if len(channel_roas) > 0 else np.nan)

        # Plot bars, skipping NaN values
        valid_indices = ~np.isnan(roas_values)
        bar_positions = x[valid_indices] + (i - len(models)/2 + 0.5) * width

        plt.bar(
            bar_positions,
            np.array(roas_values)[valid_indices],
            width,
            label=model,
            color=colors[i % len(colors)]
        )

        # Add value labels on bars
        for j, (idx, val) in enumerate(zip(valid_indices, roas_values)):
            if idx:
                pos = x[j] + (i - len(models)/2 + 0.5) * width
                plt.text(
                    pos, val + 0.1,
                    f"{val:.0f}",
                    ha='center', va='bottom',
                    fontsize=12,
                    rotation=90 if val > 10 else 0
                )

    # Add labels and formatting
    plt.xlabel('Media Channel', fontsize=14)
    plt.ylabel('Incremental ROAS', fontsize=14)
    plt.title('Incremental ROAS Comparison Across Models', fontsize=18)
    plt.xticks(x, channels)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.legend()

    # Set reasonable y-axis limits
    valid_values = df['ROAS'].dropna().values
    if len(valid_values) > 0:
        max_val = max(valid_values)
        median_val = np.median(valid_values)

        if max_val > 5 * median_val and max_val > 20:
            # If we have outliers, cap the y-axis
            plt.ylim(0, min(20, median_val * 3))
            print(f"Note: Y-axis capped at {min(20, median_val * 3):.1f} for better visualization")
            print(f"      Some values may be higher: max={max_val:.1f}")

    plt.tight_layout()

    # Show table with values
    print("\nROAS Comparison Table:")
    pivot_table = df.pivot(index='Channel', columns='Model', values='ROAS')
    print(pivot_table.round(2))

    plt.show()

    return df
