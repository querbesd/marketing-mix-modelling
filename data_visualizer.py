import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation


class MMMVisualizer:
    """
    Class for visualizing Marketing Mix Modeling data and results.

    This class provides various plotting functions to visualize MMM data,
    including contributions from different components, ROAS by channel,
    and potential outcomes.
    """

    @staticmethod
    def plot_contributions(dataset, country_params, countries, components_to_show=None):
        """
        Plot all components including base, trend, and seasonality with improved readability.

        Parameters:
        -----------
        dataset : xarray.Dataset
            The dataset containing all variables
        country_params : dict
            Dictionary containing parameters for each country
        countries : list
            List of countries to visualize
        components_to_show : dict, optional
            Dictionary controlling which components to show in each plot
            Default shows essential components only

        Returns:
        --------
        matplotlib.figure.Figure
            The matplotlib figure object
        """
        # Default component configuration (more readable)
        if components_to_show is None:
            components_to_show = {
                'base_sales': True,
                'trend': True,
                'seasonality': True,
                'individual_channels': False,  # Set to True to see individual channels
                'total_media': True,
                'base_response': True,
                'noise_line': False,
                'noise_range': True,
                'target': True,
                'scaled_base': True,
                'promo_effects': True
            }

        # Set up plot styling for better readability
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })

        # Color palette for better distinction
        color_palette = {
            'base_sales': '#666666',
            'trend': '#1b9e77',
            'seasonality': '#7570b3',
            'total_media': '#e7298a',
            'base_response': '#66a61e',
            'noise': '#999999',
            'target': '#000000',
            'scaled_base': '#66a61e',
            'promo_effects': '#e6ab02',
            'channels': ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c']
        }

        # Create figure with improved layout
        n_countries = len(countries)
        fig, all_axes = plt.subplots(
            n_countries,
            2,
            figsize=(24, 7*n_countries),  # Wider figure, taller rows
            constrained_layout=True       # Better spacing management
        )

        if n_countries == 1:
            all_axes = all_axes.reshape(1, 2)

        # Parse dates to datetime for better x-axis formatting
        dates = pd.to_datetime(dataset.coords['date'].values)

        for idx, country in enumerate(countries):
            ax_components = all_axes[idx, 0]  # Left column for components
            ax_target = all_axes[idx, 1]      # Right column for target

            # Get country-specific data
            target = dataset.target.sel(country=country)
            base_sales = dataset.base_sales.sel(country=country)
            trend = dataset.trend.sel(country=country)
            seasonality = dataset.seasonality.sel(country=country)
            media_response_by_channel = dataset.media_response.sel(country=country)
            total_media_response = media_response_by_channel.sum(dim='channel')
            noise = dataset.noise.sel(country=country)
            promo_effects = dataset.promo_effects.sel(country=country)

            size = country_params[country]['size']
            launch_date = country_params[country].get('launch_date', 0)

            # Calculate base response for reuse
            base_response = (base_sales + trend + seasonality + total_media_response + noise)

            # LEFT PLOT: Raw Components with improved clarity
            if components_to_show['base_sales']:
                ax_components.axhline(
                    y=base_sales[0],
                    color=color_palette['base_sales'],
                    linestyle='-',
                    linewidth=2,
                    label=f'Base Sales ({base_sales[0]:.1f})'
                )

            if components_to_show['trend']:
                ax_components.plot(
                    dates,
                    trend + base_sales,
                    color=color_palette['trend'],
                    linestyle='-',
                    linewidth=2.5,
                    label='Trend + Base'
                )

            if components_to_show['seasonality']:
                ax_components.plot(
                    dates,
                    seasonality + base_sales,
                    color=color_palette['seasonality'],
                    linestyle='-',
                    linewidth=2.5,
                    label='Seasonality + Base'
                )

            # Plot individual channel contributions if requested
            if components_to_show['individual_channels']:
                channels = list(dataset.coords['channel'].values)
                for i, channel in enumerate(channels):
                    channel_response = media_response_by_channel.sel(channel=channel)
                    channel_spend = dataset.spends.sel(country=country, channel=channel)
                    active_periods = (channel_spend > 0).sum().item()
                    activity_rate = active_periods / len(dates)

                    # Use the channel color from palette, or default to a basic color
                    channel_color = color_palette['channels'][i % len(color_palette['channels'])]

                    ax_components.plot(
                        dates,
                        channel_response + base_sales,
                        color=channel_color,
                        linestyle='--',
                        linewidth=2,
                        label=f'{channel} Effect (activity={activity_rate:.1%})'
                    )

            # Plot total media response
            if components_to_show['total_media']:
                ax_components.plot(
                    dates,
                    total_media_response + base_sales,
                    color=color_palette['total_media'],
                    linestyle='-',
                    linewidth=3,
                    label='Total Media Effect'
                )

            # Plot base response (all components except promos)
            if components_to_show['base_response']:
                ax_components.plot(
                    dates,
                    base_response,
                    color=color_palette['base_response'],
                    linestyle='-',
                    linewidth=3,
                    label='Base Response'
                )

            # Plot noise component if requested
            if components_to_show['noise_line']:
                ax_components.plot(
                    dates,
                    noise + base_sales,
                    color=color_palette['noise'],
                    linestyle=':',
                    linewidth=1.5,
                    label=f'Noise (σ={noise.std():.2f})'
                )

            # Add noise range
            if components_to_show['noise_range']:
                noise_scale = noise.std()
                ax_components.fill_between(
                    dates,
                    base_sales - 2*noise_scale,
                    base_sales + 2*noise_scale,
                    color=color_palette['noise'],
                    alpha=0.2,
                    label=f'Noise Range (±2σ, scale={noise_scale:.2f})'
                )

            # RIGHT PLOT: Scaled Target with improved clarity
            if components_to_show['target']:
                ax_target.plot(
                    dates,
                    target,
                    color=color_palette['target'],
                    linestyle='-',
                    linewidth=2.5,
                    label=f'Total Response (size={size:.2f})'
                )

            # Add base response for comparison
            if components_to_show['scaled_base']:
                ax_target.plot(
                    dates,
                    10 * base_response,
                    color=color_palette['scaled_base'],
                    linestyle='--',
                    linewidth=2,
                    label='Scaled Base Response',
                    alpha=0.8
                )

            # Add promo effects if requested
            if components_to_show['promo_effects']:
                ax_target.plot(
                    dates,
                    10 * (base_response + promo_effects),
                    color=color_palette['promo_effects'],
                    linestyle='--',
                    linewidth=2,
                    label='With Promo Effects',
                    alpha=0.8
                )

            # Add launch date line to both plots if applicable
            if 'launch_date' in country_params[country]:
                launch_datetime = dates[launch_date]
                ax_components.axvline(
                    x=launch_datetime,
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    label=f'Launch Date: Week {launch_date}'
                )
                ax_target.axvline(
                    x=launch_datetime,
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    label=f'Launch Date: Week {launch_date}'
                )

            # Customize left plot
            ax_components.set_title(f'{country} - Raw Components', fontsize=18, pad=15)
            ax_components.set_xlabel('Date', fontsize=14, labelpad=10)
            ax_components.set_ylabel('Response (Raw)', fontsize=14, labelpad=10)

            # Date formatting for x-axis
            ax_components.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax_components.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(ax_components.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Legend placement and style
            legend = ax_components.legend(
                bbox_to_anchor=(1.01, 1),
                loc='upper left',
                borderaxespad=0,
                frameon=True,
                fancybox=True,
                shadow=True,
                fontsize=12
            )

            # Grid for better readability
            ax_components.grid(True, linestyle='--', alpha=0.6)

            # Customize right plot
            ax_target.set_title(f'{country} - Target (×10)', fontsize=18, pad=15)
            ax_target.set_xlabel('Date', fontsize=14, labelpad=10)

            # Date formatting for x-axis
            ax_target.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax_target.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(ax_target.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Improve grid for better readability
            ax_target.grid(True, linestyle='--', alpha=0.6)

        # Add overall title
        if len(countries) == 1:
            fig.suptitle(f"Marketing Mix Modeling: {countries[0]} Performance Analysis",
                         fontsize=20, y=0.98)
        else:
            fig.suptitle("Marketing Mix Modeling: Multi-Country Performance Analysis",
                         fontsize=20, y=0.98)

        # Ensure proper spacing between plots
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        return fig

    @staticmethod
    def plot_quarterly_roas(quarterly_roas, countries):
        """
        Plot quarterly ROAS for each country

        Parameters:
        -----------
        quarterly_roas : pd.DataFrame
            DataFrame containing quarterly ROAS values
        countries : list
            List of countries to visualize

        Returns:
        --------
        matplotlib.figure.Figure
            The matplotlib figure object
        """
        fig, axes = plt.subplots(len(countries), 1, figsize=(15, 5*len(countries)))
        if len(countries) == 1:
            axes = [axes]

        for idx, country in enumerate(countries):
            ax = axes[idx]
            country_data = quarterly_roas.loc[:, country]

            country_data.plot(ax=ax, marker='o')
            ax.set_title(f'{country} - Quarterly ROAS by Channel')
            ax.set_xlabel('Quarter')
            ax.set_ylabel('ROAS')
            ax.grid(True, alpha=0.3)
            ax.legend(title='Channel', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_global_roas(global_roas_dict):
        """
        Plot global ROAS for all countries and channels

        Parameters:
        -----------
        global_roas_dict : dict
            Nested dictionary with structure {channel: {country: ROAS}}

        Returns:
        --------
        matplotlib.figure.Figure
            The matplotlib figure object
        """
        # Convert nested dict to DataFrame
        data = []
        for channel, countries_dict in global_roas_dict.items():
            for country, roas in countries_dict.items():
                data.append({
                    'Channel': channel,
                    'Country': country,
                    'ROAS': roas
                })

        df = pd.DataFrame(data)

        # Create the plot
        fig = plt.figure(figsize=(12, 6))

        # Pivot the data for plotting
        plot_data = df.pivot(index='Channel', columns='Country', values='ROAS')

        # Create the bar plot
        ax = plot_data.plot(kind='bar', width=0.8)

        plt.title('Global ROAS by Channel and Country')
        plt.xlabel('Channel')
        plt.ylabel('ROAS')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        return fig

    @staticmethod
    def plot_potential_outcomes(dataset, country_params, country, channel, period='Q'):
        """
        Plot actual vs potential outcomes for each period

        Parameters:
        -----------
        dataset : xarray.Dataset
            The dataset containing all variables
        country_params : dict
            Dictionary containing parameters for each country
        country : str
            Country to analyze
        channel : str
            Media channel to analyze
        period : str, optional
            Time period for aggregation ('Q' for quarterly, 'M' for monthly)

        Returns:
        --------
        matplotlib.figure.Figure
            The matplotlib figure object
        """
        # This method would require additional implementation from the MMMDataGenerator
        # to calculate potential outcomes. For now, we'll show a placeholder with the interface.

        # For actual implementation, you would need to move or adapt the generate_potential_y
        # method from MMMDataGenerator to work with this visualizer.

        dates = dataset.coords['date'].values
        dates_pd = pd.to_datetime(dates)
        df_dates = pd.DataFrame({'date': dates_pd})
        df_dates[period] = df_dates['date'].dt.to_period(period)
        period_ranges = df_dates.groupby(period).agg({'date': ['min', 'max']})

        fig, axes = plt.subplots(
            nrows=len(period_ranges),
            ncols=1,
            figsize=(15, 4*len(period_ranges)),
            sharex=True,
            sharey=True
        )

        if len(period_ranges) == 1:
            axes = [axes]

        # Implementation would go here

        return fig

    @staticmethod
    def plot_roas_comparison(calculated_roas, true_roas, countries, channels):
        """
        Plot comparison between calculated and true ROAS

        Parameters:
        -----------
        calculated_roas : dict
            Nested dictionary with structure {channel: {country: ROAS}}
        true_roas : dict
            Nested dictionary with structure {channel: {country: ROAS}}
        countries : list
            List of countries to analyze
        channels : list
            List of media channels to analyze

        Returns:
        --------
        matplotlib.figure.Figure
            The matplotlib figure object
        """
        # Combine data into a DataFrame
        data = []
        for channel in channels:
            for country in countries:
                calc_roas = calculated_roas[channel].get(country, 0)
                true_roas_value = true_roas[channel].get(country, 0)
                data.append({
                    'Channel': channel,
                    'Country': country,
                    'Calculated ROAS': calc_roas,
                    'True ROAS': true_roas_value,
                    'Difference': calc_roas - true_roas_value,
                    'Percent Error': 100 * (calc_roas - true_roas_value) / max(0.01, true_roas_value)
                })

        df = pd.DataFrame(data)

        # Create the bar plot
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Plot 1: ROAS Values
        plot_data = df.pivot_table(
            values=['Calculated ROAS', 'True ROAS'],
            index='Channel',
            columns='Country'
        )

        # Prepare data for grouped bar chart
        num_countries = len(countries)
        num_channels = len(channels)
        indices = np.arange(num_channels)
        bar_width = 0.35

        for i, country in enumerate(countries):
            calculated = [df[(df['Country'] == country) & (df['Channel'] == ch)]['Calculated ROAS'].values[0]
                         for ch in channels]
            true = [df[(df['Country'] == country) & (df['Channel'] == ch)]['True ROAS'].values[0]
                  for ch in channels]

            ax = axes[0]
            ax.bar(indices - bar_width/2 + i*(bar_width/num_countries),
                   calculated,
                   bar_width/num_countries,
                   label=f'{country} Calculated')
            ax.bar(indices + bar_width/2 + i*(bar_width/num_countries),
                   true,
                   bar_width/num_countries,
                   label=f'{country} True',
                   alpha=0.7)

        ax.set_xlabel('Channel')
        ax.set_ylabel('ROAS')
        ax.set_title('Calculated vs True ROAS by Channel and Country')
        ax.set_xticks(indices)
        ax.set_xticklabels(channels)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Percent Error
        errors = df.pivot(index='Channel', columns='Country', values='Percent Error')
        errors.plot(kind='bar', ax=axes[1])
        axes[1].set_title('Percent Error in ROAS Calculation')
        axes[1].set_xlabel('Channel')
        axes[1].set_ylabel('Percent Error (%)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_channel_performance_heatmap(dataset, countries, channels, metric='target'):
        """
        Create a heatmap of channel performance across countries

        Parameters:
        -----------
        dataset : xarray.Dataset
            The dataset containing all variables
        countries : list
            List of countries to analyze
        channels : list
            List of media channels to analyze
        metric : str, optional
            Metric to use for heatmap ('target', 'media_response', etc.)

        Returns:
        --------
        matplotlib.figure.Figure
            The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        data = np.zeros((len(countries), len(channels)))

        for i, country in enumerate(countries):
            for j, channel in enumerate(channels):
                if metric == 'target':
                    # Sum the media response for each channel
                    value = dataset.media_response.sel(country=country, channel=channel).sum().item()
                elif metric == 'spend':
                    # Sum the spend for each channel
                    value = dataset.spends.sel(country=country, channel=channel).sum().item()
                else:
                    # Default to media response
                    value = dataset.media_response.sel(country=country, channel=channel).sum().item()

                data[i, j] = value

        # Normalize data for better visualization
        data_normalized = data / data.max()

        # Create heatmap
        im = ax.imshow(data_normalized, cmap='viridis')

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(f'Normalized {metric.capitalize()}', rotation=-90, va="bottom")

        # Set ticks and labels
        ax.set_xticks(np.arange(len(channels)))
        ax.set_yticks(np.arange(len(countries)))
        ax.set_xticklabels(channels)
        ax.set_yticklabels(countries)

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create annotations
        for i in range(len(countries)):
            for j in range(len(channels)):
                text = ax.text(j, i, f"{data[i, j]:.1f}",
                              ha="center", va="center", color="w" if data_normalized[i, j] < 0.7 else "black")

        ax.set_title(f"Channel Performance Heatmap ({metric.capitalize()})")
        fig.tight_layout()

        return fig

    @staticmethod
    def plot_media_spend(dataset, countries, total_spend=False, split_channels=False):
        """
        Simple plot of media spend for each country and channel.

        Parameters:
        -----------
        dataset : xarray.Dataset
            The dataset containing all variables
        countries : list
            List of countries to visualize
        total_spend : bool, default=False
            If True, include a plot of total spend across all channels
        split_channels : bool, default=True
            If True, create separate subplots for each channel
            If False, plot all channels on a single plot per country

        Returns:
        --------
        matplotlib.figure.Figure or list
            The matplotlib figure object(s)
        """
        channels = dataset.coords['channel'].values
        dates = pd.to_datetime(dataset.coords['date'].values)

        # Set up plot styling
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })

        # Create a colormap for channels
        colors = plt.cm.tab10.colors
        channel_colors = {channel: colors[i % len(colors)] for i, channel in enumerate(channels)}

        # Add a color for total spend
        if total_spend:
            channel_colors['Total'] = '#000000'  # Black for total spend

        # Create figures for each country
        figure_objects = []

        for country in countries:
            if split_channels:
                # Create a grid of subplots, one for each channel
                n_plots = len(channels) + (1 if total_spend else 0)
                n_cols = min(3, n_plots)  # Max 3 columns
                n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

                fig, axes = plt.subplots(
                    n_rows, n_cols,
                    figsize=(7 * n_cols, 5 * n_rows),
                    constrained_layout=True
                )

                # Handle case when there's only one row or one column
                if n_rows == 1 and n_cols == 1:
                    axes = np.array([[axes]])
                elif n_rows == 1:
                    axes = axes.reshape(1, -1)
                elif n_cols == 1:
                    axes = axes.reshape(-1, 1)

                # Flatten axes for easier indexing
                axes_flat = axes.flatten()

                # Add overall title
                fig.suptitle(
                    f'Media Spend for {country}',
                    fontsize=20,
                    fontweight='bold',
                    y=0.98
                )

                # Plot each channel
                for ch_idx, channel in enumerate(channels):
                    ax = axes_flat[ch_idx]

                    # Get spend data
                    spend = dataset.spends.sel(country=country, channel=channel).values

                    # Plot spend
                    ax.plot(
                        dates,
                        spend,
                        color=channel_colors[channel],
                        linewidth=2.5,
                        label=channel
                    )

                    # Add labels and styling
                    ax.set_title(f'{channel} Spend', fontsize=16, pad=10)
                    ax.set_xlabel('Date', fontsize=14, labelpad=10)
                    ax.set_ylabel('Spend', fontsize=14, labelpad=10)
                    ax.grid(True, linestyle='--', alpha=0.6)

                    # Format x-axis dates
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

                # Plot total spend if requested
                if total_spend:
                    ax = axes_flat[len(channels)]

                    # Calculate total spend across all channels
                    total = np.zeros_like(dataset.spends.sel(country=country, channel=channels[0]).values)
                    for channel in channels:
                        total += dataset.spends.sel(country=country, channel=channel).values

                    # Plot total spend
                    ax.plot(
                        dates,
                        total,
                        color=channel_colors['Total'],
                        linewidth=2.5,
                        label='Total Spend'
                    )

                    # Add labels and styling
                    ax.set_title('Total Media Spend', fontsize=16, pad=10)
                    ax.set_xlabel('Date', fontsize=14, labelpad=10)
                    ax.set_ylabel('Spend', fontsize=14, labelpad=10)
                    ax.grid(True, linestyle='--', alpha=0.6)

                    # Format x-axis dates
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

                    # Add legend
                    ax.legend(
                        loc='upper right',
                        frameon=True,
                        fancybox=True,
                        shadow=True
                    )

                # Hide any unused subplots
                for idx in range(len(channels) + (1 if total_spend else 0), len(axes_flat)):
                    axes_flat[idx].set_visible(False)

            else:
                # Create a single plot for each country with all channels
                fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

                # Add title
                fig.suptitle(
                    f'Media Spend for {country}',
                    fontsize=20,
                    fontweight='bold',
                    y=0.98
                )

                # Plot each channel on the same axes
                for channel in channels:
                    spend = dataset.spends.sel(country=country, channel=channel).values

                    ax.plot(
                        dates,
                        spend,
                        color=channel_colors[channel],
                        linewidth=2.5,
                        label=channel
                    )

                # Plot total spend if requested
                if total_spend:
                    total = np.zeros_like(dataset.spends.sel(country=country, channel=channels[0]).values)
                    for channel in channels:
                        total += dataset.spends.sel(country=country, channel=channel).values

                    ax.plot(
                        dates,
                        total,
                        color=channel_colors['Total'],
                        linewidth=3.0,
                        linestyle='--',
                        label='Total Spend'
                    )

                # Add labels and styling
                ax.set_xlabel('Date', fontsize=14, labelpad=10)
                ax.set_ylabel('Spend', fontsize=14, labelpad=10)
                ax.grid(True, linestyle='--', alpha=0.6)

                # Format x-axis dates
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

                # Add legend
                ax.legend(
                    loc='upper right',
                    frameon=True,
                    fancybox=True,
                    shadow=True
                )

            figure_objects.append(fig)

        # Return the last figure if only one country, or all figures as a list
        if len(countries) == 1:
            return figure_objects[0]
        else:
            return figure_objects

    @staticmethod
    def plot_media_transformations(dataset, country_params, countries, dynamics_dict=None):
        """
        Plot media spend transformations (original, adstocked, saturated) for each country and channel
        with improved visual design and readability.

        Parameters:
        -----------
        dataset : xarray.Dataset
            The dataset containing all variables
        country_params : dict
            Dictionary containing parameters for each country
        countries : list
            List of countries to visualize
        dynamics_dict : dict, optional
            Dictionary containing diminishing returns factors for each country.
            If None or False, the diminishing returns plot will be omitted.

        Returns:
        --------
        matplotlib.figure.Figure
            The matplotlib figure object
        """
        channels = dataset.coords['channel'].values
        dates = pd.to_datetime(dataset.coords['date'].values)

        # Set up plot styling for better readability
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })

        # Color palette for consistency across plots
        colors = {
            'original': '#1f77b4',  # Blue
            'adstocked': '#2ca02c',  # Green
            'saturated': '#d62728',  # Red
            'diminished': '#d00728',  # Red variation
            'launch_date': '#7f7f7f'  # Gray
        }

        # Determine number of plot columns based on dynamics_dict
        num_cols = 4 if dynamics_dict else 3

        # Create figures for each country
        figure_objects = []

        for country in countries:
            # Create a figure with subplots for each channel
            fig, axes = plt.subplots(
                len(channels),
                num_cols,
                figsize=(8 * num_cols, 5 * len(channels)),
                constrained_layout=True  # Better spacing management
            )

            # Add overall title with improved formatting
            fig.suptitle(
                f'Media Transformations for {country}',
                fontsize=20,
                fontweight='bold',
                y=0.98
            )

            # Handle the case of a single channel
            if len(channels) == 1:
                axes = axes.reshape(1, num_cols)

            for ch_idx, channel in enumerate(channels):
                # Get channel-specific parameters
                alpha = country_params[country]['alphas'][channel]
                lam = country_params[country]['lambdas'][channel]

                # Get original spend data
                spend = dataset.spends.sel(country=country, channel=channel).values

                # Calculate transformations
                adstocked = geometric_adstock(spend, alpha, l_max=8, normalize=True).eval()
                saturated = logistic_saturation(adstocked, lam).eval()

                # Plot original spend with improved styling
                axes[ch_idx, 0].plot(
                    dates,
                    spend,
                    color=colors['original'],
                    linewidth=2.5,
                    label='Original Spend'
                )
                axes[ch_idx, 0].set_title(
                    f'{channel} - Original Spend',
                    fontsize=16,
                    pad=10
                )
                axes[ch_idx, 0].grid(True, linestyle='--', alpha=0.6)

                # Plot adstocked spend with improved styling
                axes[ch_idx, 1].plot(
                    dates,
                    adstocked,
                    color=colors['adstocked'],
                    linewidth=2.5,
                    label=f'Adstocked (α={alpha:.2f})'
                )
                axes[ch_idx, 1].set_title(
                    f'{channel} - After Adstock',
                    fontsize=16,
                    pad=10
                )
                axes[ch_idx, 1].grid(True, linestyle='--', alpha=0.6)

                # Plot saturated spend with improved styling
                axes[ch_idx, 2].plot(
                    dates,
                    saturated,
                    color=colors['saturated'],
                    linewidth=2.5,
                    label=f'Saturated (λ={lam:.2f})'
                )
                axes[ch_idx, 2].set_title(
                    f'{channel} - After Saturation',
                    fontsize=16,
                    pad=10
                )
                axes[ch_idx, 2].grid(True, linestyle='--', alpha=0.6)

                # Plot diminishing returns if dynamics_dict is provided
                if dynamics_dict:
                    diminishing_returns = dynamics_dict[country]
                    diminished = saturated * diminishing_returns

                    axes[ch_idx, 3].plot(
                        dates,
                        diminished,
                        color=colors['diminished'],
                        linewidth=2.5,
                        label=f'After Diminishing Returns'
                    )
                    axes[ch_idx, 3].set_title(
                        f'{channel} - After Diminishing returns',
                        fontsize=16,
                        pad=10
                    )
                    axes[ch_idx, 3].grid(True, linestyle='--', alpha=0.6)

                # Add launch date vertical line to all plots with improved styling
                if 'launch_date' in country_params[country]:
                    launch_date = dates[country_params[country]['launch_date']]
                    for ax_idx, ax in enumerate(axes[ch_idx]):
                        ax.axvline(
                            x=launch_date,
                            color=colors['launch_date'],
                            linestyle='--',
                            linewidth=2,
                            label='Launch Date' if ax_idx == 0 else None  # Only add to legend once
                        )

                # Improve x-axis formatting for all plots
                for ax in axes[ch_idx]:
                    ax.set_xlabel('Date', fontsize=14, labelpad=10)
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

                    # Add legends with improved styling
                    legend = ax.legend(
                        loc='upper right',
                        frameon=True,
                        fancybox=True,
                        shadow=True,
                        fontsize=12
                    )

                # Set y-label only for leftmost plot with improved styling
                axes[ch_idx, 0].set_ylabel(f'{channel} Response', fontsize=14, labelpad=10)

                # Add a textbox with key parameters in the first plot
                param_text = (
                    f"Channel: {channel}\n"
                    f"Alpha (adstock): {alpha:.2f}\n"
                    f"Lambda (saturation): {lam:.2f}"
                )

                # Add summary statistics to help interpretation
                spend_active = np.sum(spend > 0) / len(spend)
                spend_max = np.max(spend)

                stat_text = (
                    f"Activity rate: {spend_active:.1%}\n"
                    f"Max spend: {spend_max:.2f}\n"
                    f"Avg spend: {np.mean(spend):.2f}"
                )

                # Add parameter textbox to first plot
                axes[ch_idx, 0].text(
                    0.02, 0.02,
                    param_text,
                    transform=axes[ch_idx, 0].transAxes,
                    fontsize=12,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
                )

                # Add stats textbox to third plot
                axes[ch_idx, 2].text(
                    0.02, 0.02,
                    stat_text,
                    transform=axes[ch_idx, 2].transAxes,
                    fontsize=12,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
                )

            figure_objects.append(fig)

        # Return the last figure if only one country, or all figures as a list
        if len(countries) == 1:
            return figure_objects[0]
        else:
            return figure_objects


    def plot_saturation_curves(self, dataset, country_params, countries, show_raw_data=True):
        """
        Plot saturation curves showing the relationship between adstocked values and
        saturated values for each country and channel.

        Parameters:
        -----------
        dataset : xarray.Dataset
            The dataset containing all variables
        country_params : dict
            Dictionary containing parameters for each country
        countries : list
            List of countries to visualize
        show_raw_data : bool, optional
            Whether to show the actual data points in addition to the curves.
            Defaults to True.

        Returns:
        --------
        matplotlib.figure.Figure or list
            The matplotlib figure object(s)
        """

        channels = dataset.coords['channel'].values

        # Set up plot styling for better readability
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18
        })

        # Color palette
        colors = {
            'curve': '#1f77b4',  # Blue
            'data_points': '#ff7f0e',  # Orange
            'grid': '#cccccc'  # Light gray
        }

        figure_objects = []

        for country in countries:
            # Create a figure with subplots for each channel
            fig, axes = plt.subplots(
                1,
                len(channels),
                figsize=(7 * len(channels), 6),
                constrained_layout=True
            )

            # Handle the case of a single channel
            if len(channels) == 1:
                axes = np.array([axes])

            # fig.suptitle(
            #     f'Saturation Curves for {country}',
            #     fontsize=20,
            #     fontweight='bold',
            #     y=0.98
            # )

            for ch_idx, channel in enumerate(channels):
                # Get channel-specific parameters
                alpha = country_params[country]['alphas'][channel]
                lam = country_params[country]['lambdas'][channel]

                # Get original spend data
                spend = dataset.spends.sel(country=country, channel=channel).values

                # Calculate adstock transformation
                adstocked = geometric_adstock(spend, alpha, l_max=8, normalize=True).eval()

                if show_raw_data:
                    # Plot actual data points
                    axes[ch_idx].scatter(
                        adstocked,
                        logistic_saturation(adstocked, lam).eval(),
                        color=colors['data_points'],
                        alpha=0.5,
                        s=30,
                        label='Observed Data'
                    )

                # Generate points for a smooth curve
                x_range = np.linspace(0, max(adstocked) * 1.1, 1000)
                y_values = logistic_saturation(x_range, lam).eval()

                # Plot the saturation curve
                axes[ch_idx].plot(
                    x_range,
                    y_values,
                    color=colors['curve'],
                    linewidth=3,
                    label=f'Saturation Curve (λ={lam:.2f})'
                )

                # Add annotations and styling
                axes[ch_idx].set_title(f'{channel} Saturation Curve', fontsize=16, pad=10)
                axes[ch_idx].set_xlabel('Adstocked Media Spend', fontsize=14, labelpad=10)
                axes[ch_idx].set_ylabel('Saturated Response', fontsize=14, labelpad=10)
                axes[ch_idx].grid(True, linestyle='--', alpha=0.6, color=colors['grid'])

                # Add legend with improved styling
                legend = axes[ch_idx].legend(
                    loc='upper left',
                    frameon=True,
                    fancybox=True,
                    shadow=True,
                    fontsize=12
                )

                # Set axis limits
                axes[ch_idx].set_xlim(0, max(adstocked) * 1.1)
                axes[ch_idx].set_ylim(0, 1.05)

            figure_objects.append(fig)

        # Return the last figure if only one country, or all figures as a list
        if len(countries) == 1:
            return figure_objects[0]
        else:
            return figure_objects

    def calculate_simple_roas(self, dataset, countries, channels, time_constraint=None):
        """
        Calculate incremental ROAS using the direct method:
        ROAS = total media response / total media spend

        Parameters:
        -----------
        dataset : xarray.Dataset
            Dataset containing media_response and spends variables
        countries : list
            List of countries to analyze
        channels : list
            List of channels to analyze
        time_constraint : tuple, dict, slice, list, or xarray.Dataset, optional
            Specifies which time periods to include in the calculation.
            Can be:
            - tuple of (start_idx, end_idx) to apply to all countries
            - dict mapping country to (start_idx, end_idx) tuples
            - slice object to select specific indices
            - list of specific indices to include
            - xarray.Dataset or DataArray with 'date' coordinate to select matching dates
            If None, all time periods are used.

        Returns:
        --------
        dict
            Dictionary of ROAS by country and channel
        """

        # Initialize results dictionary
        roas_results = {}

        # Get date dimension from dataset
        all_dates = dataset.coords['date'].values

        # Process time_constraint to get indices for each country
        if time_constraint is None:
            # Use all available dates for all countries
            selected_dates = all_dates
        elif isinstance(time_constraint, tuple) and len(time_constraint) == 2:
            # Same start/end indices for all countries
            start_idx, end_idx = time_constraint
            selected_dates = all_dates[start_idx:end_idx+1]
        elif hasattr(time_constraint, 'coords') and 'date' in time_constraint.coords:
            # Handle xarray.Dataset or xarray.DataArray with date coordinate
            selected_dates = time_constraint.coords['date'].values
        elif isinstance(time_constraint, dict):
            # Different indices for each country - handle at country level
            selected_dates = None  # Will be set per country
        elif isinstance(time_constraint, slice):
            # Same slice for all countries
            selected_dates = all_dates[time_constraint]
        elif hasattr(time_constraint, '__iter__'):
            # Specific list of indices for all countries
            if all(isinstance(i, (int, np.integer)) for i in time_constraint):
                # List of integer indices
                selected_dates = all_dates[list(time_constraint)]
            else:
                # Assume it's a list of date values
                selected_dates = np.array(time_constraint)
        else:
            raise ValueError("Invalid time_constraint format")

        for country in countries:
            roas_results[country] = {}

            # Handle country-specific date selection if time_constraint is a dict
            if isinstance(time_constraint, dict) and selected_dates is None:
                if country in time_constraint:
                    constraint = time_constraint[country]
                    if isinstance(constraint, tuple) and len(constraint) == 2:
                        start_idx, end_idx = constraint
                        country_dates = all_dates[start_idx:end_idx+1]
                    elif hasattr(constraint, 'coords') and 'date' in constraint.coords:
                        # Handle xarray with date coordinate
                        country_dates = constraint.coords['date'].values
                    elif isinstance(constraint, slice):
                        country_dates = all_dates[constraint]
                    elif hasattr(constraint, '__iter__'):
                        if all(isinstance(i, (int, np.integer)) for i in constraint):
                            country_dates = all_dates[list(constraint)]
                        else:
                            country_dates = np.array(constraint)
                    else:
                        country_dates = all_dates
                else:
                    country_dates = all_dates
            else:
                country_dates = selected_dates

            # Get total spend and response for each channel
            for channel in channels:
                # Select data for specific dates
                channel_data = dataset.sel(country=country, channel=channel, date=country_dates)

                # Get response and spend
                response = channel_data.media_response.values * 10 ## Amplitude
                spend = channel_data.spends.values

                # Calculate total response and spend
                total_response = np.sum(response)
                total_spend = np.sum(spend)

                # Calculate ROAS (response/spend)
                if total_spend > 0:
                    roas = total_response / total_spend
                else:
                    roas = np.nan

                # Store results
                roas_results[country][channel] = {
                    'total_response': float(total_response),
                    'total_spend': float(total_spend),
                    'roas': float(roas)
                }

            # Calculate overall ROAS for this country
            total_country_response = sum(
                roas_results[country][channel]['total_response']
                for channel in channels
            )
            total_country_spend = sum(
                roas_results[country][channel]['total_spend']
                for channel in channels
            )

            if total_country_spend > 0:
                overall_roas = total_country_response / total_country_spend
            else:
                overall_roas = np.nan

            roas_results[country]['overall'] = {
                'total_response': float(total_country_response),
                'total_spend': float(total_country_spend),
                'roas': float(overall_roas)
            }

        return roas_results

    def calculate_incremental_roas(self, dataset, model, X, media_columns, test_indices):
        """
        Calculate incremental ROAS using the counterfactual approach:
        ROAS = (baseline prediction - zeroed channel prediction).sum() / channel spend.sum()

        Parameters:
        -----------
        dataset : xarray.Dataset
            Dataset containing original data
        model : object
            Fitted model object
        X : DataFrame
            Feature data
        media_columns : list
            List of media columns
        test_indices : list or array-like
            Indices to use for testing

        Returns:
        --------
        dict
            Dictionary of incremental ROAS by channel
        """

        # Get test data
        X_test = X.iloc[test_indices] if hasattr(test_indices, '__iter__') else X.loc[test_indices]

        # Get baseline prediction
        baseline_prediction = model.predict(test_indices=test_indices)

        # Calculate incremental ROAS for each channel
        roas_results = {}

        for channel in media_columns:
            # Create copy of test data with channel zeroed out
            X_zeroed = X_test.copy()
            X_zeroed[channel] = 0

            # Store original spend
            original_spend = X_test[channel].values
            total_spend = np.sum(original_spend)

            try:
                # Try to use the model's existing predict method with test_indices
                # This assumes model.predict() is implemented to handle zeroed channels

                # Create temporary X with zeroed channel for entire dataset
                X_temp = X.copy()
                X_temp.loc[X_test.index, channel] = 0

                # Using model's predict method with test_indices
                zeroed_prediction = model.predict(test_indices=test_indices, X_new=X_temp)

            except Exception as e:
                print(f"Warning: Using fallback prediction method for {channel}. Error: {e}")

                # Fallback approach: Try to manually create transformed features and predict
                # This code will need customization based on your specific model implementation
                if hasattr(model, 'transform_features'):
                    # If model has a transform_features method, use it
                    X_transformed = model.transform_features(X_zeroed)
                    zeroed_prediction = model.model.predict(X_transformed)
                elif hasattr(model, 'model') and hasattr(model.model, 'predict'):
                    # Direct prediction with model's underlying model
                    zeroed_prediction = model.model.predict(X_zeroed)
                else:
                    # No suitable prediction method found
                    print(f"Error: Could not generate zeroed prediction for {channel}")
                    zeroed_prediction = baseline_prediction  # Use baseline as fallback

            # Calculate incremental contribution
            incremental_contribution = np.sum(baseline_prediction - zeroed_prediction)

            # Calculate incremental ROAS
            if total_spend > 0:
                incremental_roas = incremental_contribution / total_spend
            else:
                incremental_roas = np.nan

            # Store results
            roas_results[channel] = {
                'incremental_contribution': float(incremental_contribution),
                'total_spend': float(total_spend),
                'incremental_roas': float(incremental_roas),
                'baseline_prediction_sum': float(np.sum(baseline_prediction)),
                'zeroed_prediction_sum': float(np.sum(zeroed_prediction))
            }

        return roas_results

    def visualize_roas(self, roas_results, countries=None, channels=None, plot_type='bar'):
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
            plt.title('ROAS by Channel and Country', fontsize=16)
            plt.legend(title='Country')

            # Add value labels on bars
            for i, bar in enumerate(ax.patches):
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.05,
                        f'{height:.2f}',
                        ha='center', va='bottom'
                    )

            plt.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()

        return plt.gcf()
