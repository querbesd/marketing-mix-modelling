import numpy as np
import pandas as pd
import pytensor.tensor as pt
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation
from datetime import datetime, timedelta
import xarray as xr
import matplotlib.pyplot as plt


class MMMDataGenerator:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def generate_global_parameters(self):
        """Generate global parameters with more realistic variations"""
        return {
            # Channel-specific base parameters with more overlap
            'alpha_channel_base': {
                'TV': 0.8,
                'Radio': 0.25,
                'OOH': 0.35,
                'Digital': 0.28
            },
            'alpha_country_sigma': 0.15,

            'lambda_channel_base': {
                'TV': 1.5,
                'Radio': 2,
                'OOH': 1,
                'Digital': 7
            },
            'lambda_country_sigma': 0.4,

            'effect_channel_base': {
                'TV': 5.0,
                'Radio': 6.0,
                'OOH': 4.7,
                'Digital': 8
            },
            'effect_country_sigma': 0.35,

            'promo_mu': 1.3,
            'promo_sigma': 0.2,
            'base_mu': 2,
            'base_sigma': 0.5
        }

    def generate_launch_dates(self, n_dates, countries):
        """Generate launch dates for each country"""
        launch_dates = {}
        # First country always starts from beginning
        launch_dates[countries[0]] = 0

        # Other countries start at random weeks
        for country in countries[1:]:
            # Random start between week 0 and week n_dates/2
            launch_dates[country] = self.rng.integers(0, n_dates//2)

        return launch_dates

    def generate_country_parameters(self, global_params, countries, channels, launch_dates):
        """Generate country parameters with launch dates"""
        country_params = {}

        # Generate country sizes
        country_sizes = self.rng.lognormal(0, 1, size=len(countries))
        country_sizes = country_sizes / np.max(country_sizes)

        # Generate shared effects
        traditional_effect = abs(self.rng.normal(0, 0.3))
        digital_effect = abs(self.rng.normal(0, 0.3))

        for i, country in enumerate(countries):
            country_effect = 0

            country_params[country] = {
                'size': country_sizes[i],
                'launch_date': launch_dates[country],

                'alphas': {
                    ch: np.clip(
                        global_params['alpha_channel_base'][ch] *
                        (1 + abs(self.rng.normal(country_effect, global_params['alpha_country_sigma']))),
                        0.1, 0.9
                    ) for ch in channels
                },

                'lambdas': {
                    ch: max(0.1,
                        global_params['lambda_channel_base'][ch] *
                        (1 + abs(self.rng.normal(country_effect, global_params['lambda_country_sigma'])))
                    ) for ch in channels
                },

                'effects': {
                    ch: global_params['effect_channel_base'][ch]*
                    (1 + abs(self.rng.normal(country_effect, global_params['effect_country_sigma'])))
                    # * (1 + traditional_effect)
#                    if ch != "Digital" else global_params['effect_channel_base'][ch] * (1 + digital_effect)
                    for ch in channels
                },

                'promo_multiplier': np.exp(
                    self.rng.normal(
                        np.log(global_params['promo_mu']),
                        global_params['promo_sigma'])
                ),

                'base_sales': np.exp(
                    self.rng.normal(
                        np.log(global_params['base_mu']),
                        global_params['base_sigma'])
                )
            }

        return country_params

    def generate_media_data(self, n_dates, channels, country_size):
        """Generate media data with varying sparsity based on country size"""
        spends = {}

        # Adjust sparsity based on country size
        base_prob = 0.2 * country_size  # Smaller countries have sparser data

        for channel in channels:
            if channel == 'Digital':
                # Generate more consistent digital spending
                base_spend = self.rng.uniform(0, 1, size=n_dates)
                spend = np.where(base_spend > (1 - base_prob),
                               base_spend * 2,
                               base_spend )
            else:
                # Generate sparser traditional media spending
                base_spend = self.rng.binomial(1, base_prob, size=n_dates) # Generate
                noise = np.where(base_spend>0, self.rng.normal(0, 0.2, size=n_dates),0)

                spend = base_spend + noise

            spend = np.clip(spend, 0, 2)
            spends[channel] = spend

        return spends


    def generate_launch_dates(self, n_dates, countries, staggered_starts=False):
        """
        Generate launch dates for each country

        Parameters:
        -----------
        n_dates : int
            Number of dates in the dataset
        countries : list
            List of countries
        staggered_starts : bool
            If True, countries start at different times with first country at 0
            If False, all countries start at time 0
        """
        launch_dates = {}

        if staggered_starts:
            # First country always starts from beginning
            launch_dates[countries[0]] = 0

            # Other countries start at random weeks
            for country in countries[1:]:
                # Random start between week 0 and week n_dates/2
                launch_dates[country] = self.rng.integers(0, n_dates//2)
        else:
            # All countries start at time 0
            for country in countries:
                launch_dates[country] = 0

        return launch_dates

    def generate_dataset(self, n_dates, countries, channels, staggered_starts=False, noise_std=0.4):
        """
        Generate dataset with optional staggered launches

        Parameters:
        -----------
        n_dates : int
            Number of dates
        countries : list
            List of countries
        channels : list
            List of channels
        staggered_starts : bool
            If True, uses staggered launch dates
            If False, all countries start at time 0
        noise_std : float
            Standard deviation for noise
        """
        global_params = self.generate_global_parameters()
        launch_dates = self.generate_launch_dates(n_dates, countries, staggered_starts)
        country_params = self.generate_country_parameters(global_params, countries, channels, launch_dates)

        # Initialize arrays for all components
        all_targets = np.zeros((len(countries), n_dates))
        all_spends = np.zeros((len(countries), n_dates, len(channels)))
        all_promos = np.zeros((len(countries), n_dates))
        all_base_sales = np.zeros((len(countries), n_dates))
        all_trends = np.zeros((len(countries), n_dates))
        all_seasonality = np.zeros((len(countries), n_dates))
        all_media_response = np.zeros((len(countries), n_dates, len(channels)))
        all_noise = np.zeros((len(countries), n_dates))
        all_promo_effects = np.zeros((len(countries), n_dates))

        t = np.arange(n_dates)
        trend_slope = 0.15
        seasonal_amplitude = 10
        amplitude = 10

        overall_trend = trend_slope * t
        overall_seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / 52)

        for i, country in enumerate(countries):
            launch_date = country_params[country]['launch_date']
            country_size = country_params[country]['size']

            # Generate data only for post-launch period
            active_dates = n_dates - launch_date
            spends = self.generate_media_data(active_dates, channels, country_size)
            promos = self.rng.binomial(1, 0.2 * country_size, size=active_dates)

            # Place data after launch date
            for j, channel in enumerate(channels):
                all_spends[i, launch_date:, j] = spends[channel]
            all_promos[i, launch_date:] = promos

            # Store base sales
            all_base_sales[i, launch_date:] = country_params[country]['base_sales']

            # Store trend and seasonality
            all_trends[i, launch_date:] = overall_trend[launch_date:]/10
            all_seasonality[i, launch_date:] = overall_seasonal[launch_date:]/10

            # Calculate and store media response
            for j, channel in enumerate(channels):
                if launch_date < n_dates:
                    spend = all_spends[i, :, j]
                    adstocked = geometric_adstock(
                        spend,
                        country_params[country]['alphas'][channel],
                        l_max=8,
                        normalize=True
                    ).eval()

                    saturated = logistic_saturation(
                        adstocked,
                        country_params[country]['lambdas'][channel]
                    ).eval()


                    all_media_response[i, :, j] = saturated * country_params[country]['effects'][channel]

            # Store noise component
            # noise_scale = noise_std * (1 + (1 - country_size))
            noise_scale = 0.75
            epsilon = np.zeros(n_dates)
            epsilon[launch_date:] = self.rng.normal(0, noise_scale, size=active_dates)
            all_noise[i, launch_date:] = epsilon[launch_date:]

            # Calculate base response before promotional effect
            base_response = (
                all_base_sales[i, :] +
                all_trends[i, :] +
                all_seasonality[i, :] +
                np.sum(all_media_response[i, :, :], axis=1) +  # Sum across channels
                all_noise[i, :]
            )

            # Calculate and store promotional effect
            promo_effect = 0 #base_response * (all_promos[i, :] * (country_params[country]['promo_multiplier'] - 1))
            all_promo_effects[i, :] = promo_effect

            # Calculate final response
            all_targets[i, :] = amplitude * (base_response + promo_effect)

        dates = [datetime(2023, 1, 1) + timedelta(weeks=i) for i in range(n_dates)]

        # Create an array of launch dates (one per country)
        launch_date_arr = np.array([country_params[c]['launch_date'] for c in countries])


        # Create dataset with all components
        ds = xr.Dataset({
            'target': xr.DataArray(all_targets, coords=[countries, dates], dims=['country', 'date']),
            'spends': xr.DataArray(all_spends, coords=[countries, dates, channels], dims=['country', 'date', 'channel']),
            'promotions': xr.DataArray(all_promos, coords=[countries, dates], dims=['country', 'date']),
            'base_sales': xr.DataArray(all_base_sales, coords=[countries, dates], dims=['country', 'date']),
            'trend': xr.DataArray(all_trends, coords=[countries, dates], dims=['country', 'date']),
            'seasonality': xr.DataArray(all_seasonality, coords=[countries, dates], dims=['country', 'date']),
            'media_response': xr.DataArray(all_media_response, coords=[countries, dates, channels],
                                     dims=['country', 'date', 'channel']),
            'noise': xr.DataArray(all_noise, coords=[countries, dates], dims=['country', 'date']),
            'promo_effects': xr.DataArray(all_promo_effects, coords=[countries, dates], dims=['country', 'date']),
            'launch_date': xr.DataArray(launch_date_arr, coords=[countries], dims=['country'])
        })

        return ds, global_params, country_params

    def generate_dataset_with_dynamics(self, n_dates, countries, channels, staggered_starts=False,
                                apply_dynamic_effects=True, noise_std=0.4):
        """
        Generate dataset with optional staggered launches and exponential decay effectiveness

        Parameters:
        -----------
        n_dates : int
            Number of dates
        countries : list
            List of countries
        channels : list
            List of channels
        staggered_starts : bool, default=False
            If True, uses staggered launch dates
            If False, all countries start at time 0
        apply_dynamic_effects: bool, default=True
            If True, applies exponential decay to media effects over time
        noise_std : float, default=0.4
            Standard deviation for noise
        """
        global_params = self.generate_global_parameters()
        launch_dates = self.generate_launch_dates(n_dates, countries, staggered_starts)
        country_params = self.generate_country_parameters(global_params, countries, channels, launch_dates)

        # If applying dynamic effects, generate exponential decay effectiveness patterns
        dynamic_effectiveness = {}
        if apply_dynamic_effects:
            for country in countries:
                dynamic_effectiveness[country] = {}


                # Parameters for exponential decay
                initial_value = 1.0  # Start at full effectiveness


                decay_rate = self.rng.uniform(0.012, 0.018)  # Digital decays slightly faster

                # Asymptote (floor) varies slightly by channel
                asymptote = self.rng.uniform(0.25, 0.35)

                # Create time indices
                t = np.arange(n_dates)

                # Generate smooth exponential decay curve
                # Formula: y = asymptote + (initial_value - asymptote) * exp(-decay_rate * t)
                effectiveness = asymptote + (initial_value - asymptote) * np.exp(-decay_rate * t)

                # Store this effectiveness pattern
                dynamic_effectiveness[country] = effectiveness

        # Initialize arrays for all components
        all_targets = np.zeros((len(countries), n_dates))
        all_spends = np.zeros((len(countries), n_dates, len(channels)))
        all_promos = np.zeros((len(countries), n_dates))
        all_base_sales = np.zeros((len(countries), n_dates))
        all_trends = np.zeros((len(countries), n_dates))
        all_seasonality = np.zeros((len(countries), n_dates))
        all_media_response = np.zeros((len(countries), n_dates, len(channels)))
        all_noise = np.zeros((len(countries), n_dates))
        all_promo_effects = np.zeros((len(countries), n_dates))
        # Store dynamic effects if used
        if apply_dynamic_effects:
            all_dynamic_effects = np.zeros((len(countries), n_dates, len(channels)))

        t = np.arange(n_dates)
        trend_slope = 0.15
        seasonal_amplitude = 10
        amplitude = 10

        overall_trend = trend_slope * t
        overall_seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / 52)

        for i, country in enumerate(countries):
            launch_date = country_params[country]['launch_date']
            country_size = country_params[country]['size']

            # Generate data only for post-launch period
            active_dates = n_dates - launch_date
            spends = self.generate_media_data(active_dates, channels, country_size)
            promos = self.rng.binomial(1, 0.2 * country_size, size=active_dates)

            # Place data after launch date
            for j, channel in enumerate(channels):
                all_spends[i, launch_date:, j] = spends[channel]
            all_promos[i, launch_date:] = promos

            # Store base sales
            all_base_sales[i, launch_date:] = country_params[country]['base_sales']

            # Store trend and seasonality
            all_trends[i, launch_date:] = overall_trend[launch_date:]/10
            all_seasonality[i, launch_date:] = overall_seasonal[launch_date:]/10

            # Calculate and store media response
            media_response = np.zeros(n_dates)
            for j, channel in enumerate(channels):
                if launch_date < n_dates:
                    spend = all_spends[i, :, j]
                    adstocked = geometric_adstock(
                        spend,
                        country_params[country]['alphas'][channel],
                        l_max=8,
                        normalize=True
                    ).eval()

                    saturated = logistic_saturation(
                        adstocked,
                        country_params[country]['lambdas'][channel]
                    ).eval()

                    # Apply dynamic effectiveness if enabled
                    if apply_dynamic_effects:
                        # Apply the effectiveness pattern for this country and channel
                        effectiveness = dynamic_effectiveness[country]
                        # Store for analysis
                        all_dynamic_effects[i, :, j] = effectiveness
                        # Apply to the media response
                        media_effect = saturated * country_params[country]['effects'][channel] * effectiveness
                    else:
                        media_effect = saturated * country_params[country]['effects'][channel]

                    all_media_response[i, :, j] = media_effect

            # Store noise component
            noise_scale = 0.75
            epsilon = np.zeros(n_dates)
            epsilon[launch_date:] = self.rng.normal(0, noise_scale, size=active_dates)
            all_noise[i, launch_date:] = epsilon[launch_date:]

            # Calculate base response before promotional effect
            base_response = (
                all_base_sales[i, :] +
                all_trends[i, :] +
                all_seasonality[i, :] +
                np.sum(all_media_response[i, :, :], axis=1) +  # Sum across channels
                all_noise[i, :]
            )

            # Calculate and store promotional effect
            promo_effect = 0  # Disabled in this version
            all_promo_effects[i, :] = promo_effect

            # Calculate final response
            all_targets[i, :] = amplitude * (base_response + promo_effect)

        dates = [datetime(2023, 1, 1) + timedelta(weeks=i) for i in range(n_dates)]

        # Create an array of launch dates (one per country)
        launch_date_arr = np.array([country_params[c]['launch_date'] for c in countries])

        # Create dataset with all components
        ds_components = {
            'target': xr.DataArray(all_targets, coords=[countries, dates], dims=['country', 'date']),
            'spends': xr.DataArray(all_spends, coords=[countries, dates, channels], dims=['country', 'date', 'channel']),
            'promotions': xr.DataArray(all_promos, coords=[countries, dates], dims=['country', 'date']),
            'base_sales': xr.DataArray(all_base_sales, coords=[countries, dates], dims=['country', 'date']),
            'trend': xr.DataArray(all_trends, coords=[countries, dates], dims=['country', 'date']),
            'seasonality': xr.DataArray(all_seasonality, coords=[countries, dates], dims=['country', 'date']),
            'media_response': xr.DataArray(all_media_response, coords=[countries, dates, channels],
                                    dims=['country', 'date', 'channel']),
            'noise': xr.DataArray(all_noise, coords=[countries, dates], dims=['country', 'date']),
            'promo_effects': xr.DataArray(all_promo_effects, coords=[countries, dates], dims=['country', 'date']),
            'launch_date': xr.DataArray(launch_date_arr, coords=[countries], dims=['country'])
        }

        # Add dynamic effects if used
        if apply_dynamic_effects:
            ds_components['dynamic_effects'] = xr.DataArray(
                all_dynamic_effects,
                coords=[countries, dates, channels],
                dims=['country', 'date', 'channel']
            )

        ds = xr.Dataset(ds_components)

        return ds, global_params, country_params, dynamic_effectiveness if apply_dynamic_effects else None


    def generate_potential_y(self, dataset, country_params, country, channel, t0, t1):
        """
        Generate potential outcome by zeroing out specified channel during time period

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
        t0, t1 : datetime
            Start and end dates for the analysis period

        Returns:
        --------
        numpy.ndarray
            Array of potential outcomes
        """
        # Create a copy of the dataset to avoid modifying original
        dataset = dataset.copy()

        # Create mask for the intervention period
        dates = dataset.coords['date'].values
        mask = ~np.logical_and(dates >= t0, dates <= t1)

        # Zero out the specified channel during intervention period
        spends = dataset.spends.sel(country=country).copy()
        spends.loc[dict(channel=channel)] = np.where(
            mask,
            spends.sel(channel=channel),
            0
        )

        # Apply adstock transformation to all channels
        adstocked = {
            ch: geometric_adstock(
                spends.sel(channel=ch).values,
                country_params[country]['alphas'][ch],
                l_max=8,
                normalize=True
            ).eval()
            for ch in dataset.coords['channel'].values
        }

        # Apply saturation transformation
        saturated = {
            ch: logistic_saturation(
                adstocked[ch],
                country_params[country]['lambdas'][ch]
            ).eval()
            for ch in dataset.coords['channel'].values
        }

        # Calculate media effects
        media_effects = sum(
            saturated[ch] * country_params[country]['effects'][ch]
            for ch in dataset.coords['channel'].values
        )

        # Get base components
        base_sales = dataset.base_sales.sel(country=country)
        trend = dataset.trend.sel(country=country)
        seasonality = dataset.seasonality.sel(country=country)
        noise = dataset.noise.sel(country=country)

        # Calculate base response
        base_response = (
            base_sales +
            trend +
            seasonality +
            media_effects +
            noise
        )

        # Add promotional effects
        promos = dataset.promotions.sel(country=country)
        promo_multiplier = country_params[country]['promo_multiplier']
        promo_effects = base_response * (promos * (promo_multiplier - 1))

        # Final potential outcome with scaling factor
        potential_y = 10 * (base_response + promo_effects)

        return potential_y


    def calculate_roas_by_period_dict(self, dataset, country_params, countries, channels, period='Q'):
        """
        Calculate ROAS by period for each channel in each country, returning a nested dictionary

        Parameters:
        -----------
        dataset : xarray.Dataset
            The dataset containing all variables
        country_params : dict
            Dictionary containing parameters for each country
        countries : list
            List of countries to analyze
        channels : list
            List of media channels to analyze
        period : str, optional
            Time period for aggregation ('Q' for quarterly, 'M' for monthly)

        Returns:
        --------
        dict
            Nested dictionary containing ROAS values by country, channel, and period
        """
        dates = pd.to_datetime(dataset.coords['date'].values)
        df_dates = pd.DataFrame({'date': dates})
        df_dates[period] = df_dates['date'].dt.to_period(period)

        period_ranges = df_dates.groupby(period).agg({
            'date': ['min', 'max']
        })

        roas_results = {}

        for country in countries:
            roas_results[country] = {}

            for channel in channels:
                channel_roas = {}

                for idx, row in period_ranges.iterrows():
                    t0, t1 = row['date'][['min', 'max']]

                    # Get actual outcome and spend for this period
                    period_mask = np.logical_and(dates >= t0, dates <= t1)
                    actual_y = dataset.target.sel(country=country).values[period_mask].sum()
                    period_spend = dataset.spends.sel(
                        country=country,
                        channel=channel
                    ).where(period_mask, 0).sum().item()

                    # Calculate ROAS if there was spend
                    if period_spend > 0:
                        potential_y = self.generate_potential_y(
                            dataset=dataset,
                            country_params=country_params,
                            country=country,
                            channel=channel,
                            t0=t0,
                            t1=t1
                        )
                        potential_y_sum = potential_y[period_mask].sum()
                        roas = (actual_y - potential_y_sum) / period_spend
                    else:
                        roas = 0

                    channel_roas[idx] = roas

                roas_results[country][channel] = channel_roas

        return roas_results

    def compute_quarterly_roas(self, dataset, country_params, countries, channels):
        """
        Compute ROAS by quarter for each channel in each country, returning a DataFrame

        Parameters:
        -----------
        dataset : xarray.Dataset
            The dataset containing all variables
        country_params : dict
            Dictionary containing parameters for each country
        countries : list
            List of countries to analyze
        channels : list
            List of media channels to analyze

        Returns:
        --------
        pd.DataFrame
            DataFrame containing quarterly ROAS values for each country-channel combination
        """
        # Get ROAS results as dictionary
        roas_results = self.calculate_roas_by_period_dict(
            dataset=dataset,
            country_params=country_params,
            countries=countries,
            channels=channels,
            period='Q'
        )

        # Convert to DataFrame
        roas_df = pd.DataFrame({
            (country, channel): pd.Series(roas_dict)
            for country, channels_dict in roas_results.items()
            for channel, roas_dict in channels_dict.items()
        }).round(2)

        return roas_df

    def compute_global_roas(self, dataset, country_params, countries, channels):
        """
        Compute global ROAS for each channel in each country

        Parameters:
        -----------
        dataset : xarray.Dataset
            The dataset containing all variables
        country_params : dict
            Dictionary containing parameters for each country
        countries : list
            List of countries to analyze
        channels : list
            List of media channels to analyze

        Returns:
        --------
        dict
            Nested dictionary with structure {channel: {country: ROAS}}
        """
        roas_dict = {channel: {} for channel in channels}

        for country in countries:
            actual_y = float(dataset.target.sel(country=country).values.sum())

            for channel in channels:
                total_spend = float(dataset.spends.sel(country=country, channel=channel).values.sum())

                if total_spend > 0:
                    potential_y = self.generate_potential_y(
                        dataset=dataset,
                        country_params=country_params,
                        country=country,
                        channel=channel,
                        t0=dataset.coords['date'].values[0],
                        t1=dataset.coords['date'].values[-1]
                    )
                    potential_y_sum = float(potential_y.sum())
                    roas = (actual_y - potential_y_sum) / total_spend
                else:
                    roas = 0.0

                roas_dict[channel][country] = round(roas, 2)

        return roas_dict



    def plot_quarterly_roas(self, quarterly_roas, countries):
        """Plot quarterly ROAS for each country"""
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

    def plot_global_roas(self, global_roas):
        """Plot global ROAS for all countries and channels"""
        fig = plt.figure(figsize=(12, 6))

        # Pivot the data for plotting
        plot_data = global_roas.pivot(index='Channel', columns='Country', values='ROAS')

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

    def plot_potential_outcomes(self, dataset, country_params, country, channel, period='Q'):
        """Plot actual vs potential outcomes for each period"""
        dates = dataset.coords['date'].values
        df_dates = pd.DataFrame({'date': dates})
        df_dates[period] = pd.to_datetime(dates).to_period(period)
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

        for i, (idx, row) in enumerate(period_ranges.iterrows()):
            ax = axes[i]
            t0, t1 = row['date'][['min', 'max']]

            # Generate potential outcome
            potential_y = self.generate_potential_y(
                dataset=dataset,
                country_params=country_params,
                country=country,
                channel=channel,
                t0=t0,
                t1=t1
            )

            # Plot actual vs potential
            actual_y = dataset.target.sel(country=country)
            ax.plot(dates, actual_y, 'k-', label='Actual')
            ax.plot(dates, potential_y, 'b--', label='Potential')

            # Add mask period
            mask_period = np.logical_and(dates >= t0, dates <= t1)
            ax.axvspan(t0, t1, alpha=0.2, color='gray')

            ax.set_title(f'{period} {idx}')
            ax.legend()

        plt.tight_layout()
        return fig

    def compute_true_roas(self, dataset):
        """
        Compute true ROAS values from generated data

        Parameters:
        -----------
        dataset : xarray.Dataset
            The generated dataset containing media_response and spends

        Returns:
        --------
        dict
            Nested dictionary with true ROAS values by channel and country
        """
        countries = dataset.coords['country'].values
        channels = dataset.coords['channel'].values
        true_roas = {channel: {} for channel in channels}

        for channel in channels:
            for country in countries:
                spends = dataset.spends.sel(country=country, channel=channel)
                total_spend = float(spends.sum().values)

                if total_spend > 0:
                    # Get media response for this channel
                    media_response = dataset.media_response.sel(
                        country=country,
                        channel=channel
                    ).sum().values

                    # Calculate ROAS
                    roas = float(media_response) / total_spend
                else:
                    roas = 0.0

                true_roas[channel][country] = round(roas, 2)

        return true_roas
