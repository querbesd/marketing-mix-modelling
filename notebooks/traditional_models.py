import numpy as np
import pandas as pd
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
import optuna


class MMM_Base:
    """Base class for Marketing Mix Models"""

    def __init__(self, alpha_ranges=None, lambda_ranges=None, l_max=8):
        self.alpha_ranges = alpha_ranges
        self.lambda_ranges = lambda_ranges
        self.l_max = l_max
        self.best_alphas = None
        self.best_params = None
        self.apply_saturation = False

    def _apply_adstock(self, X, alphas, media_cols):
        """Common adstock transformation"""
        X_adstocked = X.copy()

        for col in media_cols:
            if col in X.columns and col in alphas:
                x_values = X[col].values
                adstocked = geometric_adstock(x_values, alpha=alphas[col], l_max=self.l_max, normalize=True).eval()
                X_adstocked[col] = adstocked

        return X_adstocked

    def _apply_saturation(self, X, lambdas, media_cols):
        """Common saturation transformation"""
        X_saturated = X.copy()

        for col in media_cols:
            if col in X.columns and col in lambdas:
                x_values = X[col].values
                saturated = logistic_saturation(x_values, lam=lambdas[col]).eval()
                X_saturated[col] = saturated

        return X_saturated

    def _apply_transformation(self, X, alphas, lambdas, media_cols):
        """Apply adstock and optional saturation transformations to media columns"""
        X_transformed = X.copy()

        for col in media_cols:
            if col in X.columns and col in alphas:
                # Apply adstock transformation
                x_values = X[col].values
                adstocked = geometric_adstock(x_values, alpha=alphas[col], l_max=self.l_max, normalize=True).eval()
                X_transformed[col] = adstocked

                # Apply saturation if enabled
                if self.apply_saturation and col in lambdas:
                    saturated = logistic_saturation(X_transformed[col].values, lam=lambdas[col]).eval()
                    X_transformed[col] = saturated

        return X_transformed



    # def rssd(self, effect_share, spend_share):
    #     """Calculate Root Sum Square Distance between effect and spend shares"""
    #     return np.sqrt(np.sum((effect_share - spend_share) ** 2))


    def reset(self):
        """Clear cached computations and reset model to initial state"""
        self.best_alphas = None
        self.best_params = None
        # Clear any other instance variables that should be reset
        if hasattr(self, 'X_full_transformed'):
            del self.X_full_transformed
        if hasattr(self, 'X_full_adstocked'):
            del self.X_full_adstocked
        if hasattr(self, 'feature_names'):
            del self.feature_names


    def optimize(self, X, y, media_cols, n_trials=50, cv_splits=3, test_size=20, is_multiobjective=False):
        """
        Optimize model parameters using Optuna

        Parameters:
        -----------
        X : DataFrame
            Feature dataset
        y : Series
            Target variable
        media_cols : list
            List of media columns to apply adstock to
        n_trials : int, default=50
            Number of Optuna trials
        cv_splits : int, default=3
            Number of cross-validation splits
        test_size : int, default=20
            Size of test set in cross-validation
        is_multiobjective : bool, default=True
            Whether to use multi-objective optimization (RMSE and RSSD)

        Returns:
        --------
        optuna.study.Study
            Optuna study object with optimization results
        """
        # Reset model state to ensure a clean start
        self.reset()

        # Setup time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_splits, test_size=test_size)

        # Create study - different for single vs multi-objective
        if is_multiobjective:
            study = optuna.create_study(
                directions=["minimize", "minimize"],
                sampler=optuna.samplers.NSGAIISampler(seed=42)
            )
        else:
            study = optuna.create_study(direction="minimize")

        # Run optimization
        study.optimize(
            lambda trial: self._objective(trial, X, y, media_cols, tscv, is_multiobjective),
            n_trials=n_trials,
            show_progress_bar=False
        )

        # Get best parameters - handle differently for multi-objective
        if is_multiobjective:
            # Get all Pareto-optimal solutions
            best_trials = study.best_trials
            print(f"Number of best_trials: {len(best_trials)}")

            # Calculate best possible metrics
            min_rmse = min(trial.values[0] for trial in best_trials)
            min_rssd = min(trial.values[1] for trial in best_trials)
            print(f"Best possible RMSE: {min_rmse:.4f}, Best possible RSSD: {min_rssd:.4f}")

            # Average parameters from all Pareto-optimal models
            avg_params = {}
            param_counts = {}

            for trial in best_trials:
                for param_name, param_value in trial.params.items():
                    # Handle numeric parameters
                    if isinstance(param_value, (int, float)):
                        if param_name not in avg_params:
                            avg_params[param_name] = 0.0
                            param_counts[param_name] = 0

                        avg_params[param_name] += param_value
                        param_counts[param_name] += 1

            # Calculate averages
            for param_name in avg_params:
                avg_params[param_name] /= param_counts[param_name]

                # Convert back to integer if the original parameter was an integer
                # (This is important for parameters like n_estimators)
                if param_name in best_trials[0].params and isinstance(best_trials[0].params[param_name], int):
                    avg_params[param_name] = int(round(avg_params[param_name]))

            print("Using averaged parameters from Pareto front:")
            for param_name, param_value in avg_params.items():
                print(f"  {param_name}: {param_value}")

            self.best_params = avg_params
        else:
            # Get best parameters from single-objective optimization
            self.best_params = study.best_params

        # Extract best adstock parameters
        self.best_alphas = {
            col: self.best_params[f"alpha_{col}"]
            for col in media_cols
        }

        return study

    def _objective(self, trial, X, y, media_cols, tscv, is_multiobjective=False):
        """Optuna objective function - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _objective method")

    def fit(self, X_full, y_train, train_indices, media_cols=None):
        """Fit model - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(self, test_indices=None, X_new=None, media_cols=None):
        """Make predictions - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement predict method")

    def score(self, y_test, test_indices=None, X_new=None, y_new=None, media_cols=None):
        """Calculate R² score - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement score method")


class MMM_Linear(MMM_Base):
    """Linear model with optional scaling"""

    def __init__(self, alpha_ranges=None, lambda_ranges=None, l_max=8, use_scaling=False, apply_saturation=False, model_type='lr'):
        super().__init__(alpha_ranges, lambda_ranges, l_max)
        self.use_scaling = use_scaling
        self.model_type = model_type
        self.model = Ridge(alpha=1.0)
        self.scaler = StandardScaler() if use_scaling else None
        self.apply_saturation = apply_saturation

    def reset(self):
        """Clear cached computations and reset model to initial state"""
        super().reset()
        self.model = Ridge(alpha=1.0)
        if self.use_scaling:
            self.scaler = StandardScaler()

    # def _calculate_spend_effect_share(self, model, X_train, media_cols, X_train_original=None):
    #     """
    #     Calculate spend effect share for media channels using linear model coefficients

    #     Args:
    #         model: fitted Ridge model
    #         X_train: training data (possibly scaled)
    #         media_cols: list of media channel names
    #         X_train_original: original unscaled data (if scaling is used)

    #     Returns:
    #         effect_share: array of effect shares for media channels
    #         spend_share: array of spend shares for media channels
    #     """
    #     # Get model coefficients
    #     coefficients = model.coef_

    #     # Create a dict of column index to coefficient
    #     coef_dict = {col: coef for col, coef in zip(X_train.columns, coefficients)}

    #     # Calculate effect using |coefficient × mean(feature)|
    #     # If scaling is used, we need to adjust by the standard deviation
    #     media_effects = {}
    #     for col in media_cols:
    #         if self.use_scaling and self.scaler is not None:
    #             # Find the index of this column in the scaler
    #             col_idx = list(X_train.columns).index(col)
    #             # Get the standard deviation used for scaling
    #             col_std = self.scaler.scale_[col_idx]
    #             # Use the standard deviation to adjust the effect
    #             media_effects[col] = abs(coef_dict[col] * col_std)
    #         else:
    #             # For unscaled data, use the mean of the feature
    #             media_effects[col] = abs(coef_dict[col] * X_train[col].mean())

    #     # Calculate effect share
    #     total_effect = sum(media_effects.values())
    #     if total_effect > 0:
    #         effect_share = np.array([media_effects[col]/total_effect for col in media_cols])
    #     else:
    #         # Fallback to equal effect shares if all effects are zero
    #         effect_share = np.ones(len(media_cols)) / len(media_cols)

    #     # Calculate spend share from original data
    #     X_for_spend = X_train_original if X_train_original is not None else X_train
    #     total_spend = X_for_spend[media_cols].sum().sum()
    #     if total_spend > 0:
    #         spend_share = np.array([X_for_spend[col].sum()/total_spend for col in media_cols])
    #     else:
    #         # Fallback to equal spend shares if all spends are zero
    #         spend_share = np.ones(len(media_cols)) / len(media_cols)

    #     return effect_share, spend_share

    def _objective(self, trial, X, y, media_cols, tscv, is_multiobjective=False):
        """Optuna objective function for linear model"""
        # Suggest adstock parameters
        alphas = {}
        lambdas = {}
        for col in media_cols:
            min_val, max_val = self.alpha_ranges.get(col, (0.1, 0.7))
            alphas[col] = trial.suggest_float(f"alpha_{col}", min_val, max_val)
            if self.apply_saturation:
                # Get lambda range from self.lambda_ranges if available, otherwise use default
                lambda_min, lambda_max = self.lambda_ranges.get(col, (0, 12)) if self.lambda_ranges else (0, 12)
                lambdas[col] = trial.suggest_int(f"lambda_{col}", lambda_min, lambda_max)

        # Apply transformation (adstock and optional saturation)
        X_transformed = self._apply_transformation(X, alphas, lambdas, media_cols)

        # Suggest model parameters
        linear_alpha = trial.suggest_float("linear_alpha", 0.01, 10.0, log=True)

        # Cross-validation
        rmses = []
        rssds = []  # For multi-objective

        for train_idx, val_idx in tscv.split(X_transformed):
            try:
                X_train, X_val = X_transformed.iloc[train_idx], X_transformed.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Scale features if needed
                if self.use_scaling:
                    scaler = StandardScaler()
                    X_train_scaled = pd.DataFrame(
                        scaler.fit_transform(X_train),
                        columns=X_train.columns,
                        index=X_train.index
                    )
                    X_val_scaled = pd.DataFrame(
                        scaler.transform(X_val),
                        columns=X_val.columns,
                        index=X_val.index
                    )
                else:
                    X_train_scaled = X_train
                    X_val_scaled = X_val

                # Train linear model
                model = Ridge(alpha=linear_alpha)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)

                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                rmses.append(rmse)

                # For multi-objective optimization, calculate RSSD
                if is_multiobjective:
                    try:
                        # Calculate effect and spend shares
                        effect_share, spend_share = self._calculate_spend_effect_share(
                            model=model,
                            X_train=X_val_scaled,
                            media_cols=media_cols,
                            X_train_original=X_train
                        )

                        # Calculate RSSD
                        decomp_rssd = self.rssd(effect_share, spend_share)
                        rssds.append(decomp_rssd)
                    except Exception as e:
                        print(f"Error in RSSD calculation: {e}")
                        # Use a high RSSD value to discourage this parameter set
                        rssds.append(1.0)  # Max RSSD is 1.0 for normalized shares
            except Exception as e:
                print(f"Error in cross-validation: {e}")
                # Return high error values
                if is_multiobjective:
                    return 1e10, 1e10
                else:
                    return 1e10

        # Store trial attributes
        trial.set_user_attr("adstock_alphas", alphas)
        if self.apply_saturation:
            trial.set_user_attr("saturation_lambdas", lambdas)
        trial.set_user_attr("linear_alpha", linear_alpha)

        if is_multiobjective:
            return np.mean(rmses), np.mean(rssds)
        else:
            return np.mean(rmses)

    def fit(self, X_full, y_train, train_indices, media_cols=None):
        """Fit the linear model with the best parameters"""
        if media_cols is not None and self.best_alphas is not None:
            # Extract best lambdas if saturation is applied
            best_lambdas = {}
            if self.apply_saturation:
                best_lambdas = {
                    col: self.best_params.get(f"lambda_{col}", 0)
                    for col in media_cols
                    if f"lambda_{col}" in self.best_params
                }

            # Apply transformation (adstock and optional saturation)
            X_full_transformed = self._apply_transformation(X_full, self.best_alphas, best_lambdas, media_cols)
        else:
            X_full_transformed = X_full.copy()

        # Extract training data
        X_train = X_full_transformed.iloc[train_indices]

        # Scale data if needed
        if self.use_scaling:
            self.scaler = StandardScaler().fit(X_train)
            X_train_scaled = pd.DataFrame(
                self.scaler.transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
        else:
            X_train_scaled = X_train

        # Initialize and fit model
        linear_alpha = self.best_params.get("linear_alpha", 1.0) if hasattr(self, 'best_params') else 1.0
        self.model = Ridge(alpha=linear_alpha)
        self.model.fit(X_train_scaled, y_train)

        # Store full transformed dataset for later prediction
        self.X_full_transformed = X_full_transformed

        # Store feature names for later use
        self.feature_names = X_train.columns

        return self
    @property
    def coef_(self):
        """Return the linear model coefficients"""
        return self.model.coef_

    @property
    def intercept_(self):
        """Return the linear model intercept"""
        return self.model.intercept_


    def predict(self, test_indices=None, X_new=None, media_cols=None, zero_media_columns=None):
        """
        Make predictions with the linear model

        Parameters:
        -----------
        test_indices : array-like, optional
            Indices to use from the stored full dataset
        X_new : DataFrame, optional
            New data to use instead of stored data
        media_cols : list, optional
            List of media columns, needed if X_new is provided
        zero_media_columns : list, optional
            List of media columns to zero out in the prediction
        """
        if test_indices is not None:
            # Use indices to get the test set from the stored full dataset
            X = self.X_full_transformed.iloc[test_indices].copy()  # Make a copy to avoid modifying original

            # Zero out specified media columns if requested
            if zero_media_columns is not None:
                # Identify all columns related to the specified media channels
                for channel in zero_media_columns:
                    # Zero out the original channel if it exists
                    if channel in X.columns:
                        X[channel] = 0

                    # Zero out derived features (adstocked and saturated versions)
                    col_patterns = [
                        f"{channel}_adstocked",
                        f"{channel}_adstocked_saturated"
                    ]

                    for col in X.columns:
                        if any(pattern in col for pattern in col_patterns):
                            X[col] = 0

        elif X_new is not None:
            # Make a copy of the input data
            X_to_process = X_new.copy()

            # Zero out specified media columns in the input data if requested
            if zero_media_columns is not None:
                for channel in zero_media_columns:
                    if channel in X_to_process.columns:
                        X_to_process[channel] = 0

            # Apply transformation to new data if needed
            if media_cols is not None and self.best_alphas is not None:
                # Extract best lambdas if saturation is applied
                best_lambdas = {}
                if self.apply_saturation:
                    best_lambdas = {
                        col: self.best_params.get(f"lambda_{col}", 0)
                        for col in media_cols
                        if f"lambda_{col}" in self.best_params
                    }

                # Apply transformation (adstock and optional saturation)
                X = self._apply_transformation(X_to_process, self.best_alphas, best_lambdas, media_cols)
            else:
                X = X_to_process
        else:
            raise ValueError("Either test_indices or X_new must be provided")

        # Scale the features if needed
        if self.use_scaling and self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X

        # Make predictions
        return self.model.predict(X_scaled)

    def score(self, y_test, test_indices=None, X_new=None, y_new=None, media_cols=None):
        """Calculate R² score"""
        if test_indices is not None:
            y_pred = self.predict(test_indices=test_indices)
            return r2_score(y_test, y_pred)
        elif X_new is not None and y_new is not None:
            y_pred = self.predict(X_new=X_new, media_cols=media_cols)
            return r2_score(y_new, y_pred)
        else:
            raise ValueError("Either (test_indices, y_test) or (X_new, y_new) must be provided")



class MMM_TreeBased(MMM_Base):
    """Tree-based model"""

    def __init__(self, alpha_ranges=None, l_max=8, model_type='gb'):
        super().__init__(alpha_ranges, l_max)
        self.model_type = model_type
        if model_type == 'gb':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        elif model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                random_state=42
            )
        else:
            raise ValueError("model_type must be 'gb' or 'rf'")


    # def _calculate_spend_effect_share(self, model, X_train, media_cols):
    #     """
    #     Calculate spend effect share for media channels

    #     Args:
    #         model: trained tree-based model
    #         X_train: training data
    #         media_cols: list of media channel names

    #     Returns:
    #         effect_share: array of effect shares for media channels
    #         spend_share: array of spend shares for media channels
    #     """
    #     # Try to use SHAP values if available
    #     try:
    #         import shap
    #         explainer = shap.TreeExplainer(model)
    #         shap_values = explainer.shap_values(X_train)

    #         # Calculate absolute SHAP values for each media feature
    #         media_effects = {}
    #         for i, col in enumerate(X_train.columns):
    #             if col in media_cols:
    #                 media_effects[col] = np.abs(shap_values[:, i]).sum()
    #     except Exception:
    #         # Fall back to feature importance
    #         importances = model.feature_importances_

    #         media_effects = {}
    #         for i, col in enumerate(X_train.columns):
    #             if col in media_cols:
    #                 media_effects[col] = importances[i]

    #     # Calculate effect share
    #     if sum(media_effects.values()) > 0:
    #         total_effect = sum(media_effects.values())
    #         effect_share = np.array([media_effects[col]/total_effect for col in media_cols])
    #     else:
    #         # Fallback to equal effect shares if all effects are zero
    #         effect_share = np.ones(len(media_cols)) / len(media_cols)

    #     # Calculate spend share from original data
    #     total_spend = X_train[media_cols].sum().sum()
    #     if total_spend > 0:
    #         spend_share = np.array([X_train[col].sum()/total_spend for col in media_cols])
    #     else:
    #         # Fallback to equal spend shares if all spends are zero
    #         spend_share = np.ones(len(media_cols)) / len(media_cols)

    #     return effect_share, spend_share

    def _objective(self, trial, X, y, media_cols, tscv, is_multiobjective=False):
        """Optuna objective function for tree-based model"""
        # Suggest adstock parameters
        alphas = {}
        for col in media_cols:
            min_val, max_val = self.alpha_ranges.get(col, (0.1, 0.7))
            alphas[col] = trial.suggest_float(f"alpha_{col}", min_val, max_val)

        # Apply adstock transformation
        X_adstocked = self._apply_adstock(X, alphas, media_cols)

        # Suggest model parameters based on model type
        if self.model_type == 'gb':
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 2, 6)

            # Create model
            tree_model_params = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'random_state': 42
            }
            model_class = GradientBoostingRegressor
        else:  # RF
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 3, 20)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

            # Create model
            tree_model_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'random_state': 42
            }
            model_class = RandomForestRegressor

        # Cross-validation
        rmses = []
        rssds = []  # For multi-objective

        for train_idx, val_idx in tscv.split(X_adstocked):
            try:
                X_train, X_val = X_adstocked.iloc[train_idx], X_adstocked.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Train tree model
                model = model_class(**tree_model_params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                rmses.append(rmse)

                # For multi-objective optimization, calculate RSSD
                if is_multiobjective:
                    try:
                        # Calculate effect and spend shares
                        effect_share, spend_share = self._calculate_spend_effect_share(
                            model=model,
                            X_train=X_val,
                            media_cols=media_cols
                        )

                        # Calculate RSSD
                        decomp_rssd = self.rssd(effect_share, spend_share)
                        rssds.append(decomp_rssd)
                    except Exception as e:
                        print(f"Error in RSSD calculation: {e}")
                        # Use a high RSSD value to discourage this parameter set
                        rssds.append(1.0)  # Max RSSD is 1.0 for normalized shares
            except Exception as e:
                print(f"Error in cross-validation: {e}")
                # Return high error values
                if is_multiobjective:
                    return 1e10, 1e10
                else:
                    return 1e10

        # Store trial attributes
        trial.set_user_attr("adstock_alphas", alphas)
        trial.set_user_attr("tree_params", tree_model_params)

        if is_multiobjective:
            return np.mean(rmses), np.mean(rssds)
        else:
            return np.mean(rmses)

    def fit(self, X_full, y_train, train_indices, media_cols=None):
        """Fit the tree-based model with the best parameters"""
        if media_cols is not None and self.best_alphas is not None:
            # Apply adstock transformation to full dataset
            X_full_adstocked = self._apply_adstock(X_full, self.best_alphas, media_cols)
        else:
            X_full_adstocked = X_full.copy()

        # Extract training data
        X_train = X_full_adstocked.iloc[train_indices]

        # Configure model parameters
        if self.model_type == 'gb':
            model_params = {
                'n_estimators': self.best_params.get("n_estimators", 100),
                'learning_rate': self.best_params.get("learning_rate", 0.1),
                'max_depth': self.best_params.get("max_depth", 3),
                'random_state': 42
            } if hasattr(self, 'best_params') else {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            }
            self.model = GradientBoostingRegressor(**model_params)
        else:  # RF
            model_params = {
                'n_estimators': self.best_params.get("n_estimators", 100),
                'max_depth': self.best_params.get("max_depth", None),
                'min_samples_split': self.best_params.get("min_samples_split", 2),
                'min_samples_leaf': self.best_params.get("min_samples_leaf", 1),
                'random_state': 42
            } if hasattr(self, 'best_params') else {
                'n_estimators': 100,
                'max_depth': None,
                'random_state': 42
            }
            self.model = RandomForestRegressor(**model_params)

        # Fit the model
        self.model.fit(X_train, y_train)

        # Store full transformed dataset for later prediction
        self.X_full_adstocked = X_full_adstocked

        # Store feature names for later use
        self.feature_names = X_train.columns

        return self

    def predict(self, test_indices=None, X_new=None, media_cols=None, zero_media_columns=None):
        """
        Make predictions with the tree-based model

        Parameters:
        -----------
        test_indices : array-like, optional
            Indices to use from the stored full dataset
        X_new : DataFrame, optional
            New data to use instead of stored data
        media_cols : list, optional
            List of media columns, needed if X_new is provided
        zero_media_columns : list, optional
            List of media columns to zero out in the prediction
        """
        if test_indices is not None:
            # Use indices to get the test set from the stored full dataset
            X = self.X_full_adstocked.iloc[test_indices].copy()  # Make a copy to avoid modifying original

            # Zero out specified media columns if requested
            if zero_media_columns is not None:
                # Identify all columns related to the specified media channels
                for channel in zero_media_columns:
                    # Zero out the original channel and any derived features
                    for col in X.columns:
                        if col == channel or (f"{channel}_" in col):
                            X[col] = 0

        elif X_new is not None:
            # Make a copy of the input data
            X_to_process = X_new.copy()

            # Zero out specified media columns in the input data if requested
            if zero_media_columns is not None:
                for channel in zero_media_columns:
                    if channel in X_to_process.columns:
                        X_to_process[channel] = 0

            # Apply adstock to the data (with zeroed channels if specified)
            if media_cols is not None and self.best_alphas is not None:
                X = self._apply_adstock(X_to_process, self.best_alphas, media_cols)
            else:
                X = X_to_process
        else:
            raise ValueError("Either test_indices or X_new must be provided")

        # Make predictions
        return self.model.predict(X)

    def score(self, y_test, test_indices=None, X_new=None, y_new=None, media_cols=None):
        """Calculate R² score"""
        if test_indices is not None:
            y_pred = self.predict(test_indices=test_indices)
            return r2_score(y_test, y_pred)
        elif X_new is not None and y_new is not None:
            y_pred = self.predict(X_new=X_new, media_cols=media_cols)
            return r2_score(y_new, y_pred)
        else:
            raise ValueError("Either (test_indices, y_test) or (X_new, y_new) must be provided")

class MMM_EnsembleBased(MMM_Base):
    """
    Stacked ensemble model combining linear and tree-based models

    This class implements a two-stage ensemble:
    1. First, a linear model (Ridge) captures the main trends and relationships
    2. Then, a tree-based model is trained on the residuals to capture nonlinear patterns
    """

    def __init__(self, alpha_ranges=None, l_max=8, model_type='rf'):
        super().__init__(alpha_ranges, l_max)
        self.model_type = model_type
        self.linear_model = Ridge(alpha=1.0)

        # Initialize tree model based on type
        if model_type == 'gb':
            self.tree_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        elif model_type == 'rf':
            self.tree_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                random_state=42
            )
        else:
            raise ValueError("model_type must be 'gb' or 'rf'")

    def reset(self):
        """Clear cached computations and reset model to initial state"""
        super().reset()
        self.linear_model = Ridge(alpha=1.0)

        if self.model_type == 'gb':
            self.tree_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        elif self.model_type == 'rf':
            self.tree_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                random_state=42
            )

    # def _calculate_spend_effect_share(self, X_train, media_cols):
    #     """
    #     Calculate spend effect share for media channels using SHAP decomposition
    #     of the full stacked model.

    #     Args:
    #         X_train: training data (pandas DataFrame)
    #         media_cols: list of media channel names

    #     Returns:
    #         effect_share: array of effect shares for media channels
    #         spend_share: array of spend shares for media channels
    #     """
    #     import shap
    #     import numpy as np

    #     # Create a wrapper function that combines predictions from both models
    #     def stacked_model_predict(X):
    #         # Convert to numpy for consistency if it's not already
    #         if isinstance(X, pd.DataFrame):
    #             X = X.values

    #         # Reshape if needed (KernelExplainer sometimes passes 1D arrays)
    #         if len(X.shape) == 1:
    #             X = X.reshape(1, -1)

    #         # Create DataFrame with proper column names if X is numpy array
    #         if not isinstance(X, pd.DataFrame):
    #             X_df = pd.DataFrame(X, columns=X_train.columns)
    #         else:
    #             X_df = X

    #         # Get predictions from linear model
    #         linear_preds = self.linear_model.predict(X_df)

    #         # Get predictions from tree model
    #         tree_preds = self.tree_model.predict(X_df)

    #         # Return combined predictions (linear + residual correction)
    #         return linear_preds + tree_preds

    #     # Create background dataset for KernelExplainer
    #     # Using a subset of training data for efficiency
    #     if len(X_train) > 100:
    #         background = shap.sample(X_train, 100, random_state=42)
    #     else:
    #         background = X_train

    #     # Initialize KernelExplainer for the stacked model
    #     explainer = shap.KernelExplainer(stacked_model_predict, background)

    #     # Calculate SHAP values
    #     # Use nsamples parameter to control computation time vs. accuracy tradeoff
    #     shap_values = explainer.shap_values(X_train, nsamples=100, silent=True)

    #     # Calculate absolute SHAP values for each media feature
    #     media_effects = {}
    #     for col in media_cols:
    #         col_idx = list(X_train.columns).index(col)
    #         # Sum of absolute SHAP values indicates overall feature importance
    #         media_effects[col] = np.abs(shap_values[:, col_idx]).sum()

    #     # Calculate effect share
    #     total_effect = sum(media_effects.values())
    #     effect_share = np.array([media_effects[col]/total_effect for col in media_cols])

    #     # Calculate spend share from original data
    #     total_spend = X_train[media_cols].sum().sum()
    #     spend_share = np.array([X_train[col].sum()/total_spend for col in media_cols])

    #     return effect_share, spend_share

    def _objective(self, trial, X, y, media_cols, tscv, is_multiobjective=False):
        """Optuna objective function for stacked ensemble model"""
        # Suggest adstock parameters
        alphas = {}
        for col in media_cols:
            min_val, max_val = self.alpha_ranges.get(col, (0.1, 0.7))
            alphas[col] = trial.suggest_float(f"alpha_{col}", min_val, max_val)

        # Apply adstock transformation
        X_adstocked = self._apply_adstock(X, alphas, media_cols)

        # Suggest linear model parameters
        linear_alpha = trial.suggest_float("linear_alpha", 0.01, 10.0, log=True)

        # Suggest tree model parameters based on model type
        if self.model_type == 'gb':
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 2, 6)

            # Create tree model
            tree_model_params = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'random_state': 42
            }
            tree_model_class = GradientBoostingRegressor
        else:  # RF
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 3, 20)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

            # Create tree model
            tree_model_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'random_state': 42
            }
            tree_model_class = RandomForestRegressor

        # Cross-validation
        rmses = []
        rssds = []  # For multi-objective

        for train_idx, val_idx in tscv.split(X_adstocked):
            try:
                X_train, X_val = X_adstocked.iloc[train_idx], X_adstocked.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Train linear model
                linear_model = Ridge(alpha=linear_alpha)
                linear_model.fit(X_train, y_train)

                # Get linear model predictions
                y_pred_linear_train = linear_model.predict(X_train)
                y_pred_linear_val = linear_model.predict(X_val)

                # Train tree model on residuals
                residuals = y_train - y_pred_linear_train
                tree_model = tree_model_class(**tree_model_params)
                tree_model.fit(X_train, residuals)

                # Make stacked predictions
                y_pred = y_pred_linear_val + tree_model.predict(X_val)

                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                rmses.append(rmse)

                # For multi-objective optimization, calculate RSSD
                if is_multiobjective:
                    try:
                        # Store models temporarily to use in _calculate_spend_effect_share
                        self.linear_model = linear_model
                        self.tree_model = tree_model

                        # Calculate effect and spend shares with the new method signature
                        effect_share, spend_share = self._calculate_spend_effect_share(
                            X_train=X_val,
                            media_cols=media_cols
                        )

                        # Calculate RSSD
                        decomp_rssd = self.rssd(effect_share, spend_share)
                        rssds.append(decomp_rssd)
                    except Exception as e:
                        print(f"Error in RSSD calculation: {e}")
                        # Use a high RSSD value to discourage this parameter set
                        rssds.append(1.0)  # Max RSSD is 1.0 for normalized shares
            except Exception as e:
                print(f"Error in cross-validation: {e}")
                # Return high error values
                if is_multiobjective:
                    return 1e10, 1e10
                else:
                    return 1e10

        # Store trial attributes
        trial.set_user_attr("adstock_alphas", alphas)
        trial.set_user_attr("linear_alpha", linear_alpha)
        trial.set_user_attr("tree_params", tree_model_params)

        if is_multiobjective:
            return np.mean(rmses), np.mean(rssds)
        else:
            return np.mean(rmses)

    def fit(self, X_full, y_train, train_indices, media_cols=None):
        """Fit the stacked ensemble model with the best parameters"""
        if media_cols is not None and self.best_alphas is not None:
            # Apply adstock transformation to full dataset
            X_full_adstocked = self._apply_adstock(X_full, self.best_alphas, media_cols)
        else:
            X_full_adstocked = X_full.copy()

        # Extract training data
        X_train = X_full_adstocked.iloc[train_indices]

        # Configure linear model
        linear_alpha = self.best_params.get("linear_alpha", 1.0) if hasattr(self, 'best_params') else 1.0
        self.linear_model = Ridge(alpha=linear_alpha)

        # Fit linear model
        self.linear_model.fit(X_train, y_train)

        # Get linear predictions
        y_pred_linear = self.linear_model.predict(X_train)

        # Train tree model on residuals
        residuals = y_train - y_pred_linear

        # Configure tree model
        if self.model_type == 'gb':
            tree_params = {
                'n_estimators': self.best_params.get("n_estimators", 100),
                'learning_rate': self.best_params.get("learning_rate", 0.1),
                'max_depth': self.best_params.get("max_depth", 3),
                'random_state': 42
            } if hasattr(self, 'best_params') else {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            }
            self.tree_model = GradientBoostingRegressor(**tree_params)
        else:  # RF
            tree_params = {
                'n_estimators': self.best_params.get("n_estimators", 100),
                'max_depth': self.best_params.get("max_depth", None),
                'min_samples_split': self.best_params.get("min_samples_split", 2),
                'min_samples_leaf': self.best_params.get("min_samples_leaf", 1),
                'random_state': 42
            } if hasattr(self, 'best_params') else {
                'n_estimators': 100,
                'max_depth': None,
                'random_state': 42
            }
            self.tree_model = RandomForestRegressor(**tree_params)

        # Fit tree model on residuals
        self.tree_model.fit(X_train, residuals)

        # Store full transformed dataset for later prediction
        self.X_full_adstocked = X_full_adstocked

        # Store feature names for later use
        self.feature_names = X_train.columns

        return self

    def predict(self, test_indices=None, X_new=None, media_cols=None, zero_media_columns=None):
        """
        Make predictions with the stacked ensemble model

        Parameters:
        -----------
        test_indices : array-like, optional
            Indices to use from the stored full dataset
        X_new : DataFrame, optional
            New data to use instead of stored data
        media_cols : list, optional
            List of media columns, needed if X_new is provided
        zero_media_columns : list, optional
            List of media columns to zero out in the prediction
        """
        if test_indices is not None:
            # Use indices to get the test set from the stored full dataset
            X = self.X_full_adstocked.iloc[test_indices].copy()  # Make a copy to avoid modifying original

            # Zero out specified media columns if requested
            if zero_media_columns is not None:
                # Identify all columns related to the specified media channels
                for channel in zero_media_columns:
                    # Zero out the original channel and any derived features
                    for col in X.columns:
                        if col == channel or (f"{channel}_" in col):
                            X[col] = 0

        elif X_new is not None:
            # Make a copy of the input data
            X_to_process = X_new.copy()

            # Zero out specified media columns in the input data if requested
            if zero_media_columns is not None:
                for channel in zero_media_columns:
                    if channel in X_to_process.columns:
                        X_to_process[channel] = 0

            # Apply adstock to the data (with zeroed channels if specified)
            if media_cols is not None and self.best_alphas is not None:
                X = self._apply_adstock(X_to_process, self.best_alphas, media_cols)
            else:
                X = X_to_process
        else:
            raise ValueError("Either test_indices or X_new must be provided")

        # Make stacked predictions (linear + tree_model on residuals)
        y_pred_linear = self.linear_model.predict(X)
        y_pred_tree = self.tree_model.predict(X)

        return y_pred_linear + y_pred_tree

    def score(self, y_test, test_indices=None, X_new=None, y_new=None, media_cols=None):
        """Calculate R² score"""
        if test_indices is not None:
            y_pred = self.predict(test_indices=test_indices)
            return r2_score(y_test, y_pred)
        elif X_new is not None and y_new is not None:
            y_pred = self.predict(X_new=X_new, media_cols=media_cols)
            return r2_score(y_new, y_pred)
        else:
            raise ValueError("Either (test_indices, y_test) or (X_new, y_new) must be provided")
