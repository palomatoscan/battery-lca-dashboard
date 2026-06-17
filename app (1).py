import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


ORIGINAL_DATE_COL = 'Month_num'
DATE_COL = 'Month_index'
TEST_SIZE = 0.2
RANDOM_STATE = 42
LAGS = [1, 2, 3, 6, 12]
ROLLING_WINDOWS = [3, 6, 12]
EPS = 1e-6

COMPOSITION_COLS = [
    'NMC_share_%',
    'LFP_share_%',
    'LCO_share_%',
    'NCA_share_%'
]

FEEDSTOCK_INPUTS = [
    'Battery_mass_t',
    'NMC_share_%',
    'LFP_share_%',
    'LCO_share_%',
    'NCA_share_%'
]

PRE_RECYCLING_OUTPUTS = [
    'Ferrous_kg_t',
    'Aluminium_kg_t',
    'Copper_kg_t',
    'Mixed_plastics_kg_t',
    'Other_byproducts_kg_t'
]

PROCESS_INPUTS = [
    'Contamination_%',
    'State_of_charge_%',
    'Feed_rate_kg_h',
    'Sorting_eff_%',
    'Renewable_share_%'
]

ENERGY_RECOVERY_OUTPUTS = [
    'Overall_recovery_rate_%'
]

OPERATIONAL_INPUTS = [
    'Electricity_total_kWh_t',
    'Diesel_MJ_t',
    'Water_L_t',
    'Transport_km'
]

FINAL_TARGETS = [
    'Black_mass_kg_t',
    'GWP_service_kgCO2e_tBattery',
    'Water_use_m3_tBattery',
    'GWP_kgCO2e_kgBM',
    'Black_mass_purity_%',
    'Li_%_BM',
    'Co_%_BM',
    'Ni_%_BM',
    'Mn_%_BM'
]

DERIVED_ENERGY_COLS = [
    'Recovered_energy_kWh_t',
    'External_electricity_kWh_t',
    'Renewable_external_kWh_t',
    'Nonrenewable_external_kWh_t'
]

ALL_MODELLED_VARIABLES = (
    FEEDSTOCK_INPUTS +
    PRE_RECYCLING_OUTPUTS +
    PROCESS_INPUTS +
    ENERGY_RECOVERY_OUTPUTS +
    OPERATIONAL_INPUTS +
    FINAL_TARGETS
)

DISPLAY_VARIABLES = ALL_MODELLED_VARIABLES + DERIVED_ENERGY_COLS

PERCENTAGE_COLS = [
    'NMC_share_%',
    'LFP_share_%',
    'LCO_share_%',
    'NCA_share_%',
    'Contamination_%',
    'State_of_charge_%',
    'Sorting_eff_%',
    'Renewable_share_%',
    'Overall_recovery_rate_%',
    'Black_mass_purity_%',
    'Li_%_BM',
    'Co_%_BM',
    'Ni_%_BM',
    'Mn_%_BM'
]

MODEL_PARAMS = {
    'n_estimators': 700,
    'learning_rate': 0.025,
    'max_depth': 3,
    'min_child_weight': 3,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'reg_lambda': 3.0,
    'random_state': RANDOM_STATE,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'n_jobs': -1
}

FRIENDLY_NAMES = {
    'Battery_mass_t': 'Incoming battery mass',
    'NMC_share_%': 'NMC battery share',
    'LFP_share_%': 'LFP battery share',
    'LCO_share_%': 'LCO battery share',
    'NCA_share_%': 'NCA battery share',
    'Ferrous_kg_t': 'Ferrous metals separated before recycling',
    'Aluminium_kg_t': 'Aluminium separated before recycling',
    'Copper_kg_t': 'Copper separated before recycling',
    'Mixed_plastics_kg_t': 'Mixed plastics separated before recycling',
    'Other_byproducts_kg_t': 'Other byproducts separated before recycling',
    'Contamination_%': 'Material contamination',
    'State_of_charge_%': 'Battery state of charge',
    'Feed_rate_kg_h': 'Process feed rate',
    'Sorting_eff_%': 'Sorting efficiency',
    'Renewable_share_%': 'Renewable energy share',
    'Overall_recovery_rate_%': 'Battery energy recovery rate',
    'Electricity_total_kWh_t': 'Total electricity demand',
    'Recovered_energy_kWh_t': 'Recovered battery energy used internally',
    'External_electricity_kWh_t': 'Grid Energy Consumption',
    'Renewable_external_kWh_t': 'Renewable external electricity',
    'Nonrenewable_external_kWh_t': 'Non-renewable external electricity',
    'Diesel_MJ_t': 'Diesel consumption',
    'Water_L_t': 'Operational water consumption',
    'Transport_km': 'Transport distance',
    'Black_mass_kg_t': 'Black mass production',
    'GWP_service_kgCO2e_tBattery': 'Climate impact per tonne of battery',
    'Water_use_m3_tBattery': 'Water use per tonne of battery',
    'GWP_kgCO2e_kgBM': 'Climate impact per kg of black mass',
    'Black_mass_purity_%': 'Black mass purity',
    'Li_%_BM': 'Lithium content in black mass',
    'Co_%_BM': 'Cobalt content in black mass',
    'Ni_%_BM': 'Nickel content in black mass',
    'Mn_%_BM': 'Manganese content in black mass'
}


def friendly_name(variable):
    return FRIENDLY_NAMES.get(variable, variable)


def friendly_composition_strategy_details(strategy, reference=None):
    if strategy == 'direct_4_normalized':
        return {
            'Best strategy': 'Direct forecast of all battery types',
            'Reference battery type': 'Not applicable'
        }
    if strategy == 'direct_3_residual_nca':
        return {
            'Best strategy': 'Direct forecast with residual balance',
            'Reference battery type': 'NCA'
        }
    if isinstance(strategy, str) and strategy.startswith('alr_ref_'):
        if reference is None or pd.isna(reference):
            reference = strategy.replace('alr_ref_', '')
        reference_name = str(reference).replace('_share_%', '')
        return {
            'Best strategy': 'Log-ratio method',
            'Reference battery type': reference_name
        }
    return {
        'Best strategy': str(strategy),
        'Reference battery type': 'Not applicable'
    }


def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def build_model():
    return XGBRegressor(**MODEL_PARAMS)


def normalize_composition_df(df):
    df = df.copy()
    values = df[COMPOSITION_COLS].clip(lower=0)
    total = values.sum(axis=1)
    for col in COMPOSITION_COLS:
        df[col] = np.where(
            total == 0,
            100 / len(COMPOSITION_COLS),
            values[col] / total * 100
        )
    return df


def normalize_composition_dict(row):
    row = row.copy()
    values = np.array([max(0.0, float(row[col])) for col in COMPOSITION_COLS])
    total = values.sum()
    values = np.ones(len(COMPOSITION_COLS)) * 25.0 if total == 0 else values / total * 100.0
    for col, value in zip(COMPOSITION_COLS, values):
        row[col] = value
    return row


def add_energy_recovery_columns(df):
    df = df.copy()
    required = ['Electricity_total_kWh_t', 'Overall_recovery_rate_%', 'Renewable_share_%']
    if not all(col in df.columns for col in required):
        return df
    recovery_fraction = df['Overall_recovery_rate_%'].clip(lower=0, upper=100) / 100
    renewable_fraction = df['Renewable_share_%'].clip(lower=0, upper=100) / 100
    df['Recovered_energy_kWh_t'] = df['Electricity_total_kWh_t'] * recovery_fraction
    df['External_electricity_kWh_t'] = df['Electricity_total_kWh_t'] - df['Recovered_energy_kWh_t']
    df['Renewable_external_kWh_t'] = df['External_electricity_kWh_t'] * renewable_fraction
    df['Nonrenewable_external_kWh_t'] = df['External_electricity_kWh_t'] * (1 - renewable_fraction)
    return df


def clip_logical_bounds(target, value):
    value = float(value)
    if target.startswith('alr_'):
        return value
    value = max(0.0, value)
    if target in PERCENTAGE_COLS:
        value = min(value, 100.0)
    return value


def add_time_features(df):
    df = df.copy()
    df['time_trend'] = df[DATE_COL]
    df['month_sin'] = np.sin(2 * np.pi * df[DATE_COL] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df[DATE_COL] / 12)
    return df


def lag_feature_names(target):
    names = []
    for lag in LAGS:
        names.append(f'{target}_lag_{lag}')
    for window in ROLLING_WINDOWS:
        names.append(f'{target}_roll_mean_{window}')
        names.append(f'{target}_roll_std_{window}')
    return names


def add_lag_features_for_target(df, target):
    df = df.copy()
    for lag in LAGS:
        df[f'{target}_lag_{lag}'] = df[target].shift(lag)
    shifted = df[target].shift(1)
    for window in ROLLING_WINDOWS:
        df[f'{target}_roll_mean_{window}'] = shifted.rolling(window=window, min_periods=1).mean()
        df[f'{target}_roll_std_{window}'] = shifted.rolling(window=window, min_periods=2).std()
    return df


def make_future_lag_features(history_df, target):
    values = history_df[target].dropna().values
    features = {}
    if len(values) == 0:
        for lag in LAGS:
            features[f'{target}_lag_{lag}'] = 0.0
        for window in ROLLING_WINDOWS:
            features[f'{target}_roll_mean_{window}'] = 0.0
            features[f'{target}_roll_std_{window}'] = 0.0
        return features
    for lag in LAGS:
        features[f'{target}_lag_{lag}'] = values[-lag] if len(values) >= lag else values[-1]
    for window in ROLLING_WINDOWS:
        window_values = values[-window:]
        features[f'{target}_roll_mean_{window}'] = np.mean(window_values)
        features[f'{target}_roll_std_{window}'] = np.std(window_values, ddof=1) if len(window_values) >= 2 else 0.0
    return features


def fit_trend_model(X_time, y):
    trend_model = LinearRegression()
    trend_model.fit(X_time[[DATE_COL]], y)
    return trend_model


def split_for_validation(X, y, validation_type, horizon):
    if validation_type == 'random_structural':
        return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
    horizon = min(horizon, max(1, len(X) - max(LAGS) - 5))
    X_train = X.iloc[:-horizon]
    X_test = X.iloc[-horizon:]
    y_train = y.iloc[:-horizon]
    y_test = y.iloc[-horizon:]
    return X_train, X_test, y_train, y_test


def fit_hybrid_from_xy(X_train, y_train, feature_cols):
    trend_model = fit_trend_model(X_train[[DATE_COL]], y_train)
    train_trend = trend_model.predict(X_train[[DATE_COL]])
    y_train_residual = y_train - train_trend
    residual_model = build_model()
    residual_model.fit(X_train[feature_cols], y_train_residual, verbose=False)
    return {
        'trend_model': trend_model,
        'residual_model': residual_model,
        'feature_cols': feature_cols
    }


def predict_hybrid(model_pack, X):
    feature_cols = model_pack['feature_cols']
    trend_pred = model_pack['trend_model'].predict(X[[DATE_COL]])
    residual_pred = model_pack['residual_model'].predict(X[feature_cols])
    return trend_pred + residual_pred


def train_hybrid_model(df, target, exogenous_cols, stage_name, validation_type, horizon):
    work = add_time_features(df.copy())
    work = add_lag_features_for_target(work, target)
    feature_cols = [DATE_COL, 'time_trend', 'month_sin', 'month_cos'] + exogenous_cols + lag_feature_names(target)
    model_df = work[feature_cols + [target]].dropna().copy()
    if len(model_df) < max(10, horizon + 5):
        raise ValueError(f'Not enough valid rows to train {target}. Check missing values and history length.')
    X = model_df[feature_cols]
    y = model_df[target]
    X_train, X_test, y_train, y_test = split_for_validation(X, y, validation_type, horizon)
    validation_model_pack = fit_hybrid_from_xy(X_train, y_train, feature_cols)
    y_pred = pd.Series(predict_hybrid(validation_model_pack, X_test)).reset_index(drop=True)
    y_pred = y_pred.apply(lambda v: clip_logical_bounds(target, v))
    y_test_plot = y_test.reset_index(drop=True)
    rmse, mae, r2 = evaluate(y_test_plot, y_pred)
    metrics = {
        'Validation': validation_type,
        'Stage': stage_name,
        'Variable': target,
        'Horizon': horizon if validation_type == 'temporal_backtest' else np.nan,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    validation = {
        'Validation': validation_type,
        'Stage': stage_name,
        'Variable': target,
        'Horizon': horizon if validation_type == 'temporal_backtest' else np.nan,
        'Time': X_test[DATE_COL].reset_index(drop=True),
        'Actual': y_test_plot,
        'Predicted': y_pred
    }
    final_model_pack = fit_hybrid_from_xy(X, y, feature_cols)
    return final_model_pack, metrics, validation


def predict_one_step_hybrid(model_pack, row_dict, target):
    feature_cols = model_pack['feature_cols']
    X_future = pd.DataFrame([row_dict])[feature_cols]
    pred = predict_hybrid(model_pack, X_future)[0]
    return clip_logical_bounds(target, pred)


def alr_col(numerator, reference):
    return f'alr_{numerator}_over_{reference}'


def add_alr_columns(df, reference_col):
    df = df.copy()
    for col in COMPOSITION_COLS:
        if col == reference_col:
            continue
        new_col = alr_col(col, reference_col)
        df[new_col] = np.log((df[col].clip(lower=EPS)) / (df[reference_col].clip(lower=EPS)))
    return df


def reconstruct_from_alr(pred_logratios, reference_col):
    ratios = {}
    for col in COMPOSITION_COLS:
        ratios[col] = 1.0 if col == reference_col else np.exp(pred_logratios[alr_col(col, reference_col)])
    total_ratio = sum(ratios.values())
    return {col: ratios[col] / total_ratio * 100.0 for col in COMPOSITION_COLS}


def reconstruct_direct_4(pred_values):
    row = {col: max(0.0, float(pred_values[col])) for col in COMPOSITION_COLS}
    return normalize_composition_dict(row)


def reconstruct_direct_3_residual_nca(pred_values):
    row = {
        'NMC_share_%': max(0.0, float(pred_values['NMC_share_%'])),
        'LFP_share_%': max(0.0, float(pred_values['LFP_share_%'])),
        'LCO_share_%': max(0.0, float(pred_values['LCO_share_%']))
    }
    row['NCA_share_%'] = max(0.0, 100.0 - row['NMC_share_%'] - row['LFP_share_%'] - row['LCO_share_%'])
    return normalize_composition_dict(row)


def evaluate_composition_strategy(validations, strategy_name, horizon, reference_col=None):
    pred_parts = {}
    actual_parts = {}
    if strategy_name == 'direct_4_normalized':
        needed = COMPOSITION_COLS
    elif strategy_name == 'direct_3_residual_nca':
        needed = ['NMC_share_%', 'LFP_share_%', 'LCO_share_%']
    else:
        needed = [alr_col(col, reference_col) for col in COMPOSITION_COLS if col != reference_col]
    for validation in validations:
        if (
            validation['Validation'] == 'temporal_backtest'
            and validation['Horizon'] == horizon
            and validation['Variable'] in needed
        ):
            pred_parts[validation['Variable']] = validation['Predicted'].reset_index(drop=True)
            actual_parts[validation['Variable']] = validation['Actual'].reset_index(drop=True)
    if len(pred_parts) == 0:
        return None, None
    n = len(next(iter(pred_parts.values())))
    pred_rows = []
    actual_rows = []
    for i in range(n):
        if strategy_name == 'direct_4_normalized':
            pred_values = {col: pred_parts[col].iloc[i] for col in COMPOSITION_COLS}
            actual_values = {col: actual_parts[col].iloc[i] for col in COMPOSITION_COLS}
            pred_row = reconstruct_direct_4(pred_values)
            actual_row = normalize_composition_dict(actual_values)
        elif strategy_name == 'direct_3_residual_nca':
            cols = ['NMC_share_%', 'LFP_share_%', 'LCO_share_%']
            pred_row = reconstruct_direct_3_residual_nca({col: pred_parts[col].iloc[i] for col in cols})
            actual_row = reconstruct_direct_3_residual_nca({col: actual_parts[col].iloc[i] for col in cols})
        else:
            pred_row = reconstruct_from_alr({col: pred_parts[col].iloc[i] for col in pred_parts.keys()}, reference_col)
            actual_row = reconstruct_from_alr({col: actual_parts[col].iloc[i] for col in actual_parts.keys()}, reference_col)
        pred_rows.append(pred_row)
        actual_rows.append(actual_row)
    pred_df = pd.DataFrame(pred_rows)
    actual_df = pd.DataFrame(actual_rows)
    metric_rows = []
    for col in COMPOSITION_COLS:
        rmse, mae, r2 = evaluate(actual_df[col], pred_df[col])
        metric_rows.append({
            'Strategy': strategy_name,
            'Reference': reference_col,
            'Horizon': horizon,
            'Variable': col,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        })
    metrics_df = pd.DataFrame(metric_rows)
    summary = {
        'Strategy': strategy_name,
        'Reference': reference_col,
        'Horizon': horizon,
        'Mean_MAE': metrics_df['MAE'].mean(),
        'Max_MAE': metrics_df['MAE'].max(),
        'Mean_RMSE': metrics_df['RMSE'].mean()
    }
    return metrics_df, summary


def trend_summary(forecast_df, variable):
    first_value = forecast_df[variable].iloc[0]
    last_value = forecast_df[variable].iloc[-1]
    diff = last_value - first_value
    pct = diff / first_value * 100 if first_value != 0 else np.nan
    if abs(pct) < 1:
        return 'stable', pct, 'remains practically stable over the forecast horizon.'
    if pct > 0:
        return 'increases', pct, f'increases by approximately {pct:.1f}% by the end of the forecast period.'
    return 'decreases', pct, f'decreases by approximately {abs(pct):.1f}% by the end of the forecast period.'


def reliability_from_metric(row):
    temporal_r2 = row['R2']
    if temporal_r2 >= 0.70:
        return 'High', 'The model performed well in the temporal backtest.'
    if temporal_r2 >= 0.30:
        return 'Medium', 'The model captured part of the pattern, but uncertainty remains.'
    return 'Low', 'The forecast should be interpreted with caution.'


def build_simple_reliability(metrics_df):
    temporal = metrics_df[metrics_df['Validation'] == 'temporal_backtest'].copy()
    rows = []
    for _, row in temporal.iterrows():
        variable = row['Variable']
        if variable not in ALL_MODELLED_VARIABLES:
            continue
        reliability, explanation = reliability_from_metric(row)
        rows.append({
            'Variable': variable,
            'Simple name': friendly_name(variable),
            'Reliability': reliability,
            'Temporal R2': row['R2'],
            'Mean MAE': row['MAE'],
            'Interpretation': explanation
        })
    return pd.DataFrame(rows)


def scenario_comparison_table(base_forecast_df, scenario_forecast_df):
    comparison_rows = []
    last_base = base_forecast_df.iloc[-1]
    last_scenario = scenario_forecast_df.iloc[-1]
    available_cols = [col for col in DISPLAY_VARIABLES if col in base_forecast_df.columns and col in scenario_forecast_df.columns]
    for col in available_cols:
        base_value = last_base[col]
        scenario_value = last_scenario[col]
        diff = scenario_value - base_value
        pct = diff / base_value * 100 if base_value != 0 else np.nan
        comparison_rows.append({
            'Variable': col,
            'Simple name': friendly_name(col),
            'Base final month': base_value,
            'Scenario final month': scenario_value,
            'Difference': diff,
            'Difference_%': pct
        })
    return pd.DataFrame(comparison_rows)


def explain_scenario_impact(comparison_df, variable):
    row = comparison_df[comparison_df['Variable'] == variable]
    if len(row) == 0:
        return 'There is not enough information to interpret this variable.'
    pct = row['Difference_%'].iloc[0]
    if pd.isna(pct):
        return 'It was not possible to calculate the percentage change.'
    if abs(pct) < 1:
        return f'In the scenario, {friendly_name(variable)} remains practically unchanged compared with the base forecast.'
    if pct > 0:
        return f'In the scenario, {friendly_name(variable)} increases by about {pct:.1f}% compared with the base forecast.'
    return f'In the scenario, {friendly_name(variable)} decreases by about {abs(pct):.1f}% compared with the base forecast.'


@st.cache_resource(show_spinner=False)
def train_pipeline(file_bytes, forecast_steps, backtest_horizon):
    df_raw = pd.read_excel(io.BytesIO(file_bytes))
    df = df_raw.copy().dropna(how='all').reset_index(drop=True)
    if ORIGINAL_DATE_COL in df.columns:
        sort_col = df[ORIGINAL_DATE_COL]
        if pd.api.types.is_datetime64_any_dtype(sort_col):
            df = df.sort_values(ORIGINAL_DATE_COL).reset_index(drop=True)
        else:
            numeric_sort = pd.to_numeric(sort_col, errors='coerce')
            if numeric_sort.notna().sum() > 0:
                df['_sort_key'] = numeric_sort
            else:
                df['_sort_key'] = pd.to_datetime(sort_col, errors='coerce')
            df = df.sort_values('_sort_key').drop(columns=['_sort_key']).reset_index(drop=True)
    for col in df.columns:
        if col != ORIGINAL_DATE_COL:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    missing_cols = [col for col in ALL_MODELLED_VARIABLES if col not in df.columns]
    if missing_cols:
        raise ValueError('The uploaded Excel is missing these required columns: ' + ', '.join(missing_cols))
    df = df.dropna(subset=ALL_MODELLED_VARIABLES).reset_index(drop=True)
    if len(df) < 30:
        raise ValueError('Not enough complete rows after cleaning. Check missing values in required columns.')
    df[DATE_COL] = np.arange(1, len(df) + 1)
    df = normalize_composition_df(df)
    df = add_energy_recovery_columns(df)
    for ref in COMPOSITION_COLS:
        df = add_alr_columns(df, ref)
    models = {}
    metrics = []
    validations = []
    for validation_type in ['random_structural', 'temporal_backtest']:
        model_pack, metric, validation = train_hybrid_model(df, 'Battery_mass_t', [], 'Feedstock', validation_type, backtest_horizon)
        metrics.append(metric)
        validations.append(validation)
        if validation_type == 'temporal_backtest':
            models['Battery_mass_t'] = model_pack
    for target in COMPOSITION_COLS:
        for validation_type in ['random_structural', 'temporal_backtest']:
            model_pack, metric, validation = train_hybrid_model(df, target, [], 'Feedstock_direct', validation_type, backtest_horizon)
            metrics.append(metric)
            validations.append(validation)
            if validation_type == 'temporal_backtest':
                models[target] = model_pack
    for ref in COMPOSITION_COLS:
        for col in COMPOSITION_COLS:
            if col == ref:
                continue
            target = alr_col(col, ref)
            for validation_type in ['random_structural', 'temporal_backtest']:
                model_pack, metric, validation = train_hybrid_model(df, target, [], f'Feedstock_ALR_ref_{ref}', validation_type, backtest_horizon)
                metrics.append(metric)
                validations.append(validation)
                if validation_type == 'temporal_backtest':
                    models[target] = model_pack
    composition_metric_tables = []
    composition_summaries = []
    strategies = [
        {'name': 'direct_4_normalized', 'reference': None},
        {'name': 'direct_3_residual_nca', 'reference': None}
    ]
    for ref in COMPOSITION_COLS:
        strategies.append({'name': f'alr_ref_{ref}', 'reference': ref})
    for strategy in strategies:
        metric_table, summary = evaluate_composition_strategy(validations, strategy['name'], backtest_horizon, strategy['reference'])
        if metric_table is not None:
            composition_metric_tables.append(metric_table)
            composition_summaries.append(summary)
    composition_metrics_df = pd.concat(composition_metric_tables, ignore_index=True)
    composition_strategy_summary_df = pd.DataFrame(composition_summaries)
    best_strategy_row = composition_strategy_summary_df.sort_values(['Mean_MAE', 'Max_MAE', 'Mean_RMSE']).iloc[0]
    stages = [
        {'stage_name': 'Pre-recycling outputs', 'targets': PRE_RECYCLING_OUTPUTS, 'exogenous_cols': FEEDSTOCK_INPUTS},
        {'stage_name': 'Process', 'targets': PROCESS_INPUTS, 'exogenous_cols': FEEDSTOCK_INPUTS},
        {'stage_name': 'Energy recovery', 'targets': ENERGY_RECOVERY_OUTPUTS, 'exogenous_cols': FEEDSTOCK_INPUTS + PROCESS_INPUTS},
        {'stage_name': 'Operational', 'targets': OPERATIONAL_INPUTS, 'exogenous_cols': FEEDSTOCK_INPUTS + PROCESS_INPUTS + ENERGY_RECOVERY_OUTPUTS},
        {'stage_name': 'Final', 'targets': FINAL_TARGETS, 'exogenous_cols': FEEDSTOCK_INPUTS + PROCESS_INPUTS + ENERGY_RECOVERY_OUTPUTS + OPERATIONAL_INPUTS}
    ]
    for stage in stages:
        for target in stage['targets']:
            for validation_type in ['random_structural', 'temporal_backtest']:
                model_pack, metric, validation = train_hybrid_model(df, target, stage['exogenous_cols'], stage['stage_name'], validation_type, backtest_horizon)
                metrics.append(metric)
                validations.append(validation)
                if validation_type == 'temporal_backtest':
                    models[target] = model_pack
    metrics_df = pd.DataFrame(metrics)
    simple_reliability_df = build_simple_reliability(metrics_df)
    return {
        'df': df,
        'models': models,
        'metrics_df': metrics_df,
        'validations': validations,
        'composition_metrics_df': composition_metrics_df,
        'composition_strategy_summary_df': composition_strategy_summary_df,
        'simple_reliability_df': simple_reliability_df,
        'best_strategy': best_strategy_row['Strategy'],
        'best_reference': best_strategy_row['Reference'],
        'forecast_steps': forecast_steps,
        'backtest_horizon': backtest_horizon
    }


def predict_future_composition(row, history_df, models, strategy, reference):
    if strategy == 'direct_4_normalized':
        pred_values = {}
        for target in COMPOSITION_COLS:
            feature_row = row.copy()
            feature_row.update(make_future_lag_features(history_df, target))
            pred_values[target] = predict_one_step_hybrid(models[target], feature_row, target)
        return reconstruct_direct_4(pred_values)
    if strategy == 'direct_3_residual_nca':
        pred_values = {}
        for target in ['NMC_share_%', 'LFP_share_%', 'LCO_share_%']:
            feature_row = row.copy()
            feature_row.update(make_future_lag_features(history_df, target))
            pred_values[target] = predict_one_step_hybrid(models[target], feature_row, target)
        return reconstruct_direct_3_residual_nca(pred_values)
    if strategy.startswith('alr_ref_'):
        ref = reference
        pred_logratios = {}
        for col in COMPOSITION_COLS:
            if col == ref:
                continue
            target = alr_col(col, ref)
            feature_row = row.copy()
            feature_row.update(make_future_lag_features(history_df, target))
            pred_logratios[target] = predict_one_step_hybrid(models[target], feature_row, target)
        return reconstruct_from_alr(pred_logratios, ref)
    raise ValueError('Invalid composition strategy')


def forecast_engine(result, scenario_config=None):
    df = result['df']
    models = result['models']
    strategy = result['best_strategy']
    reference = result['best_reference']
    forecast_steps = result['forecast_steps']
    history_df = df.copy()
    future_rows = []
    last_month = int(history_df[DATE_COL].iloc[-1])
    for step in range(1, forecast_steps + 1):
        future_month = last_month + step
        t = step / forecast_steps
        scenario_mode = scenario_config.get('application_mode', 'gradual') if scenario_config is not None else 'gradual'
        scenario_factor = 1.0 if scenario_mode == 'direct' else t
        row = {
            DATE_COL: future_month,
            'time_trend': future_month,
            'month_sin': np.sin(2 * np.pi * future_month / 12),
            'month_cos': np.cos(2 * np.pi * future_month / 12)
        }
        feature_row = row.copy()
        feature_row.update(make_future_lag_features(history_df, 'Battery_mass_t'))
        row['Battery_mass_t'] = predict_one_step_hybrid(models['Battery_mass_t'], feature_row, 'Battery_mass_t')
        row.update(predict_future_composition(row, history_df, models, strategy, reference))
        if scenario_config is not None:
            row['Battery_mass_t'] *= 1 + scenario_config['battery_mass_pct'] / 100 * scenario_factor
            if scenario_mode == 'direct' and 'target_mix' in scenario_config:
                for col in COMPOSITION_COLS:
                    row[col] = scenario_config['target_mix'][col]
            else:
                for col in COMPOSITION_COLS:
                    row[col] = row[col] + scenario_config[f'{col}_pp'] * scenario_factor
            row = normalize_composition_dict(row)
        for ref in COMPOSITION_COLS:
            for col in COMPOSITION_COLS:
                if col != ref:
                    row[alr_col(col, ref)] = np.log(max(row[col], EPS) / max(row[ref], EPS))
        for target in PRE_RECYCLING_OUTPUTS:
            feature_row = row.copy()
            feature_row.update(make_future_lag_features(history_df, target))
            row[target] = predict_one_step_hybrid(models[target], feature_row, target)
        for target in PROCESS_INPUTS:
            feature_row = row.copy()
            feature_row.update(make_future_lag_features(history_df, target))
            row[target] = predict_one_step_hybrid(models[target], feature_row, target)
        if scenario_config is not None:
            for target in PROCESS_INPUTS:
                if scenario_mode == 'direct' and f'{target}_value' in scenario_config:
                    row[target] = scenario_config[f'{target}_value']
                else:
                    row[target] = row[target] + scenario_config.get(f'{target}_delta', 0.0) * scenario_factor
                row[target] = clip_logical_bounds(target, row[target])
        for target in ENERGY_RECOVERY_OUTPUTS:
            feature_row = row.copy()
            feature_row.update(make_future_lag_features(history_df, target))
            row[target] = predict_one_step_hybrid(models[target], feature_row, target)
        if scenario_config is not None:
            for target in ENERGY_RECOVERY_OUTPUTS:
                if scenario_mode == 'direct' and f'{target}_value' in scenario_config:
                    row[target] = scenario_config[f'{target}_value']
                else:
                    row[target] = row[target] + scenario_config.get(f'{target}_delta', 0.0) * scenario_factor
                row[target] = clip_logical_bounds(target, row[target])
        for target in OPERATIONAL_INPUTS:
            feature_row = row.copy()
            feature_row.update(make_future_lag_features(history_df, target))
            row[target] = predict_one_step_hybrid(models[target], feature_row, target)
        if scenario_config is not None:
            for target in OPERATIONAL_INPUTS:
                row[target] = row[target] * (1 + scenario_config.get(f'{target}_pct', 0.0) / 100 * scenario_factor)
                row[target] = clip_logical_bounds(target, row[target])
        for target in FINAL_TARGETS:
            feature_row = row.copy()
            feature_row.update(make_future_lag_features(history_df, target))
            row[target] = predict_one_step_hybrid(models[target], feature_row, target)
        future_rows.append(row)
        history_df = pd.concat([history_df, pd.DataFrame([row])], ignore_index=True)
    future_df = pd.DataFrame(future_rows)
    future_df = add_energy_recovery_columns(future_df)
    return future_df


def plot_history_forecast(df, base_df, scenario_df, variable):
    fig = go.Figure()
    if variable in df.columns:
        fig.add_trace(go.Scatter(x=df[DATE_COL], y=df[variable], mode='lines', name='Historical'))
    fig.add_trace(go.Scatter(x=base_df[DATE_COL], y=base_df[variable], mode='lines+markers', name='Base forecast'))
    if scenario_df is not None:
        fig.add_trace(go.Scatter(x=scenario_df[DATE_COL], y=scenario_df[variable], mode='lines+markers', name='Scenario'))
    fig.update_layout(height=460, xaxis_title='Month', yaxis_title=friendly_name(variable), template='plotly_white')
    return fig


def to_excel_bytes(sheets):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, dataframe in sheets.items():
            dataframe.to_excel(writer, sheet_name=name[:31], index=False)
    return output.getvalue()


def energy_breakdown(total_electricity_kwh_t, renewable_share_pct, recovery_rate_pct):
    recovery_fraction = np.clip(recovery_rate_pct, 0, 100) / 100
    renewable_fraction = np.clip(renewable_share_pct, 0, 100) / 100
    recovered_energy = total_electricity_kwh_t * recovery_fraction
    external_electricity = max(0.0, total_electricity_kwh_t - recovered_energy)
    renewable_external = external_electricity * renewable_fraction
    nonrenewable_external = external_electricity * (1 - renewable_fraction)
    return {
        'Recovered_energy_kWh_t': recovered_energy,
        'External_electricity_kWh_t': external_electricity,
        'Renewable_external_kWh_t': renewable_external,
        'Nonrenewable_external_kWh_t': nonrenewable_external
    }


def energy_from_model_row(row):
    return {
        'Recovered_energy_kWh_t': float(row.get('Recovered_energy_kWh_t', 0.0)),
        'External_electricity_kWh_t': float(row.get('External_electricity_kWh_t', 0.0)),
        'Renewable_external_kWh_t': float(row.get('Renewable_external_kWh_t', 0.0)),
        'Nonrenewable_external_kWh_t': float(row.get('Nonrenewable_external_kWh_t', 0.0))
    }


def scale_energy(energy, factor):
    return {key: value * factor for key, value in energy.items()}


def show_energy_and_gwp_metrics(quantity_label, quantity, gwp_label, gwp_value, energy, total_gwp):
    render_result_cards([
        {'label': quantity_label, 'value': format_compact(quantity)},
        {'label': gwp_label, 'value': format_compact(gwp_value, 'kg CO2e')},
        {'label': 'Total GWP', 'value': format_compact(total_gwp, 'kg CO2e')},
        {'label': 'Recovered energy', 'value': format_compact(energy['Recovered_energy_kWh_t'], 'kWh')},
        {'label': 'External electricity', 'value': format_compact(energy['External_electricity_kWh_t'], 'kWh')}
    ])


def show_normalized_scenario_metrics(quantity_label, quantity, gwp_label, gwp_value, energy):
    html = f"""
    <div class="scenario-kpi-board">
        <div class="scenario-kpi-card scenario-kpi-card-soft">
            <div class="scenario-kpi-label">{quantity_label}</div>
            <div class="scenario-kpi-value">{format_compact(quantity)}</div>
            <div class="scenario-kpi-note">normalized basis</div>
        </div>
        <div class="scenario-kpi-card scenario-kpi-card-impact">
            <div class="scenario-kpi-label">{gwp_label}</div>
            <div class="scenario-kpi-value">{format_compact(gwp_value, 'kg CO2e')}</div>
            <div class="scenario-kpi-note">model forecast</div>
        </div>
        <div class="scenario-kpi-card scenario-kpi-card-recovered">
            <div class="scenario-kpi-label">Recovered energy</div>
            <div class="scenario-kpi-value">{format_compact(energy['Recovered_energy_kWh_t'], 'kWh')}</div>
            <div class="scenario-kpi-note">used internally</div>
        </div>
        <div class="scenario-grid-energy-card">
            <div class="scenario-grid-energy-main">
                <div class="scenario-kpi-label">Grid Energy Consumption</div>
                <div class="scenario-kpi-value">{format_compact(energy['External_electricity_kWh_t'], 'kWh')}</div>
                <div class="scenario-kpi-note">external electricity after recovery</div>
            </div>
            <div class="scenario-grid-energy-split">
                <div class="scenario-energy-pill scenario-energy-pill-renewable">
                    <div class="scenario-energy-pill-label">Renewable</div>
                    <div class="scenario-energy-pill-value">{format_compact(energy['Renewable_external_kWh_t'], 'kWh')}</div>
                </div>
                <div class="scenario-energy-pill scenario-energy-pill-nonrenewable">
                    <div class="scenario-energy-pill-label">Non-renewable</div>
                    <div class="scenario-energy-pill-value">{format_compact(energy['Nonrenewable_external_kWh_t'], 'kWh')}</div>
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def show_small_metric_cards(items):
    render_result_cards([
        {'label': label, 'value': value}
        for label, value in items
    ])


def show_energy_outcome_panel(title, energy):
    st.markdown(f'### {title}')
    st.markdown(
        '<div class="section-note">Model-based electricity balance for the selected scenario.</div>',
        unsafe_allow_html=True
    )
    render_result_cards([
        {'label': 'Recovered battery energy', 'value': format_compact(energy['Recovered_energy_kWh_t'], 'kWh')},
        {'label': 'External electricity required', 'value': format_compact(energy['External_electricity_kWh_t'], 'kWh')},
        {'label': 'Renewable external electricity', 'value': format_compact(energy['Renewable_external_kWh_t'], 'kWh')},
        {'label': 'Non-renewable external electricity', 'value': format_compact(energy['Nonrenewable_external_kWh_t'], 'kWh')}
    ])


def show_gwp_outcome_panel(title, gwp_label, gwp_value, total_gwp):
    st.markdown(f'### {title}')
    st.markdown(
        '<div class="section-note">Model-based climate impact for the selected scenario.</div>',
        unsafe_allow_html=True
    )
    render_result_cards([
        {'label': gwp_label, 'value': format_compact(gwp_value, 'kg CO2e')},
        {'label': 'Total GWP for selected quantity', 'value': format_compact(total_gwp, 'kg CO2e')}
    ])


def scenario_energy_chart(energy, title):
    labels = [
        'Recovered battery energy',
        'Renewable external electricity',
        'Non-renewable external electricity'
    ]
    values = [
        energy['Recovered_energy_kWh_t'],
        energy['Renewable_external_kWh_t'],
        energy['Nonrenewable_external_kWh_t']
    ]
    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=labels,
                orientation='h',
                marker_color=['#2ca25f', '#4c78a8', '#e45756'],
                text=[f'{value:,.1f} kWh' for value in values],
                textposition='auto'
            )
        ]
    )
    fig.update_layout(
        title=title,
        height=330,
        template='plotly_white',
        margin=dict(l=10, r=10, t=55, b=20),
        xaxis_title='kWh',
        yaxis_title=None
    )
    return fig


def scenario_gwp_chart(gwp_per_unit, total_gwp, unit_label):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[f'Per {unit_label}', 'Total selected quantity'],
            y=[gwp_per_unit, total_gwp],
            marker_color=['#f58518', '#b279a2'],
            text=[
                f'{gwp_per_unit:,.1f} kg CO2e/{unit_label}',
                f'{total_gwp:,.1f} kg CO2e'
            ],
            textposition='auto'
        )
    )
    fig.update_layout(
        title='GWP result',
        height=330,
        template='plotly_white',
        margin=dict(l=10, r=10, t=55, b=20),
        yaxis_title='kg CO2e'
    )
    return fig


def scenario_inputs_panel(title, body):
    st.markdown(f"#### {title}")
    st.caption(body)


def scenario_forecast_charts(title, scenario_df, variables, default_variables, key_prefix):
    available = [
        col for col in variables
        if col in scenario_df.columns and (col in base_forecast_df.columns or col in df.columns)
    ]
    if not available:
        return

    defaults = [col for col in default_variables if col in available]
    if not defaults:
        defaults = available[:2]

    with st.expander(title):
        st.caption(
            'These charts compare the base forecast with the forecast recalculated using the selected scenario inputs.'
        )
        selected = st.multiselect(
            'Variables to plot',
            available,
            default=defaults,
            format_func=friendly_name,
            key=f'{key_prefix}_forecast_vars'
        )
        for variable in selected:
            st.plotly_chart(
                plot_history_forecast(df, base_forecast_df, scenario_df, variable),
                use_container_width=True,
                key=f'{key_prefix}_forecast_{variable}'
            )


def battery_mix_controls(base_row, key_prefix):
    st.markdown('##### Battery chemistry mix')
    st.caption('Set the future battery chemistry shares. Values are normalized to sum to 100%.')

    c1, c2 = st.columns(2)
    with c1:
        nmc = st.slider(
            'NMC share (%)',
            0.0,
            100.0,
            float(base_row.get('NMC_share_%', 25.0)),
            0.5,
            key=f'{key_prefix}_nmc_share'
        )
        lfp = st.slider(
            'LFP share (%)',
            0.0,
            100.0,
            float(base_row.get('LFP_share_%', 25.0)),
            0.5,
            key=f'{key_prefix}_lfp_share'
        )
    with c2:
        lco = st.slider(
            'LCO share (%)',
            0.0,
            100.0,
            float(base_row.get('LCO_share_%', 25.0)),
            0.5,
            key=f'{key_prefix}_lco_share'
        )
        nca = st.slider(
            'NCA share (%)',
            0.0,
            100.0,
            float(base_row.get('NCA_share_%', 25.0)),
            0.5,
            key=f'{key_prefix}_nca_share'
        )

    target_mix = normalize_composition_dict({
        'NMC_share_%': nmc,
        'LFP_share_%': lfp,
        'LCO_share_%': lco,
        'NCA_share_%': nca
    })
    st.caption(
        'Normalized mix used by the model: '
        f"NMC {target_mix['NMC_share_%']:.1f}% | "
        f"LFP {target_mix['LFP_share_%']:.1f}% | "
        f"LCO {target_mix['LCO_share_%']:.1f}% | "
        f"NCA {target_mix['NCA_share_%']:.1f}%"
    )
    return target_mix


def scenario_config_from_mix_and_energy(base_row, target_mix, renewable_share, recovery_rate):
    return {
        'application_mode': 'direct',
        'target_mix': target_mix,
        'battery_mass_pct': 0.0,
        'NMC_share_%_pp': target_mix['NMC_share_%'] - float(base_row.get('NMC_share_%', 0.0)),
        'LFP_share_%_pp': target_mix['LFP_share_%'] - float(base_row.get('LFP_share_%', 0.0)),
        'LCO_share_%_pp': target_mix['LCO_share_%'] - float(base_row.get('LCO_share_%', 0.0)),
        'NCA_share_%_pp': target_mix['NCA_share_%'] - float(base_row.get('NCA_share_%', 0.0)),
        'Contamination_%_delta': 0.0,
        'State_of_charge_%_delta': 0.0,
        'Feed_rate_kg_h_delta': 0.0,
        'Sorting_eff_%_delta': 0.0,
        'Renewable_share_%_value': renewable_share,
        'Renewable_share_%_delta': renewable_share - float(base_row.get('Renewable_share_%', 0.0)),
        'Overall_recovery_rate_%_value': recovery_rate,
        'Overall_recovery_rate_%_delta': recovery_rate - float(base_row.get('Overall_recovery_rate_%', 0.0)),
        'Electricity_total_kWh_t_pct': 0.0,
        'Diesel_MJ_t_pct': 0.0,
        'Water_L_t_pct': 0.0,
        'Transport_km_pct': 0.0
    }


def format_compact(value, unit=''):
    if pd.isna(value):
        return 'n/a'
    abs_value = abs(value)
    if abs_value >= 1000:
        text = f'{value:,.0f}'
    elif abs_value >= 100:
        text = f'{value:,.1f}'
    else:
        text = f'{value:,.2f}'
    return f'{text} {unit}'.strip()


def render_result_cards(items):
    cards = []
    for item in items:
        cards.append(
            '<div class="result-card">'
            f'<div class="result-label">{item["label"]}</div>'
            f'<div class="result-value">{item["value"]}</div>'
            '</div>'
        )
    st.markdown(
        '<div class="result-grid">' + ''.join(cards) + '</div>',
        unsafe_allow_html=True
    )


st.set_page_config(page_title='Battery Recycling LCA Forecast Dashboard', layout='wide')
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .scenario-intro {
        color: #4b5563;
        font-size: 0.98rem;
        line-height: 1.55;
        margin-bottom: 1rem;
    }
    .result-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 0.75rem 0 0.75rem 0;
    }
    .result-card {
        border: 1px solid #dbe4ef;
        border-radius: 8px;
        padding: 0.75rem 0.8rem;
        background: #ffffff;
        min-height: 82px;
    }
    .result-label {
        color: #64748b;
        font-size: 0.78rem;
        line-height: 1.25;
        margin-bottom: 0.45rem;
    }
    .result-value {
        color: #0f172a;
        font-size: clamp(0.95rem, 1.25vw, 1.25rem);
        font-weight: 650;
        line-height: 1.15;
        overflow-wrap: anywhere;
    }
    .scenario-kpi-board {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.9rem;
        margin: 0.85rem 0 0.9rem 0;
    }
    .scenario-kpi-card,
    .scenario-grid-energy-card {
        border: 1px solid #d8e2ee;
        border-radius: 8px;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .scenario-kpi-card {
        min-height: 96px;
        padding: 0.9rem 1rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border-left: 4px solid #94a3b8;
    }
    .scenario-kpi-card-impact {
        background: linear-gradient(180deg, #fff7ed 0%, #ffffff 78%);
        border-left-color: #f97316;
    }
    .scenario-kpi-card-recovered {
        background: linear-gradient(180deg, #f0fdf4 0%, #ffffff 78%);
        border-left-color: #22c55e;
    }
    .scenario-kpi-card-soft {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 78%);
        border-left-color: #64748b;
    }
    .scenario-kpi-label {
        color: #52637a;
        font-size: 0.78rem;
        line-height: 1.25;
        margin-bottom: 0.35rem;
    }
    .scenario-kpi-value {
        color: #0f172a;
        font-size: clamp(1.05rem, 1.45vw, 1.45rem);
        font-weight: 750;
        line-height: 1.12;
        overflow-wrap: anywhere;
    }
    .scenario-kpi-note {
        color: #7b8798;
        font-size: 0.72rem;
        margin-top: 0.45rem;
        line-height: 1.2;
    }
    .scenario-grid-energy-card {
        grid-column: span 2;
        display: grid;
        grid-template-columns: minmax(0, 1.15fr) minmax(220px, 0.85fr);
        gap: 0.9rem;
        padding: 0.9rem;
        background: linear-gradient(180deg, #eff6ff 0%, #ffffff 76%);
    }
    .scenario-grid-energy-main {
        border-left: 4px solid #2563eb;
        padding: 0.15rem 0.2rem 0.15rem 0.9rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 104px;
    }
    .scenario-grid-energy-split {
        display: grid;
        grid-template-rows: repeat(2, minmax(0, 1fr));
        gap: 0.65rem;
    }
    .scenario-energy-pill {
        border: 1px solid #d8e2ee;
        border-radius: 8px;
        background: #ffffff;
        padding: 0.62rem 0.75rem;
        min-height: 50px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .scenario-energy-pill-renewable {
        border-left: 4px solid #22c55e;
    }
    .scenario-energy-pill-nonrenewable {
        border-left: 4px solid #ef4444;
    }
    .scenario-energy-pill-label {
        color: #52637a;
        font-size: 0.72rem;
        line-height: 1.2;
        margin-bottom: 0.2rem;
    }
    .scenario-energy-pill-value {
        color: #0f172a;
        font-size: clamp(0.86rem, 1vw, 1rem);
        font-weight: 720;
        line-height: 1.1;
        overflow-wrap: anywhere;
    }
    @media (max-width: 760px) {
        .scenario-kpi-board {
            grid-template-columns: 1fr;
        }
        .scenario-grid-energy-card {
            grid-column: span 1;
            grid-template-columns: 1fr;
        }
    }
    .section-note {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: -0.35rem;
        margin-bottom: 0.75rem;
    }
    .scenario-shell {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        gap: 1rem;
        margin-top: 1rem;
    }
    .scenario-panel {
        border: 1px solid #dbe4ef;
        border-radius: 8px;
        padding: 1rem 1.1rem 1.15rem 1.1rem;
        background: #ffffff;
        min-height: 260px;
    }
    .scenario-panel h4 {
        margin-top: 0;
        margin-bottom: 0.25rem;
    }
    .scenario-summary {
        border: 1px solid #dbe4ef;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        background: #f8fafc;
        margin: 1rem 0;
    }
    .scenario-summary-title {
        color: #0f172a;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .scenario-summary-text {
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.45;
    }
    .scenario-visual-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        gap: 1rem;
        margin-top: 1rem;
    }
    .scenario-visual-panel {
        border: 1px solid #dbe4ef;
        border-radius: 8px;
        padding: 0.7rem 0.8rem;
        background: #ffffff;
        min-height: 360px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.45rem;
    }
    @media (max-width: 900px) {
        .result-grid {
            grid-template-columns: 1fr;
        }
        .scenario-shell,
        .scenario-visual-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title('Battery Recycling LCA Forecast & Scenario Dashboard')
st.caption('Dashboard with a simple decision view and a technical model-audit view.')

with st.sidebar:
    st.header('Data and horizon')
    uploaded_file = st.file_uploader('Upload Excel file', type=['xlsx', 'xls'])
    forecast_steps = st.slider('Forecast months', min_value=3, max_value=60, value=12, step=3)
    backtest_horizon = st.slider('Temporal backtest months', min_value=6, max_value=36, value=12, step=6)
    train_button = st.button('Train / Refresh', type='primary')
    st.markdown('---')
    st.caption('Tip: the longer the forecast horizon, the greater the uncertainty tends to be.')

if uploaded_file is None:
    st.info('Upload an Excel file to get started.')
    st.stop()

file_bytes = uploaded_file.getvalue()

if train_button or 'result' not in st.session_state:
    try:
        with st.spinner('Training models and generating base forecast...'):
            st.session_state['result'] = train_pipeline(file_bytes, forecast_steps, backtest_horizon)
            st.session_state['base_forecast'] = forecast_engine(st.session_state['result'])
    except Exception as exc:
        st.error(str(exc))
        st.stop()

result = st.session_state['result']
df = result['df']
base_forecast_df = st.session_state['base_forecast']
battery_mix_method = friendly_composition_strategy_details(result['best_strategy'], result['best_reference'])

st.success(
    f"Model trained. Best strategy: {battery_mix_method['Best strategy']} | "
    f"Reference battery type: {battery_mix_method['Reference battery type']}"
)

tab_supplier, tab_black_mass, tab_forecast, tab_scenario, tab_explain, tab_logic, tab_export = st.tabs([
    'Scenario 1 - Recycling Process',
    'Scenario 2 - Material Receiver',
    'Base Forecast',
    'Scenario Simulator',
    'How to Interpret',
    'Model Logic',
    'Export'
])

with tab_forecast:
    st.subheader('Base Forecast')
    st.markdown(
        """
        The base forecast is the model prediction assuming that historical patterns continue.
        This section summarizes the main expected trends before exploring individual variables.
        """
    )
    main_vars = [
        'Black_mass_kg_t',
        'Overall_recovery_rate_%',
        'Recovered_energy_kWh_t',
        'GWP_service_kgCO2e_tBattery'
    ]
    cols = st.columns(4)
    for i, var in enumerate(main_vars):
        if var not in base_forecast_df.columns:
            continue
        _, pct, _ = trend_summary(base_forecast_df, var)
        with cols[i]:
            st.metric(
                label=friendly_name(var),
                value=f"{base_forecast_df[var].iloc[-1]:.3f}",
                delta=f"{pct:.1f}%" if not pd.isna(pct) else None
            )
    with st.expander('Automatic interpretation of the base forecast'):
        for var in main_vars:
            if var in base_forecast_df.columns:
                _, _, sentence = trend_summary(base_forecast_df, var)
                st.write(f"**{friendly_name(var)}:** {sentence}")

        simple_reliability_df = result['simple_reliability_df']
        default_reliability_vars = [var for var in main_vars if var in ALL_MODELLED_VARIABLES]
        reliability_view = simple_reliability_df[
            simple_reliability_df['Variable'].isin(default_reliability_vars)
        ][['Simple name', 'Reliability', 'Mean MAE', 'Interpretation']]
        st.markdown('#### Reliability in plain language')
        st.dataframe(reliability_view, use_container_width=True)

    st.markdown('---')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Historical samples', len(df))
    c2.metric('Forecast months', result['forecast_steps'])
    c3.metric('Temporal backtest', result['backtest_horizon'])
    with c4:
        st.markdown('**Battery mix method**')
        st.markdown(
            f"**Best strategy:** {battery_mix_method['Best strategy']}  \n"
            f"**Reference battery type:** {battery_mix_method['Reference battery type']}"
        )
    available_display = [col for col in DISPLAY_VARIABLES if col in base_forecast_df.columns or col in df.columns]
    default_index = available_display.index('GWP_service_kgCO2e_tBattery') if 'GWP_service_kgCO2e_tBattery' in available_display else 0
    variable = st.selectbox('Choose a variable', available_display, format_func=friendly_name, index=default_index, key='forecast_variable')
    st.plotly_chart(plot_history_forecast(df, base_forecast_df, None, variable), use_container_width=True, key=f"forecast_base_{variable}")
    _, _, sentence = trend_summary(base_forecast_df, variable)
    st.info(f"Interpretation: {friendly_name(variable)} {sentence}")
    with st.expander('View full base forecast table'):
        st.dataframe(base_forecast_df, use_container_width=True)

with tab_supplier:
    st.subheader('Scenario 1 - Recycling Process')
    st.markdown(
        """
        <div class="scenario-intro">
        Recycling process view: estimate the energy needs and climate impact to treat batteries.
        Results are normalized per **1 tonne of batteries**.
        </div>
        """
        .replace('**1 tonne of batteries**', '<strong>1 tonne of batteries</strong>'),
        unsafe_allow_html=True
    )

    base_row = base_forecast_df.iloc[-1]

    top_left, top_right = st.columns(2)
    with top_left:
        with st.container(border=True):
            scenario_inputs_panel('Input and model assumptions', 'The base values come from the predictive model. Scenario controls change selected inputs and the model recalculates the outputs.')
            supplier_battery_t = st.number_input(
                'Battery quantity to treat (tonnes)',
                min_value=0.01,
                value=1.00,
                step=0.10,
                key='supplier_battery_t'
            )
            supplier_electricity = float(base_row.get('Electricity_total_kWh_t', 0.0))
            st.info(f"Base forecast electricity demand: {supplier_electricity:,.2f} kWh/t battery")
            supplier_target_mix = battery_mix_controls(base_row, 'supplier')
    with top_right:
        with st.container(border=True):
            scenario_inputs_panel('Energy scenario controls', 'Use these controls to test renewable electricity and recovered battery energy.')
            supplier_renewable = st.slider(
                'Renewable share of external electricity (%)',
                0.0,
                100.0,
                float(base_row.get('Renewable_share_%', 0.0)),
                1.0,
                key='supplier_renewable'
            )
            supplier_recovery = st.slider(
                'Battery energy recovery rate (%)',
                0.0,
                100.0,
                float(base_row.get('Overall_recovery_rate_%', 0.0)),
                1.0,
                key='supplier_recovery'
            )
            st.caption(
                f"Non-renewable external electricity is automatically treated as "
                f"{100.0 - supplier_renewable:.1f}%."
            )

    supplier_model_config = scenario_config_from_mix_and_energy(
        base_row,
        supplier_target_mix,
        supplier_renewable,
        supplier_recovery
    )
    supplier_model_forecast = forecast_engine(result, supplier_model_config)
    supplier_model_row = supplier_model_forecast.iloc[-1]
    supplier_electricity = float(supplier_model_row.get('Electricity_total_kWh_t', supplier_electricity))
    gwp_per_t_battery = float(supplier_model_row.get('GWP_service_kgCO2e_tBattery', 0.0))
    supplier_energy_per_t = energy_from_model_row(supplier_model_row)
    supplier_total_energy = scale_energy(supplier_energy_per_t, supplier_battery_t)
    supplier_total_gwp = gwp_per_t_battery * supplier_battery_t

    st.caption(
        f"Scenario values below come directly from the predictive model after applying the selected "
        f"battery mix, renewable share, and energy recovery assumptions. Model electricity demand: "
        f"{supplier_electricity:,.2f} kWh/t battery."
    )

    with st.container(border=True):
        st.markdown('### Per 1 tonne of batteries')
        st.markdown('<div class="section-note">Treatment intensity on a fixed 1 tonne battery basis.</div>', unsafe_allow_html=True)
        show_normalized_scenario_metrics(
            'Battery basis',
            1.0,
            'GWP per tonne battery',
            gwp_per_t_battery,
            supplier_energy_per_t
        )

    supplier_material_cols = [
        col for col in PRE_RECYCLING_OUTPUTS + ['Black_mass_kg_t']
        if col in supplier_model_row.index
    ]
    if supplier_material_cols:
        supplier_material_rows = [
            {
                'Material / output': friendly_name(col),
                'Model result per tonne battery (kg/t battery)': float(supplier_model_row[col]),
                'Share of 1 tonne battery (%)': float(supplier_model_row[col]) / 1000 * 100
            }
            for col in supplier_material_cols
        ]
        material_sum = sum(row['Model result per tonne battery (kg/t battery)'] for row in supplier_material_rows)
        electrolyte_kg = 1000.0 - material_sum
        supplier_material_rows.append({
            'Material / output': 'Electrolyte',
            'Model result per tonne battery (kg/t battery)': electrolyte_kg,
            'Share of 1 tonne battery (%)': electrolyte_kg / 1000 * 100
        })
        supplier_material_table = pd.DataFrame(supplier_material_rows)
        with st.expander('Expected material outputs'):
            st.dataframe(supplier_material_table, use_container_width=True)

    scenario_forecast_charts(
        'Forecast charts for supplier scenario',
        supplier_model_forecast,
        [
            'Battery_mass_t',
            'NMC_share_%',
            'LFP_share_%',
            'LCO_share_%',
            'NCA_share_%',
            'Ferrous_kg_t',
            'Aluminium_kg_t',
            'Copper_kg_t',
            'Mixed_plastics_kg_t',
            'Other_byproducts_kg_t',
            'Black_mass_kg_t',
            'Electricity_total_kWh_t',
            'Overall_recovery_rate_%',
            'Recovered_energy_kWh_t',
            'External_electricity_kWh_t',
            'GWP_service_kgCO2e_tBattery'
        ],
        [
            'Electricity_total_kWh_t',
            'Recovered_energy_kWh_t',
            'External_electricity_kWh_t',
            'GWP_service_kgCO2e_tBattery',
            'Black_mass_kg_t'
        ],
        'supplier'
    )

    supplier_table = pd.DataFrame([
        {'Indicator': 'Battery quantity treated', 'Value': supplier_battery_t, 'Unit': 't battery'},
        {'Indicator': 'Model GWP per tonne battery', 'Value': gwp_per_t_battery, 'Unit': 'kg CO2e/t battery'},
        {'Indicator': 'Total model GWP', 'Value': supplier_total_gwp, 'Unit': 'kg CO2e'},
        {'Indicator': 'Total electricity demand', 'Value': supplier_electricity * supplier_battery_t, 'Unit': 'kWh'},
        {'Indicator': 'Recovered battery energy', 'Value': supplier_total_energy['Recovered_energy_kWh_t'], 'Unit': 'kWh'},
        {'Indicator': 'Grid Energy Consumption', 'Value': supplier_total_energy['External_electricity_kWh_t'], 'Unit': 'kWh'},
        {'Indicator': 'Renewable external electricity', 'Value': supplier_total_energy['Renewable_external_kWh_t'], 'Unit': 'kWh'},
        {'Indicator': 'Non-renewable external electricity', 'Value': supplier_total_energy['Nonrenewable_external_kWh_t'], 'Unit': 'kWh'}
    ])
    with st.expander('Detailed supplier scenario table'):
        st.dataframe(supplier_table, use_container_width=True)

with tab_black_mass:
    st.subheader('Scenario 2 - Material Receiver')
    st.markdown(
        """
        <div class="scenario-intro">
        Material receiver view: estimate the battery input, energy needs, chemistry, and climate impact
        associated with obtaining **1 tonne of black mass** or another selected black mass quantity.
        </div>
        """
        .replace('**1 tonne of black mass**', '<strong>1 tonne of black mass</strong>'),
        unsafe_allow_html=True
    )

    base_row = base_forecast_df.iloc[-1]
    black_mass_kg_per_t_battery = max(float(base_row.get('Black_mass_kg_t', 0.0)), 1e-9)

    top_left, top_right = st.columns(2)
    with top_left:
        with st.container(border=True):
            scenario_inputs_panel('Target and model assumptions', 'The base values come from the predictive model. Scenario controls change selected inputs and the model recalculates the outputs.')
            target_black_mass_t = st.number_input(
                'Target black mass quantity (tonnes)',
                min_value=0.01,
                value=1.00,
                step=0.10,
                key='target_black_mass_t'
            )
            receiver_electricity = float(base_row.get('Electricity_total_kWh_t', 0.0))
            st.info(f"Base forecast electricity demand: {receiver_electricity:,.2f} kWh/t battery")
            receiver_target_mix = battery_mix_controls(base_row, 'receiver')
    with top_right:
        with st.container(border=True):
            scenario_inputs_panel('Energy scenario controls', 'Use these controls to test renewable electricity and recovered battery energy.')
            receiver_renewable = st.slider(
                'Renewable share of external electricity (%)',
                0.0,
                100.0,
                float(base_row.get('Renewable_share_%', 0.0)),
                1.0,
                key='receiver_renewable'
            )
            receiver_recovery = st.slider(
                'Battery energy recovery rate (%)',
                0.0,
                100.0,
                float(base_row.get('Overall_recovery_rate_%', 0.0)),
                1.0,
                key='receiver_recovery'
            )
            st.caption(
                f"Non-renewable external electricity is automatically treated as "
                f"{100.0 - receiver_renewable:.1f}%."
            )

    receiver_model_config = scenario_config_from_mix_and_energy(
        base_row,
        receiver_target_mix,
        receiver_renewable,
        receiver_recovery
    )
    receiver_model_forecast = forecast_engine(result, receiver_model_config)
    receiver_model_row = receiver_model_forecast.iloc[-1]
    receiver_electricity = float(receiver_model_row.get('Electricity_total_kWh_t', receiver_electricity))
    black_mass_kg_per_t_battery = max(float(receiver_model_row.get('Black_mass_kg_t', black_mass_kg_per_t_battery)), 1e-9)

    battery_t_per_t_black_mass = 1000.0 / black_mass_kg_per_t_battery
    required_battery_t = target_black_mass_t * battery_t_per_t_black_mass

    gwp_per_t_battery_receiver = float(receiver_model_row.get('GWP_service_kgCO2e_tBattery', 0.0))
    receiver_energy_per_t_battery = energy_from_model_row(receiver_model_row)
    energy_per_t_black_mass = scale_energy(receiver_energy_per_t_battery, battery_t_per_t_black_mass)
    total_receiver_energy = scale_energy(receiver_energy_per_t_battery, required_battery_t)
    gwp_per_t_black_mass = gwp_per_t_battery_receiver * battery_t_per_t_black_mass
    total_receiver_gwp = gwp_per_t_black_mass * target_black_mass_t

    st.caption(
        f"Scenario values below come directly from the predictive model after applying the selected "
        f"battery mix, renewable share, and energy recovery assumptions. Model electricity demand: "
        f"{receiver_electricity:,.2f} kWh/t battery."
    )

    with st.container(border=True):
        st.markdown('### Per 1 tonne of black mass')
        st.markdown('<div class="section-note">Common basis for comparing black mass supply.</div>', unsafe_allow_html=True)
        show_normalized_scenario_metrics(
            'Black mass basis',
            1.0,
            'GWP per tonne black mass',
            gwp_per_t_black_mass,
            energy_per_t_black_mass
        )

    show_small_metric_cards([
        ('Black mass yield', f'{black_mass_kg_per_t_battery:,.2f} kg/t battery'),
        ('Battery input per tonne black mass', f'{battery_t_per_t_black_mass:,.2f} t battery')
    ])

    bm_composition_cols = [col for col in ['Li_%_BM', 'Co_%_BM', 'Ni_%_BM', 'Mn_%_BM'] if col in base_forecast_df.columns]
    if bm_composition_cols:
        composition_table = pd.DataFrame([
            {
                'Component': friendly_name(col),
                'Mass in selected black mass (kg)': float(receiver_model_row[col]) / 100 * target_black_mass_t * 1000,
                'Content in black mass (%)': float(receiver_model_row[col])
            }
            for col in bm_composition_cols
        ])
        with st.expander('Expected material outputs'):
            st.dataframe(composition_table, use_container_width=True)

    scenario_forecast_charts(
        'Forecast charts for material receiver scenario',
        receiver_model_forecast,
        [
            'Black_mass_kg_t',
            'Black_mass_purity_%',
            'Li_%_BM',
            'Co_%_BM',
            'Ni_%_BM',
            'Mn_%_BM',
            'GWP_kgCO2e_kgBM',
            'GWP_service_kgCO2e_tBattery',
            'Electricity_total_kWh_t',
            'Overall_recovery_rate_%',
            'Recovered_energy_kWh_t',
            'External_electricity_kWh_t',
            'Water_use_m3_tBattery'
        ],
        [
            'Black_mass_kg_t',
            'Black_mass_purity_%',
            'Li_%_BM',
            'GWP_kgCO2e_kgBM',
            'GWP_service_kgCO2e_tBattery'
        ],
        'receiver'
    )

    receiver_table = pd.DataFrame([
        {'Indicator': 'Target black mass', 'Value': target_black_mass_t, 'Unit': 't black mass'},
        {'Indicator': 'Battery input needed', 'Value': required_battery_t, 'Unit': 't battery'},
        {'Indicator': 'Model GWP per tonne black mass', 'Value': gwp_per_t_black_mass, 'Unit': 'kg CO2e/t black mass'},
        {'Indicator': 'Total model GWP', 'Value': total_receiver_gwp, 'Unit': 'kg CO2e'},
        {'Indicator': 'Recovered battery energy', 'Value': total_receiver_energy['Recovered_energy_kWh_t'], 'Unit': 'kWh'},
        {'Indicator': 'Grid Energy Consumption', 'Value': total_receiver_energy['External_electricity_kWh_t'], 'Unit': 'kWh'},
        {'Indicator': 'Renewable external electricity', 'Value': total_receiver_energy['Renewable_external_kWh_t'], 'Unit': 'kWh'},
        {'Indicator': 'Non-renewable external electricity', 'Value': total_receiver_energy['Nonrenewable_external_kWh_t'], 'Unit': 'kWh'}
    ])
    with st.expander('Detailed material receiver scenario table'):
        st.dataframe(receiver_table, use_container_width=True)

with tab_scenario:
    st.subheader('Scenario Simulator')
    st.markdown('Scenario adjustments are applied gradually until the final forecast month.')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('### Feedstock')
        battery_mass_pct = st.slider('Battery mass - change by the end (%)', -50.0, 100.0, 0.0, 1.0)
        nmc_pp = st.slider('NMC - change by the end (p.p.)', -30.0, 30.0, 0.0, 0.5)
        lfp_pp = st.slider('LFP - change by the end (p.p.)', -30.0, 30.0, 0.0, 0.5)
        lco_pp = st.slider('LCO - change by the end (p.p.)', -30.0, 30.0, 0.0, 0.5)
        nca_pp = st.slider('NCA - change by the end (p.p.)', -30.0, 30.0, 0.0, 0.5)
    with c2:
        st.markdown('### Process and energy')
        contamination_delta = st.slider('Contamination - delta by the end', -5.0, 5.0, 0.0, 0.1)
        soc_delta = st.slider('State of charge - delta by the end', -20.0, 20.0, 0.0, 0.5)
        feed_rate_delta = st.slider('Feed rate - delta by the end', -100.0, 100.0, 0.0, 1.0)
        sorting_delta = st.slider('Sorting efficiency - delta by the end', -10.0, 10.0, 0.0, 0.1)
        renewable_delta = st.slider('Renewable energy - delta by the end', -50.0, 50.0, 0.0, 1.0)
        recovery_delta = st.slider('Battery energy recovery - delta by the end', -30.0, 30.0, 0.0, 0.5)
    st.markdown('### Operational')
    c1, c2 = st.columns(2)
    with c1:
        electricity_pct = st.slider('Electricity demand - change by the end (%)', -50.0, 50.0, 0.0, 1.0)
        diesel_pct = st.slider('Diesel - change by the end (%)', -50.0, 50.0, 0.0, 1.0)
    with c2:
        water_pct = st.slider('Operational water - change by the end (%)', -50.0, 50.0, 0.0, 1.0)
        transport_pct = st.slider('Transport - change by the end (%)', -50.0, 50.0, 0.0, 1.0)
    scenario_config = {
        'battery_mass_pct': battery_mass_pct,
        'NMC_share_%_pp': nmc_pp,
        'LFP_share_%_pp': lfp_pp,
        'LCO_share_%_pp': lco_pp,
        'NCA_share_%_pp': nca_pp,
        'Contamination_%_delta': contamination_delta,
        'State_of_charge_%_delta': soc_delta,
        'Feed_rate_kg_h_delta': feed_rate_delta,
        'Sorting_eff_%_delta': sorting_delta,
        'Renewable_share_%_delta': renewable_delta,
        'Overall_recovery_rate_%_delta': recovery_delta,
        'Electricity_total_kWh_t_pct': electricity_pct,
        'Diesel_MJ_t_pct': diesel_pct,
        'Water_L_t_pct': water_pct,
        'Transport_km_pct': transport_pct
    }
    scenario_forecast_df = forecast_engine(result, scenario_config)
    st.session_state['scenario_forecast'] = scenario_forecast_df
    variable_scenario = st.selectbox('Choose the variable to view in the scenario', available_display, format_func=friendly_name, index=default_index, key='scenario_variable')
    st.plotly_chart(plot_history_forecast(df, base_forecast_df, scenario_forecast_df, variable_scenario), use_container_width=True, key=f"scenario_{variable_scenario}")
    comparison_df = scenario_comparison_table(base_forecast_df, scenario_forecast_df)
    st.info(explain_scenario_impact(comparison_df, variable_scenario))
    with st.expander('View full scenario table'):
        st.dataframe(scenario_forecast_df, use_container_width=True)

with tab_explain:
    st.subheader('How to interpret this dashboard')
    with st.expander('What is the Base Forecast?'):
        st.write('It is the future forecast made by the model assuming that historical patterns continue.')
    with st.expander('What is a Scenario?'):
        st.write('It is a simulation where the user changes future assumptions and the dashboard recalculates downstream effects.')
    with st.expander('What is ALR in battery composition?'):
        st.write('ALR means Additive Log-Ratio. It represents relationships between battery chemistries, such as log(NMC/NCA), then reconstructs percentages that sum to 100%.')
    with st.expander('What is battery energy recovery?'):
        st.write('Overall_recovery_rate_% is treated as the share of electricity demand supplied by recovered battery energy, reducing external electricity demand.')

with tab_logic:
    st.subheader('General model logic')
    st.markdown(
        """
        This section explains, in simple terms, how the dashboard transforms historical monthly data
        into future forecasts and scenario simulations.
        """
    )

    st.markdown('### 1. Data input')
    st.write(
        """
        The user uploads an Excel file with a monthly historical dataset. Each row represents one
        observed month, and each column represents a system variable, such as battery mass, battery
        chemistry, electricity demand, water use, transport, material recovery, black mass quality,
        and climate impact.
        """
    )

    st.markdown('### 2. Time organization')
    st.write(
        """
        If the file contains a month/date column, the dashboard first sorts the data chronologically.
        Then it creates an internal monthly index called `Month_index`. This gives the model a clear
        time order, so it can understand whether a value belongs to an earlier or later month.
        """
    )

    st.markdown('### 3. Historical memory with lags')
    st.write(
        """
        To forecast a variable, the model does not only look at the current month. It also uses past
        values of that same variable, called `lags`.

        For example, to forecast the next NMC share, the model can look at the previous month, two
        months ago, three months ago, six months ago, and twelve months ago.
        """
    )
    st.info(
        """
        In simple words: lags give memory to the model. Without them, the model would have much less
        information about the recent trajectory of each variable.
        """
    )

    st.markdown('### 4. Trend + XGBoost')
    st.write(
        """
        The dashboard uses a hybrid predictive model. First, a simple linear regression learns the
        general time trend of each variable. Then XGBoost learns the remaining patterns that the
        linear trend does not explain.

        In simple words, XGBoost builds many small decision trees. Each tree tries to correct part of
        the previous error. After many trees, the model combines these corrections to make a final
        prediction.
        """
    )
    st.code(
        """
        Final prediction = linear time trend + XGBoost correction
        """,
        language='text'
    )

    st.markdown('### 5. Hierarchical process structure')
    st.write(
        """
        The model follows the physical logic of the recycling process instead of treating every
        variable as independent.

        1. It first forecasts the **feedstock**, meaning incoming battery mass and battery chemistry.
        2. Then it forecasts **pre-recycling separated outputs**, such as ferrous metals, aluminium,
           copper, mixed plastics, and other byproducts. These depend on battery mass and chemistry,
           but they are not used as inputs for later stages.
        3. Next it forecasts **process variables**, using feedstock as input.
        4. Then it forecasts **energy recovery**, using feedstock and process conditions.
        5. After that it forecasts **operational variables**, such as electricity demand, diesel,
           water, and transport.
        6. Finally it forecasts the **final outputs**, such as black mass production, black mass
           purity, recovered chemical content, GWP, and water use.
        """
    )
    st.info(
        """
        This structure is important because, in the real system, one stage influences the next one.
        For example, battery chemistry can affect separated materials, process behavior, black mass
        output, and environmental indicators.
        """
    )

    st.markdown('### 6. Battery composition')
    st.write(
        """
        Battery composition is special because NMC, LFP, LCO, and NCA are parts of a total and must
        sum to 100%. For this reason, the dashboard tests different strategies for forecasting the
        battery mix.
        """
    )
    st.markdown(
        """
        - **Direct forecast of all battery types:** forecast NMC, LFP, LCO, and NCA separately, then normalize to 100%.
        - **Direct forecast with residual balance:** forecast three battery types and calculate the fourth as the remaining balance.
        - **Log-ratio method:** forecast relationships between battery types, such as `log(NMC/NCA)`, then reconstruct percentages.
        """
    )
    st.info(
        f"Best strategy: {battery_mix_method['Best strategy']}  \n"
        f"Reference battery type: {battery_mix_method['Reference battery type']}"
    )

    st.markdown('### 7. Temporal validation')
    st.write(
        """
        To check whether the model can forecast future behavior, the dashboard uses a temporal
        backtest. This means the model trains on older months and then tries to predict the most
        recent months.

        This is more realistic than a random split because real forecasting always starts from the
        past and tries to estimate what comes next.
        """
    )

    st.markdown('### 8. Recursive future forecasting')
    st.write(
        """
        Future forecasting is done month by month. First, the model predicts the first future month.
        Then that predicted month is added to the history. After that, the model uses this updated
        history to predict the second future month.

        This process continues until the selected forecast horizon is complete.
        """
    )

    st.markdown('### 9. Energy recovery and external electricity')
    st.write(
        """
        Battery energy recovery is predicted before operational and final outputs. It is used because
        recovered battery energy can reduce the amount of electricity that must be purchased from the
        grid.
        """
    )
    st.code(
        """
        Recovered energy = Total electricity demand x Battery energy recovery rate
        External electricity = Total electricity demand - Recovered energy
        Renewable external electricity = External electricity x Renewable share
        Non-renewable external electricity = External electricity x (1 - Renewable share)
        """,
        language='text'
    )

    st.markdown('### 10. Scenario simulation')
    st.write(
        """
        Scenario tabs allow the user to change selected future assumptions, such as battery chemistry,
        renewable electricity share, and battery energy recovery. The dashboard then recalculates the
        forecast using those scenario inputs.

        The dedicated scenario tabs apply the selected assumptions directly. The general Scenario
        Simulator applies changes gradually until the final forecast month.
        """
    )

    st.markdown('### 11. Reliability and limitations')
    st.write(
        """
        Reliability is estimated mainly from the temporal backtest. If the model predicted recent
        months well, the forecast is considered more reliable. If it struggled, the forecast should be
        interpreted with more caution.
        """
    )
    st.warning(
        """
        The model learns historical patterns. It cannot forecast external events that are not
        represented in the dataset, such as sudden market shifts, new regulations, radical technology
        changes, or operational decisions never seen in the historical data.
        """
    )

with tab_export:
    st.subheader('Export Results')
    scenario_forecast_df = st.session_state.get('scenario_forecast')
    sheets = {
        'Historical_clean_data': df,
        'Base_forecast': base_forecast_df,
        'Model_metrics': result['metrics_df'],
        'Simple_reliability': result['simple_reliability_df'],
        'Composition_strategy': result['composition_strategy_summary_df'],
        'Composition_metrics': result['composition_metrics_df']
    }
    if scenario_forecast_df is not None:
        sheets['Scenario_forecast'] = scenario_forecast_df
        sheets['Scenario_comparison'] = scenario_comparison_table(base_forecast_df, scenario_forecast_df)
    excel_bytes = to_excel_bytes(sheets)
    st.download_button(
        label='Download Excel with results',
        data=excel_bytes,
        file_name='battery_lca_forecast_dashboard_results.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

