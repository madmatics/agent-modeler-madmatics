from __future__ import annotations
from typing import Dict, Any, Iterable, Tuple
import numpy as np
import pandas as pd
import sys
import os
import copy
from properscoring import crps_ensemble
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
# MMM 폴더 경로 추가
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(os.getcwd()), 'ACTION_MMM/MMM')
    )
)

# ---------------------------
# 1차: madmatics import 시도
# ---------------------------
try:
    from madmatics.mmm.evaluation import (
        calculate_metric_distributions,
        summarize_metric_distributions,
    )
    print("📌 madmatics.mmm.evaluation 모듈 로드 성공")

# ---------------------------
# 2차: madmatics 실패 → pymc_marketing fallback
# ---------------------------
except Exception as e:
    print("⚠️ madmatics import 실패, pymc_marketing으로 대체합니다.")
    print(f"에러 내용: {e}")

    try:
        from pymc_marketing.mmm.evaluation import (
            calculate_metric_distributions,
            summarize_metric_distributions,
        )
        print("📌 pymc_marketing.mmm.evaluation 모듈 로드 성공")
    except Exception as e2:
        print("❌ pymc_marketing import도 실패했습니다.")
        print(f"에러 내용: {e2}")
        raise e2  # 더이상 대체할 옵션 없으므로 에러 발생시킴
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_features_vs_sales(data, features, target="sales_amount", corr_method="kendall"):
    """
    여러 feature와 target(sales_amount)을 날짜별로 subplot에 그려 비교합니다.
    
    Parameters
    ----------
    data : pd.DataFrame
        데이터프레임 (date, target, features 포함)
    features : list of str
        비교할 feature 컬럼 리스트
    target : str, default="sales_amount"
        타겟 변수명
    corr_method : str, default="kendall"
        상관계수 계산 방식 ('pearson', 'spearman', 'kendall')
    """
    n_channels = len(features)

    fig, axes = plt.subplots(
        nrows=n_channels,
        ncols=1,
        figsize=(15, 3 * n_channels),
        sharex=True,
        sharey=False,
        layout="constrained",
    )

    if n_channels == 1:
        axes = [axes]  # 하나만 있으면 리스트로 변환

    for i, channel in enumerate(features):
        ax = axes[i]
        ax_twin = ax.twinx()

        # feature 값 (왼쪽 y축)
        sns.lineplot(data=data, x="date", y=channel, color=f"C{i}", ax=ax)

        # sales_amount (오른쪽 y축)
        sns.lineplot(data=data, x="date", y=target, color="black", ax=ax_twin)

        # 상관계수
        correlation = data[[channel, target]].corr(method=corr_method).iloc[0, 1]

        ax_twin.grid(False)
        ax.set(title=f"{channel} (Correlation: {correlation:.2f})")

    axes[-1].set_xlabel("date")
    plt.show()

def plot_corr_heatmap(df, cols, target="sales_amount", title=None, annotate=False):
    """
    선택한 칼럼들과 타겟 변수 간 상관관계 히트맵을 그립니다.
    
    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터프레임
    cols : list of str
        비교할 feature 컬럼 리스트
    target : str, default="sales_amount"
        타겟 변수명
    title : str, optional
        그래프 제목 (None이면 자동 생성)
    annotate : bool, default=False
        True이면 셀 안에 상관계수 숫자를 표시
    """
    use_cols = [target] + cols
    num_df = df[use_cols].apply(pd.to_numeric, errors="coerce")
    corr = num_df.corr()

    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    # colorbar
    plt.colorbar(im, fraction=0.046, pad=0.04)

    # ticks
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)

    # title
    if title is None:
        title = f"Correlation Heatmap: {target} vs Selected Features"
    plt.title(title, fontsize=14)

    # annotate
    if annotate:
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                         ha="center", va="center", color="black", fontsize=8)

    # plt.tight_layout()
    plt.show()

def calculate_vif(data, features, threshold=10, dropna=True):
    """
    주어진 feature들의 VIF(Variance Inflation Factor)를 계산합니다.
    NaN/Inf/상수열(분산 0) 컬럼을 자동으로 제거한 뒤 계산합니다.
    
    Parameters
    ----------
    data : pd.DataFrame
        입력 데이터 (features 포함)
    features : list of str
        VIF 계산 대상 feature 컬럼 리스트
    threshold : float, default=10
        다중공선성 판단 기준. 이 값보다 작은 VIF만 필터링하여 반환
    dropna : bool, default=True
        NaN/Inf가 포함된 행 제거 여부
    
    Returns
    -------
    pd.DataFrame
        Variable, VIF 값이 담긴 DataFrame
    """
    X = data[features].copy()

    # 숫자로 강제 변환
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Inf → NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # dropna 옵션
    if dropna:
        X = X.dropna(axis=0, how="any")

    # 분산 0 컬럼 제거
    drop_cols = []
    for col in X.columns:
        if np.nanvar(X[col].values) == 0:
            drop_cols.append(col)
    if drop_cols:
        X = X.drop(columns=drop_cols)
        print(f"[VIF] 제거된 상수열 컬럼: {drop_cols}")

    if X.shape[1] < 2:
        raise ValueError("VIF를 계산할 유효한 컬럼이 부족합니다.")

    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif[vif["VIF"] < threshold].reset_index(drop=True)

def plot_waterfall_real_scale_from_mmm(
    mmm,
    baseline_components: list[str] | None = None,
    agg: str = "sum",   # "sum" or "mean"
    figsize: tuple[int, int] = (14, 7),
    intercept_label: str = "Intercept (Baseline)",
    **kwargs,
) -> plt.Figure:
    """
    log(sales)를 타겟으로 쓰는 MMM에서,
    time_varying_intercept / time_varying_media 여부와 상관없이
    '실제 매출(real scale)' 기준 waterfall 을 그리는 함수.

    가정:
    - mmm.fit_result (xarray.Dataset)에
        "mu"                      : (chain, draw, date)
        "channel_contributions"   : (chain, draw, date, channel)
        "control_contributions"   : (chain, draw, date, control)
      가 존재한다.
    - log 변환은 ln(sales) 기준이라고 가정.
      (log1p 를 썼다면 np.exp → np.expm1 으로 바꿔야 함)
    """

    idata = mmm.fit_result

    # 1) posterior mean linear predictor (log-space)
    mu = idata["mu"].mean(dim=("chain", "draw"))  # (date,)
    dates = mu["date"].values

    # 2) 각 컴포넌트의 log-contribution 평균
    comp_df = pd.DataFrame(index=dates)

    if "channel_contributions" in idata:
        ch_contrib = idata["channel_contributions"].mean(dim=("chain", "draw"))
        for ch in ch_contrib["channel"].values:
            comp_df[str(ch)] = ch_contrib.sel(channel=ch).values

    if "control_contributions" in idata:
        ct_contrib = idata["control_contributions"].mean(dim=("chain", "draw"))
        for ct in ct_contrib["control"].values:
            comp_df[str(ct)] = ct_contrib.sel(control=ct).values

    all_components = list(comp_df.columns)

    # baseline_components: baseline 으로 빼고 싶은 애들 (trend, yearly_seasonality 등)
    if baseline_components is None:
        baseline_components = []
    baseline_components = set(baseline_components)

    # driver_components: 실제로 bar 로 그리고 싶은 애들
    driver_components = [c for c in all_components if c not in baseline_components]

    # -------- 시간별 real-scale contribution 계산 --------
    contrib_accum = {c: 0.0 for c in driver_components}
    y_base_list = []
    lift_total_list = []

    for t in dates:
        row = comp_df.loc[t]

        mu_t = float(mu.sel(date=t).values)

        # baseline_log_t = mu_t - Σ_{drivers} L_{j,t}
        # (driver 말고 나머지 항 = intercept + time-varying intercept + baseline 컴포넌트 전부)
        if driver_components:
            baseline_log_t = mu_t - float(row[driver_components].sum())
        else:
            baseline_log_t = mu_t

        # real-scale prediction
        y_hat_t = np.exp(mu_t)
        y_base_t = np.exp(baseline_log_t)
        lift_total_t = y_hat_t - y_base_t

        y_base_list.append(y_base_t)
        lift_total_list.append(lift_total_t)

        # driver별 solo lift
        lift_raw = {}
        for comp in driver_components:
            L_jt = float(row[comp])
            y_jt = np.exp(baseline_log_t + L_jt)
            lift_raw[comp] = y_jt - y_base_t

        lift_raw_series = pd.Series(lift_raw)
        denom = lift_raw_series.sum()

        if denom <= 0 or lift_total_t <= 0:
            weights = lift_raw_series * 0.0
        else:
            weights = lift_raw_series / denom

        contrib_t = weights * lift_total_t

        for comp in driver_components:
            contrib_accum[comp] += contrib_t[comp]

    # -------- 시간축 aggregation (sum / mean) --------
    n = len(dates)
    if agg == "mean":
        contrib_accum = {k: v / n for k, v in contrib_accum.items()}
        base_total = float(np.mean(y_base_list))
    else:  # "sum"
        base_total = float(np.sum(y_base_list))

    contrib_series = pd.Series(contrib_accum).sort_values(ascending=True)

    # 전체 매출 = baseline + incremental
    total_sales = base_total + contrib_series.sum()

    if total_sales != 0:
        perc_inc = contrib_series / total_sales * 100
        perc_intercept = base_total / total_sales * 100
    else:
        perc_inc = contrib_series * 0.0
        perc_intercept = 0.0

    # -------- intercept(베이스라인)까지 포함해서 waterfall plot --------
    components = [intercept_label] + list(contrib_series.index)
    values = [base_total] + list(contrib_series.values)
    percs = [perc_intercept] + list(perc_inc.values)

    fig, ax = plt.subplots(figsize=figsize, layout="constrained", **kwargs)

    cumulative = 0.0
    for i, (comp, val, pct) in enumerate(zip(components, values, percs)):
        if i == 0:
            color = "0.7"          # baseline: 회색
        else:
            color = "C0" if val >= 0 else "C3"

        bar_start = cumulative if val >= 0 else cumulative + val
        ax.barh(comp, val, left=bar_start, color=color, alpha=0.7)

        if val > 0:
            cumulative += val

        label_pos = bar_start + val / 2.0
        ax.text(
            label_pos,
            i,
            f"{val:,.0f}\n({pct:.1f}%)",
            ha="center",
            va="center",
            fontsize=10,
        )

    ax.set_title("Response Decomposition Waterfall (Real Sales Scale)")
    ax.set_xlabel("Cumulative Contribution (sales)")
    ax.set_ylabel("Components")

    ax.ticklabel_format(style="plain", axis="x")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks(np.arange(len(components)))
    ax.set_yticklabels(components)

    return fig

def to_weekly(
    df, 
    channel_cols, 
    control_sum_cols,     
    control_mean_cols,    
    target
):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    df_w = pd.DataFrame()
    shift_dict = {}

    def safe_log_transform(series, col_name):
        min_val = series.min()
        if min_val < 0:
            shift = abs(min_val) + 1
            series = series + shift
        else:
            shift = 0

        shift_dict[col_name] = shift
        return np.log1p(series)

    # ---- 광고비 (sum) ---
    for col in channel_cols:
        weekly = (
            df[col]
            .resample("W-SUN", label="left", closed="left")
            .sum()
        )
        df_w[col] = safe_log_transform(weekly, col)

    # ---- control sum ---
    for col in control_sum_cols:
        weekly = (
            df[col]
            .resample("W-SUN", label="left", closed="left")
            .sum()
        )
        df_w[col] = safe_log_transform(weekly, col)

    # ---- control mean ---
    for col in control_mean_cols:
        weekly = (
            df[col]
            .resample("W-SUN", label="left", closed="left")
            .mean()
        )
        df_w[col] = safe_log_transform(weekly, col)

    # ---- target ---
    weekly_target = (
        df[target]
        .resample("W-SUN", label="left", closed="left")
        .sum()
    )
    df_w[target] = safe_log_transform(weekly_target, target)

    return df_w.reset_index(), shift_dict


def inverse_log1p(x, shift):
    return np.expm1(x) - shift

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0
    if not np.any(mask):
        return np.nan

    return (np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])).mean() * 100

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot

def prepare_ppc_samples_exp(ppc_da, n_obs, shift):
    """
    ppc_da: (n_chains, n_draws, n_obs)
    반환: (n_obs, n_samples) on original scale
    """
    samples = np.asarray(ppc_da)
    samples_2d = samples.reshape(-1, n_obs).T   # (n_obs, n_chains*n_draws)
    samples_exp = inverse_log1p(samples_2d, shift)
    return samples_exp


def evaluate_log_model_with_mape_crps_r2(
    is_pred_df,
    oos_pred_df,
    is_pred,
    oos_pred,
    output_var,        # mmm.output_var
    shift_dict_train,  # to_weekly(train) 에서 나온 shift
    shift_dict_test,   # to_weekly(test) 에서 나온 shift
    target_col,        # target 컬럼 이름 (예: 'sales_amount' 또는 mmm.output_var 와 동일)
):
    # 1) 로그 → 원래 스케일 (point prediction)
    shift_train = shift_dict_train[target_col]
    shift_test  = shift_dict_test[target_col]

    y_true_is = inverse_log1p(is_pred_df["y_true"].values, shift_train)
    y_pred_is = inverse_log1p(is_pred_df["y_pred"].values, shift_train)

    y_true_oos = inverse_log1p(oos_pred_df["y_true"].values, shift_test)
    y_pred_oos = inverse_log1p(oos_pred_df["y_pred"].values, shift_test)

    # 2) MAPE
    is_mape = mean_absolute_percentage_error(y_true_is, y_pred_is)
    oos_mape = mean_absolute_percentage_error(y_true_oos, y_pred_oos)

    # 3) R²
    is_r2 = r2_score(y_true_is, y_pred_is)
    oos_r2 = r2_score(y_true_oos, y_pred_oos)

    # 4) posterior predictive 샘플 준비 (원래 스케일)
    n_obs_is = is_pred_df.shape[0]
    n_obs_oos = oos_pred_df.shape[0]

    is_ppc_samples_exp = prepare_ppc_samples_exp(
        is_pred[output_var].values, n_obs_is, shift_train
    )
    oos_ppc_samples_exp = prepare_ppc_samples_exp(
        oos_pred[output_var].values, n_obs_oos, shift_test
    )

    # 5) CRPS
    is_crps_scores = crps_ensemble(y_true_is, is_ppc_samples_exp)
    oos_crps_scores = crps_ensemble(y_true_oos, oos_ppc_samples_exp)

    is_crps = np.mean(is_crps_scores)
    oos_crps = np.mean(oos_crps_scores)

    print(f"In-sample (Train) MAPE: {is_mape:.2f}%")
    print(f"Out-of-sample (Test) MAPE: {oos_mape:.2f}%")
    print(f"In-sample (Train) R^2  : {is_r2:.4f}")
    print(f"Out-of-sample (Test) R^2: {oos_r2:.4f}")
    print(f"In-sample (Train) CRPS : {is_crps:.4f}")
    print(f"Out-of-sample (Test) CRPS: {oos_crps:.4f}")

    return {
        "is_mape": is_mape,
        "oos_mape": oos_mape,
        "is_r2": is_r2,
        "oos_r2": oos_r2,
        "is_crps": is_crps,
        "oos_crps": oos_crps,
    }

# ---------------------------
# 내부 유틸
# ---------------------------
def _get_output_var(mmm) -> str:
    """
    madmatics의 mmm 객체에서 타겟 변수명(key)을 안전하게 찾는다.
    통상 mmm.output_var 가 존재하므로 우선 사용하고,
    없을 경우 'y'를 기본값으로 사용.
    """
    return getattr(mmm, "output_var", "y")

def _posterior_df_from_predictive(
    mmm,
    X_pred: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    *,
    date_col: str = "date",
    include_last_observations: bool = False,
    var_names: Iterable[str] = ("y",),
    q_lo: float = 0.05,
    q_hi: float = 0.95,
) -> pd.DataFrame:
    """
    sample_posterior_predictive를 호출해 평균/구간을 담은 예측 DF를 만든다.
    반환 컬럼: [date, y_true, y_pred, ci_lower, ci_upper]
    """
    is_train = bool(len(X_pred) == len(y_true))
    pred = mmm.sample_posterior_predictive(
        X_pred=X_pred,
        extend_idata=False,
        combined=False,
        include_last_observations=include_last_observations,  # 홀드아웃에서 Adstock 등 반영
        var_names=list(var_names),
    )

    out_key = _get_output_var(mmm)  # 보통 'y'
    posterior = pred[out_key]       # xarray.DataArray [chain, draw, time]
    mean_ = posterior.mean(dim=["chain", "draw"], keep_attrs=True).data
    lo_ = posterior.quantile(q_lo, dim=["chain", "draw"]).data
    hi_ = posterior.quantile(q_hi, dim=["chain", "draw"]).data

    df = pd.DataFrame({
        "date": X_pred[date_col].to_numpy(),      # 날짜
        "y_true": np.asarray(y_true).reshape(-1), # 정답
        "y_pred": mean_.reshape(-1),              # 예측 평균(포인트 포캐스트)
        "ci_lower": lo_.reshape(-1),              # 하한
        "ci_upper": hi_.reshape(-1),              # 상한
    })
    df.attrs["is_train"] = is_train
    df.attrs["interval"] = (q_lo, q_hi)
    return df

def _compute_metric_summaries_on_train(
    mmm,
    data_train: pd.DataFrame,
    *,
    rng_seed: str | int = "mmm-evaluation",
    hdi_prob: float = 0.89,
    metrics: Iterable[str] = ("r_squared", "rmse", "mae", "mape"),
) -> Dict[str, Dict[str, float]]:
    """
    In-sample(Train)에 대해 posterior predictive 샘플을 뽑아
    madmatics의 evaluation 유틸로 지표 분포를 계산/요약한다.
    """
    rng = np.random.default_rng(sum(map(ord, str(rng_seed))) if isinstance(rng_seed, str) else rng_seed)

    # 패키지 시그니처에 맞춰 data_train 자체를 넣는 형태 유지
    posterior_preds = mmm.sample_posterior_predictive(data_train, random_seed=rng)

    # y_true: mmm.y (훈련 타깃), y_pred: posterior 샘플 (np.ndarray)
    y_true = mmm.y
    # posterior_preds.y 가 xarray.DataArray일 가능성 -> numpy 변환
    if hasattr(posterior_preds, "y"):
        y_pred = posterior_preds.y.to_numpy()
    else:
        # 혹시 구조가 다르면 알려주기
        raise RuntimeError("posterior_preds 안에서 'y'를 찾지 못했습니다. madmatics 버전/반환 구조를 확인하세요.")

    metric_distributions = calculate_metric_distributions(
        y_true=y_true,
        y_pred=y_pred,
        metrics_to_calculate=list(metrics),
    )
    summaries = summarize_metric_distributions(metric_distributions, hdi_prob=hdi_prob)
    return summaries

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return (np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])).mean() * 100

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot

def _posterior_pred_to_df(
    pred_idata_like,
    mmm,
    X_df,
    target_col: str,
    date_col: str = "date",
    q_low: float = 0.05,
    q_high: float = 0.95,
):
    """
    pred_idata_like: mmm.sample_posterior_predictive(...) 결과(dict-like)
    """
    y_samples = pred_idata_like[mmm.output_var]  # xarray DataArray 기대 (chain, draw, obs)
    y_mean = y_samples.mean(dim=["chain", "draw"], keep_attrs=True).data
    y_lo = y_samples.quantile(q_low, dim=["chain", "draw"]).data
    y_hi = y_samples.quantile(q_high, dim=["chain", "draw"]).data

    df = pd.DataFrame({
        date_col: X_df[date_col].values,
        "y_true": X_df[target_col].values,
        "y_pred": np.asarray(y_mean).reshape(-1),
        "ci_lower": np.asarray(y_lo).reshape(-1),
        "ci_upper": np.asarray(y_hi).reshape(-1),
    })
    return df

def _crps_from_ppc(y_true, y_samples_xr, n_obs: int):
    """
    y_samples_xr: xarray DataArray (chain, draw, obs) -> (n_obs, n_samples)로 변환 후 CRPS
    """
    # (chain*draw, obs) -> (obs, chain*draw)
    samples_2d = y_samples_xr.values.reshape(-1, n_obs).T
    scores = crps_ensemble(y_true, samples_2d)
    return float(np.mean(scores))

def evaluate_mmm_train_test_ppc(
    mmm,
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    target_col: str,
    date_col: str = "date",
    var_names=("y",),
    q_low: float = 0.05,
    q_high: float = 0.95,
    include_last_observations_oos: bool = True,
    extend_idata: bool = False,
    combined: bool = False,
):
    """
    반환:
      - is_pred_df, oos_pred_df: date/y_true/y_pred/CI 포함
      - metrics: dict (R2/MAPE/CRPS for train & test)
      - raw: dict (원하면 posterior predictive 원본도 접근 가능)
    """

    # -------------------------
    # In-sample posterior predictive
    # -------------------------
    is_pred = mmm.sample_posterior_predictive(
        X_pred=data_train,
        extend_idata=extend_idata,
        combined=combined,
        include_last_observations=False,
        var_names=list(var_names),
    )

    is_pred_df = _posterior_pred_to_df(
        pred_idata_like=is_pred,
        mmm=mmm,
        X_df=data_train,
        target_col=target_col,
        date_col=date_col,
        q_low=q_low,
        q_high=q_high,
    )

    # -------------------------
    # Out-of-sample posterior predictive
    # -------------------------
    oos_pred = mmm.sample_posterior_predictive(
        X_pred=data_test,
        extend_idata=extend_idata,
        combined=combined,
        include_last_observations=include_last_observations_oos,
        var_names=list(var_names),
    )

    oos_pred_df = _posterior_pred_to_df(
        pred_idata_like=oos_pred,
        mmm=mmm,
        X_df=data_test,
        target_col=target_col,
        date_col=date_col,
        q_low=q_low,
        q_high=q_high,
    )

    # -------------------------
    # Metrics
    # -------------------------
    is_r2 = r2_score(is_pred_df["y_true"].values, is_pred_df["y_pred"].values)
    oos_r2 = r2_score(oos_pred_df["y_true"].values, oos_pred_df["y_pred"].values)

    is_mape = mean_absolute_percentage_error(is_pred_df["y_true"].values, is_pred_df["y_pred"].values)
    oos_mape = mean_absolute_percentage_error(oos_pred_df["y_true"].values, oos_pred_df["y_pred"].values)

    # CRPS uses posterior predictive distribution (not mean)
    is_crps = _crps_from_ppc(
        y_true=is_pred_df["y_true"].values,
        y_samples_xr=is_pred[mmm.output_var],
        n_obs=is_pred_df.shape[0],
    )
    oos_crps = _crps_from_ppc(
        y_true=oos_pred_df["y_true"].values,
        y_samples_xr=oos_pred[mmm.output_var],
        n_obs=oos_pred_df.shape[0],
    )

    metrics = {
        "train": {"R2": float(is_r2), "MAPE": float(is_mape), "CRPS": float(is_crps)},
        "test":  {"R2": float(oos_r2), "MAPE": float(oos_mape), "CRPS": float(oos_crps)},
    }

    raw = {"is_pred": is_pred, "oos_pred": oos_pred}

    return is_pred_df, oos_pred_df, metrics, raw

def prettify_metrics(
    metrics: dict,
    y_scale_train: float | None = None,
    y_scale_test: float | None = None,
    crps_scale_mode: str = "mean",  # "mean" or "median"
    clip_0_100: bool = True,
):
    """
    metrics = {
      "train": {"R2": ..., "MAPE": ..., "CRPS": ...},
      "test":  {"R2": ..., "MAPE": ..., "CRPS": ...},
    }

    returns:
      - metrics_df: 비교용 표
      - score_df: 모두 '클수록 좋은 점수'로 변환한 표
    """

    # 1) 원본 metrics -> DataFrame
    metrics_df = (
        pd.DataFrame(metrics)
          .T[["R2", "MAPE", "CRPS"]]
          .rename_axis("sample")
          .reset_index()
    )

    # 2) CRPS 스케일(단위 정규화) 결정
    # 외부에서 주면 그걸 쓰고, 아니면 train/test 각자 scale이 있어야 함
    # (권장: y_true 평균이나 중앙값)
    if y_scale_train is None or y_scale_test is None:
        # scale을 안 넣으면 CRPS 점수는 NaN 처리(안전)
        crps_train_scale = np.nan
        crps_test_scale = np.nan
    else:
        crps_train_scale = float(y_scale_train)
        crps_test_scale = float(y_scale_test)

    # 3) "클수록 좋게" 점수화
    score_rows = []
    for s in ["train", "test"]:
        r2 = float(metrics[s]["R2"])
        mape = float(metrics[s]["MAPE"])
        crps = float(metrics[s]["CRPS"])

        # MAPE -> Accuracy (100이 최고)
        acc = 100.0 - mape

        # CRPS -> Skill (100이 최고)
        scale = crps_train_scale if s == "train" else crps_test_scale
        if np.isnan(scale) or scale == 0:
            crps_skill = np.nan
        else:
            crps_skill = 100.0 * (1.0 - (crps / scale))

        if clip_0_100:
            acc = float(np.clip(acc, 0, 100))
            if not np.isnan(crps_skill):
                crps_skill = float(np.clip(crps_skill, 0, 100))

        score_rows.append({
            "sample": s,
            "R2 (higher better)": r2,
            "MAPE Acc = 100 - MAPE": acc,
            "CRPS Skill (scaled)": crps_skill,
        })

    score_df = pd.DataFrame(score_rows)

    # 4) 보기 좋게 퍼센트/소수점 정리한 display용 버전도 같이
    display_df = metrics_df.copy()
    display_df["R2"] = display_df["R2"].map(lambda x: f"{x:.3f}")
    display_df["MAPE"] = display_df["MAPE"].map(lambda x: f"{x:.2f}%")
    display_df["CRPS"] = display_df["CRPS"].map(lambda x: f"{x:.4f}")

    return metrics_df, score_df, display_df

# ---------------------------
# 외부 공개 함수
# ---------------------------
def build_in_sample_df(
    mmm,
    data_train: pd.DataFrame,
    *,
    target_col: str,
    date_col: str = "date",
    q_lo: float = 0.05,
    q_hi: float = 0.95,
) -> pd.DataFrame:
    """
    훈련 구간(In-sample)의 posterior predictive 결과를 DF로 반환.
    """
    return _posterior_df_from_predictive(
        mmm,
        X_pred=data_train,
        y_true=data_train[target_col],
        date_col=date_col,
        include_last_observations=False,
        var_names=("y",),
        q_lo=q_lo, q_hi=q_hi,
    )

def build_df(
    mmm,
    data_test: pd.DataFrame,
    *,
    target_col: str,
    date_col: str = "date",
    q_lo: float = 0.05,
    q_hi: float = 0.95,
) -> pd.DataFrame:
    """
    홀드아웃 구간(Out-of-sample)의 posterior predictive 결과를 DF로 반환.
    include_last_observations=True 로 adstock 등 상태를 이어받도록 함.
    """
    return _posterior_df_from_predictive(
        mmm,
        X_pred=data_test,
        y_true=data_test[target_col],
        date_col=date_col,
        include_last_observations=True,
        var_names=("y",),
        q_lo=q_lo, q_hi=q_hi,
    )

def compute_train_metric_summaries(
    mmm,
    data_train: pd.DataFrame,
    *,
    rng_seed: str | int = "mmm-evaluation",
    hdi_prob: float = 0.89,
    metrics: Iterable[str] = ("r_squared", "rmse", "mae", "mape"),
) -> Dict[str, Dict[str, float]]:
    """
    In-sample(Train) 지표 요약만 필요할 때 호출.
    반환 예:
    {
      "rmse": {"mean":..., "median":..., "std":..., "89%_hdi_lower":..., "89%_hdi_upper":...},
      ...
    }
    """
    return _compute_metric_summaries_on_train(
        mmm,
        data_train=data_train,
        rng_seed=rng_seed,
        hdi_prob=hdi_prob,
        metrics=metrics,
    )

def run_full_evaluation(
    mmm,
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    *,
    target_col: str,
    date_col: str = "date",
    rng_seed: str | int = "mmm-evaluation",
    hdi_prob: float = 0.89,
    metrics: Iterable[str] = ("r_squared", "rmse", "mae", "mape"),
    q_lo: float = 0.05,
    q_hi: float = 0.95,
) -> Dict[str, Any]:
    """
    한 번에:
      1) 훈련 구간 예측 DF
      2) 홀드아웃 구간 예측 DF
      3) 훈련 구간 지표 요약(HDI 포함)
    을 반환한다.
    """
    train_df = build_in_sample_df(
        mmm, data_train, target_col=target_col, date_col=date_col, q_lo=q_lo, q_hi=q_hi
    )
    test_df = build_df(
        mmm, data_test, target_col=target_col, date_col=date_col, q_lo=q_lo, q_hi=q_hi
    )
    metric_summaries = compute_train_metric_summaries(
        mmm, data_train, rng_seed=rng_seed, hdi_prob=hdi_prob, metrics=metrics
    )
    return {metric: summary["mean"] for metric, summary in metric_summaries.items()}


def to_weekly_agg(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    target_col: str = "sales",
    channel_columns: list[str] = None,
    control_columns: list[str] = None,
    week_rule: str = "W-MON",  # 월요일 기준 주간. 일요일이면 "W-SUN"
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if channel_columns is None:
        channel_columns = []
    if control_columns is None:
        control_columns = []

    # 집계 방식 딕셔너리
    agg_dict: dict[str, str] = {}

    # 1) 타겟은 sum
    agg_dict[target_col] = "sum"

    # 2) 채널(spend 등)은 sum
    for col in channel_columns:
        agg_dict[col] = "sum"

    # 3) control 변수는 mean
    for col in control_columns:
        agg_dict[col] = "mean"

    # 4) 나머지 수치형 컬럼이 있으면 적당히 sum/mean 중 하나 골라서 추가해도 됨
    #    (지금은 지정한 것만 집계)

    weekly = (
        df.set_index(date_col)
          .resample(week_rule)
          .agg(agg_dict)
          .reset_index()
          .rename(columns={date_col: "date"})
    )

    return weekly

def plot_in_sample(X, y, ax, n_points: int = 20):
    """실제 데이터 (In-sample) 시각화"""
    actuals = y.to_frame().set_index(X['date']).iloc[-n_points:]
    actuals.plot(ax=ax, color="black", label="Actuals (In-Sample)", linestyle="-", linewidth=1.5)
    return ax

def plot_out_of_sample(X_out_of_sample, y_out_of_sample, actual_values, ax, color, label):
    """모델 예측 데이터 (Out-of-sample) + 실제값 시각화"""
    y_out_of_sample_groupby = y_out_of_sample["y"].to_series().groupby("date")

    lower, upper = quantiles = [0.025, 0.975]
    conf = y_out_of_sample_groupby.quantile(quantiles).unstack()
    
    # 신뢰 구간 시각화
    ax.fill_between(
        X_out_of_sample['date'].dt.to_pydatetime(),
        conf[lower],
        conf[upper],
        alpha=0.3,
        color=color,
        label=f"{label} 95% CI"
    )

    # 예측 평균 추가
    mean = y_out_of_sample_groupby.mean()
    mean.plot(ax=ax, marker="o", label=f"{label} Mean Prediction", color=color, linestyle="--", linewidth=1.5)

    # 실제값 추가 (Out-of-Sample 구간)
    actual_values.plot(ax=ax, color="black", label="Actuals (Out-of-Sample)", linestyle="-", linewidth=1.5)

    ax.set(ylabel="Original Target Scale", title="Out-of-Sample Predictions for MMM")
    return ax

def plot_prediction_comparison(data_train, data_test, y_out_of_sample, target):
    """전체 시각화: 실제 데이터 vs 예측 데이터 + b2c 포함"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 실제 데이터 (In-Sample) 시각화
    plot_in_sample(data_train, data_train[target], ax=ax)

    # 실제값을 Out-of-Sample 구간에서도 표시
    actual_out_of_sample = data_test[target].to_frame().set_index(data_test['date'])

    # 모델 예측 시각화 + 실제값 추가
    plot_out_of_sample(data_test, y_out_of_sample, actual_out_of_sample, ax=ax, color="blue", label="Out of Sample")

    # MSE 계산 및 출력 (b2c와 비교)
    actual_values = data_test[target].values
    predicted_values = y_out_of_sample["y"].mean().values
    mse = np.mean((actual_values - predicted_values) ** 2)
    ax.text(0.02, 0.95, f"MSE: {mse:.2f}", transform=ax.transAxes, fontsize=12, verticalalignment='top')

    ax.legend(loc="upper right")

    return fig

def evaluate_mmm_with_multi_holdout(
    mmm,
    data: pd.DataFrame,
    target: str,
    date_col: str = "date",
    holdout_list=(30, 60, 90),
    include_last_observations_oos: bool = True,
):
    data = data.sort_values(date_col).reset_index(drop=True)

    def _fit_and_eval_for_holdout(h: int):
        if h <= 0:
            raise ValueError("holdout 값은 0보다 커야 합니다.")
        if h >= len(data):
            raise ValueError(f"holdout={h} 이(가) 데이터 길이 {len(data)} 이상입니다.")

        data_train = data.iloc[:-h].copy()
        data_test = data.iloc[-h:].copy()

        mmm_copy = copy.deepcopy(mmm)
        mmm_copy.build_model(data_train, data_train[target])
        mmm_copy.sample_prior_predictive(data_train, data_train[target], samples=2000)
        mmm_copy.fit(data_train, data_train[target])

        # ✅ diverging count 추가 (fit 이후에 가능)
        try:
            diverging = int(mmm_copy.idata["sample_stats"]["diverging"].sum().item())
        except Exception:
            diverging = np.nan  # idata 구조가 다르거나 없을 때 대비

        output_var = getattr(mmm_copy, "output_var", "y")

        # In-sample
        is_pred = mmm_copy.sample_posterior_predictive(
            X_pred=data_train,
            extend_idata=False,
            combined=False,
            include_last_observations=False,
            var_names=[output_var],
        )
        is_samples = is_pred[output_var]
        is_means = is_samples.mean(dim=["chain", "draw"], keep_attrs=True).data
        is_ci_lower = is_samples.quantile(0.05, dim=["chain", "draw"]).data
        is_ci_upper = is_samples.quantile(0.95, dim=["chain", "draw"]).data

        is_pred_df = pd.DataFrame({
            "date": data_train[date_col].values,
            "y_true": data_train[target].values,
            "y_pred": is_means.flatten(),
            "ci_lower": is_ci_lower.flatten(),
            "ci_upper": is_ci_upper.flatten(),
        })

        # Out-of-sample
        oos_pred = mmm_copy.sample_posterior_predictive(
            X_pred=data_test,
            extend_idata=False,
            combined=False,
            include_last_observations=include_last_observations_oos,
            var_names=[output_var],
        )
        oos_samples = oos_pred[output_var]
        oos_means = oos_samples.mean(dim=["chain", "draw"], keep_attrs=True).data
        oos_ci_lower = oos_samples.quantile(0.05, dim=["chain", "draw"]).data
        oos_ci_upper = oos_samples.quantile(0.95, dim=["chain", "draw"]).data

        oos_pred_df = pd.DataFrame({
            "date": data_test[date_col].values,
            "y_true": data_test[target].values,
            "y_pred": oos_means.flatten(),
            "ci_lower": oos_ci_lower.flatten(),
            "ci_upper": oos_ci_upper.flatten(),
        })

        def mape(y_true, y_pred):
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            mask = y_true != 0
            return (np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])).mean() * 100

        def r2(y_true, y_pred):
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

        is_arr = is_samples.values.reshape(-1, is_pred_df.shape[0]).T
        oos_arr = oos_samples.values.reshape(-1, oos_pred_df.shape[0]).T

        is_crps = float(np.mean(crps_ensemble(is_pred_df["y_true"].values, is_arr)))
        oos_crps = float(np.mean(crps_ensemble(oos_pred_df["y_true"].values, oos_arr)))

        metrics = {
            "train": {
                "R2": r2(is_pred_df["y_true"], is_pred_df["y_pred"]),
                "MAPE": mape(is_pred_df["y_true"], is_pred_df["y_pred"]),
                "CRPS": is_crps,
            },
            "test": {
                "R2": r2(oos_pred_df["y_true"], oos_pred_df["y_pred"]),
                "MAPE": mape(oos_pred_df["y_true"], oos_pred_df["y_pred"]),
                "CRPS": oos_crps,
            },
        }

        return {
            "metrics": metrics,
            "diverging": diverging,   # ✅ 여기로 반환
            "is_pred_df": is_pred_df,
            "oos_pred_df": oos_pred_df,
            "model": mmm_copy,
        }

    rows = []
    details = {}

    for h in holdout_list:
        res = _fit_and_eval_for_holdout(int(h))

        metrics = res["metrics"]
        diverging = res["diverging"]  # ✅ holdout별 diverging
        details[h] = res

        rows.append({
            "holdout": h,
            "dataset": "train",
            "R2": metrics["train"]["R2"],
            "MAPE": metrics["train"]["MAPE"],
            "CRPS": metrics["train"]["CRPS"],
            "diverging": diverging,   # ✅ 추가
        })
        rows.append({
            "holdout": h,
            "dataset": "test",
            "R2": metrics["test"]["R2"],
            "MAPE": metrics["test"]["MAPE"],
            "CRPS": metrics["test"]["CRPS"],
            "diverging": diverging,   # ✅ 추가
        })

    metrics_table = pd.DataFrame(rows)
    metrics_table = metrics_table.sort_values(["holdout", "dataset"]).reset_index(drop=True)

    return {
        "metrics_table": metrics_table,
        "details": details,
    }


def plot_waterfall_grouped_from_mmm(
    mmm,
    *,
    intercept_name: str = "intercept",
    seasonality_components: list[str] | None = None,
    baseline_control_components: list[str] | None = None,
    other_control_components: list[str] | None = None,
    marketing_components: list[str] | None = None,
    original_scale: bool = True,
    figsize: tuple[int, int] = (14, 7),
    intercept_label: str = "Intercept",
    seasonality_label: str = "Seasonality",
    baseline_control_label: str = "Baseline controls",
    other_control_label: str = "Other controls",
    marketing_label: str = "Marketing",
    debug: bool = False,
    **kwargs,
) -> plt.Figure:
    """
    Response Decomposition 결과를
    [Intercept] + [Seasonality] + [Baseline controls] + [Other controls] + [Marketing]
    구조로 그룹핑해서 waterfall bar로 시각화.

    ✔ trend / fourier 자동 seasonality 포함
    ✔ 그룹 덮어쓰기 방지 (우선순위 기반)
    ✔ categorical barh + ytick 꼬임 방지
    """

    # =========================================================
    # 1) Decomposition 결과 가져오기
    # =========================================================
    df = mmm.compute_mean_contributions_over_time(
        original_scale=original_scale
    )
    df = mmm._process_decomposition_components(data=df)

    df["contribution"] = df["contribution"].astype(float)

    # =========================================================
    # 2) group 컬럼 초기화
    # =========================================================
    df["group"] = "Others"

    # ---------------------------------------------------------
    # helper: 아직 그룹이 할당되지 않은 경우만 지정
    # ---------------------------------------------------------
    def assign_group(components, label):
        if components:
            mask = df["component"].isin(components) & (df["group"] == "Others")
            df.loc[mask, "group"] = label

    # =========================================================
    # 3) 자동 seasonality 감지 (trend / fourier)
    # =========================================================
    auto_seasonality_mask = (
        df["component"].str.contains("trend", case=False, na=False)
        | df["component"].str.contains("fourier", case=False, na=False)
    )
    df.loc[auto_seasonality_mask, "group"] = seasonality_label

    # =========================================================
    # 4) 명시적 그룹 할당 (우선순위)
    # =========================================================
    # 우선순위:
    # Marketing > Other controls > Baseline controls > Seasonality > Intercept

    assign_group(marketing_components, marketing_label)
    assign_group(other_control_components, other_control_label)
    assign_group(baseline_control_components, baseline_control_label)
    assign_group(seasonality_components, seasonality_label)

    # Intercept는 무조건 최우선
    df.loc[df["component"] == intercept_name, "group"] = intercept_label

    # =========================================================
    # 5) 디버깅 출력
    # =========================================================
    if debug:
        print("\n[DEBUG] component → group mapping")
        print(
            df[["component", "group"]]
            .drop_duplicates()
            .sort_values(["group", "component"])
            .to_string(index=False)
        )

    # =========================================================
    # 6) 그룹별 합산
    # =========================================================
    df_grouped = (
        df.groupby("group", as_index=False)["contribution"]
        .sum()
    )

    desired_order = [
        intercept_label,
        seasonality_label,
        baseline_control_label,
        other_control_label,
        marketing_label,
    ]

    df_grouped["group"] = pd.Categorical(
        df_grouped["group"],
        categories=[g for g in desired_order if g in df_grouped["group"].unique()],
        ordered=True,
    )
    df_grouped = df_grouped.sort_values("group").reset_index(drop=True)

    # =========================================================
    # 7) 퍼센트 계산
    # =========================================================
    total_contribution = df_grouped["contribution"].sum()
    if total_contribution != 0:
        df_grouped["percentage"] = (
            df_grouped["contribution"] / total_contribution * 100
        )
    else:
        df_grouped["percentage"] = 0.0

    # =========================================================
    # 8) plot용 값 결정
    # =========================================================
    contrib_orig = df_grouped["contribution"].astype(float)

    if original_scale:
        contrib_plot = contrib_orig.copy()
        x_label = "Cumulative Contribution"
        use_percentage_xticks = True
    else:
        contrib_plot = np.sign(contrib_orig) * np.log1p(np.abs(contrib_orig))
        x_label = "Cumulative Contribution (signed log scale)"
        use_percentage_xticks = False

    df_grouped["contribution_plot"] = contrib_plot

    # =========================================================
    # 9) Waterfall 플롯
    # =========================================================
    fig, ax = plt.subplots(figsize=figsize, layout="constrained", **kwargs)

    cumulative = 0.0

    for idx, row in df_grouped.iterrows():
        orig_val = float(row["contribution"])
        plot_val = float(row["contribution_plot"])

        color = "C0" if orig_val >= 0 else "C3"

        bar_start = cumulative if plot_val >= 0 else cumulative + plot_val

        ax.barh(
            row["group"],
            plot_val,
            left=bar_start,
            color=color,
            alpha=0.55,
        )

        if plot_val > 0:
            cumulative += plot_val

        label_x = bar_start + plot_val / 2.0

        ax.text(
            label_x,
            row["group"],
            f"{orig_val:,.0f}\n({row['percentage']:.1f}%)",
            ha="center",
            va="center",
            fontsize=10,
        )

    # =========================================================
    # 10) 축 / 스타일
    # =========================================================
    ax.set_title("Grouped Response Decomposition Waterfall")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Component Groups")

    if use_percentage_xticks and total_contribution != 0:
        xticks = np.linspace(0, total_contribution, num=11)
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            [f"{x / total_contribution * 100:.0f}%" for x in xticks]
        )
    else:
        ax.ticklabel_format(style="plain", axis="x")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return fig

import gc

def evaluate_mmm_weekly_and_aggregate_multi_holdout(
    mmm,
    data: pd.DataFrame,
    target: str,
    date_col: str = "date",
    holdout_weeks_list=(20, 24, 28, 32, 36),
    weekly_eval_weeks: int = 6,                 # 6주 주별 정확도
    rolling: bool = True,                       # “어느 포인트에서든” 보려면 True
    step_weeks: int = 4,                        # rolling 이동 간격(4주=1개월)
    min_train_weeks: int = 52,                  # rolling 시작 최소 학습 길이
    include_last_observations_oos: bool = True, # adstock state 유지
    prior_ppc_samples: int = 0,                 # 필요하면 2000 등. 기본은 스킵(속도)
    store_pred_dfs: bool = False,               # ✅ 메모리 절약: 기본은 DF 저장 안 함
    max_crps_draws: int | None = None,          # ✅ CRPS 계산용 draw 다운샘플(예: 500)
    random_state: int = 0,
):
    """
    madmatics MMM 스타일로:
    - deepcopy → build_model → (optional) sample_prior_predictive → fit
    - sample_posterior_predictive로 OOS 샘플링
    - (A) 6주 주별: MAPE/R2/CRPS
    - (B) holdout 전체기간 합계: MAPE_sum/CRPS_sum (+ rolling이면 R2_sum 가능)

    Returns
    -------
    {
      "metrics_table": pd.DataFrame,   # long format
      "summary": pd.DataFrame,         # holdout별 집계(mean/median 등)
      "details": dict                  # holdout, cut_date별 상세(기본은 metrics만)
    }
    """

    # -----------------------
    # Preprocess
    # -----------------------
    data = data.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).reset_index(drop=True)

    rng = np.random.default_rng(random_state)

    # -----------------------
    # Metric helpers
    # -----------------------
    def mape(y_true, y_pred, eps=1e-8):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        denom = np.maximum(np.abs(y_true), eps)
        return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

    def r2(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return np.nan if ss_tot == 0 else float(1 - ss_res / ss_tot)

    def crps_sum_scalar(y_true_sum, y_samples_sum):
        """
        CRPS for scalar aggregate using ensemble CRPS.
        obs: (1,)
        forecasts: (1, S)
        """
        obs = np.array([float(y_true_sum)], dtype=float)
        fc  = np.asarray(y_samples_sum, dtype=float).reshape(1, -1)
        return float(np.mean(crps_ensemble(obs, fc)))

    def _downsample_draws(arr_chain_draw_time):
        """
        arr_chain_draw_time: np.ndarray shape (chain, draw, T)
        return: np.ndarray shape (chain, draw', T) if max_crps_draws set
        """
        if max_crps_draws is None:
            return arr_chain_draw_time

        c, d, t = arr_chain_draw_time.shape
        S = c * d
        if S <= max_crps_draws:
            return arr_chain_draw_time

        # flatten -> sample indices -> reshape back to (chain', draw', T) is messy
        # easier: flatten to (S,T) then take subset and return (S',T)
        flat = arr_chain_draw_time.reshape(S, t)
        idx = rng.choice(S, size=int(max_crps_draws), replace=False)
        flat_sub = flat[idx]  # (S',T)
        return flat_sub  # NOTE: returns (S',T) if downsampled

    # -----------------------
    # single split evaluator
    # -----------------------
    def _fit_and_eval(train_df: pd.DataFrame, test_df: pd.DataFrame, holdout_weeks: int, cut_date):
        mmm_copy = copy.deepcopy(mmm)
        try:
            # build + fit
            mmm_copy.build_model(train_df, train_df[target])
            if prior_ppc_samples and prior_ppc_samples > 0:
                mmm_copy.sample_prior_predictive(
                    train_df, train_df[target], samples=int(prior_ppc_samples)
                )
            mmm_copy.fit(train_df, train_df[target])

            output_var = getattr(mmm_copy, "output_var", "y")

            # posterior predictive (OOS)
            oos_pred = mmm_copy.sample_posterior_predictive(
                X_pred=test_df,
                extend_idata=False,
                combined=False,  # (chain, draw, date) 형태 기대
                include_last_observations=include_last_observations_oos,
                var_names=[output_var],
            )

            oos_samples = oos_pred[output_var]  # xarray DataArray (chain, draw, date)
            time_dim = oos_samples.dims[-1]     # 보통 "date"

            # point forecast & interval
            oos_mean = oos_samples.mean(dim=["chain", "draw"], keep_attrs=True).data
            oos_p05  = oos_samples.quantile(0.05, dim=["chain", "draw"]).data
            oos_p95  = oos_samples.quantile(0.95, dim=["chain", "draw"]).data

            oos_pred_df = pd.DataFrame({
                date_col: test_df[date_col].values,
                "y_true": test_df[target].values,
                "y_pred": oos_mean.flatten(),
                "ci_lower": oos_p05.flatten(),
                "ci_upper": oos_p95.flatten(),
            })

            # ---------- (A) full holdout 주별 MAPE/R2/CRPS ----------
            n_full = len(oos_pred_df)

            full_arr_oos = oos_samples.values  # (chain, draw, T)
            c, d, T = full_arr_oos.shape

            # CRPS needs (T, S)
            if max_crps_draws is None:
                arr_full = full_arr_oos.reshape(c * d, T).T  # (T, S)
            else:
                sub = _downsample_draws(full_arr_oos)
                if sub.ndim == 3:
                    arr_full = sub.reshape(-1, T).T
                else:
                    arr_full = sub.T

            full_crps = float(np.mean(crps_ensemble(oos_pred_df["y_true"].values, arr_full)))

            weekly_metrics = {
                "MAPE": mape(oos_pred_df["y_true"].values, oos_pred_df["y_pred"].values),
                "R2":   r2(oos_pred_df["y_true"].values, oos_pred_df["y_pred"].values),
                "CRPS": full_crps,
            }

            # ---------- (B) aggregate sum over full holdout ----------
            agg_true_sum = float(oos_pred_df["y_true"].sum())
            agg_pred_sum = float(oos_pred_df["y_pred"].sum())

            # sum per draw: (chain, draw, T) -> (S, T) -> (S,)
            full_arr = oos_samples.values  # (chain, draw, T)
            c, d, T = full_arr.shape
            flat_full = full_arr.reshape(c * d, T)

            if max_crps_draws is not None and (c * d) > max_crps_draws:
                idx = rng.choice(c * d, size=int(max_crps_draws), replace=False)
                flat_full = flat_full[idx]

            sum_samples = flat_full.sum(axis=1)  # (S,)

            agg_metrics = {
                "MAPE_sum": float(abs(agg_true_sum - agg_pred_sum) / max(abs(agg_true_sum), 1e-8) * 100),
                "CRPS_sum": crps_sum_scalar(agg_true_sum, sum_samples),
                "actual_sum": agg_true_sum,
                "pred_sum": agg_pred_sum,
            }

            out = {
                "cut_date": cut_date,
                "holdout_weeks": holdout_weeks,
                "weekly_metrics": weekly_metrics,
                "agg_metrics": agg_metrics,
            }

            if store_pred_dfs:
                out["weekly_pred_df"] = weekly_df
                out["oos_pred_df"] = oos_pred_df

            return out

        finally:
            # ✅ 항상 정리
            try:
                del mmm_copy
            except Exception:
                pass
            gc.collect()

    # -----------------------------
    # main loop
    # -----------------------------
    rows = []
    details = {}  # details[holdout_weeks][cut_date] = res (metrics only by default)

    for h in holdout_weeks_list:
        h = int(h)
        details[h] = {}

        if len(data) <= (min_train_weeks + h):
            print(f"[skip] holdout_weeks={h}: not enough data.")
            continue

        if not rolling:
            cut_positions = [len(data) - h]
        else:
            start_cut = int(min_train_weeks)
            end_cut = len(data) - h
            cut_positions = list(range(start_cut, end_cut + 1, int(step_weeks)))

        for cut in cut_positions:
            train_df = data.iloc[:cut].copy()
            test_df  = data.iloc[cut:cut + h].copy()
            cut_date = test_df[date_col].iloc[0]

            res = _fit_and_eval(train_df, test_df, h, cut_date)
            details[h][pd.Timestamp(cut_date)] = res

            # weekly metrics row
            rows.append({
                "holdout_weeks": h,
                "cut_date": cut_date,
                "eval_type": "weekly",
                "horizon": f"{h}w",
                "MAPE": res["weekly_metrics"]["MAPE"],
                "R2":   res["weekly_metrics"]["R2"],
                "CRPS": res["weekly_metrics"]["CRPS"],
                "MAPE_sum": np.nan,
                "CRPS_sum": np.nan,
                "actual_sum": np.nan,
                "pred_sum": np.nan,
            })

            # aggregate metrics row
            rows.append({
                "holdout_weeks": h,
                "cut_date": cut_date,
                "eval_type": "agg_sum",
                "horizon": f"{h}w",
                "MAPE": np.nan,
                "R2": np.nan,
                "CRPS": np.nan,
                "MAPE_sum": res["agg_metrics"]["MAPE_sum"],
                "CRPS_sum": res["agg_metrics"]["CRPS_sum"],
                "actual_sum": res["agg_metrics"]["actual_sum"],
                "pred_sum": res["agg_metrics"]["pred_sum"],
            })

    metrics_table = pd.DataFrame(rows)

    # rolling이면 합계 R2를 추가로 계산(holdout별 여러 cut_date가 있을 때)
    if rolling and not metrics_table.empty:
        add_rows = []
        agg_df = metrics_table[metrics_table["eval_type"] == "agg_sum"].copy()

        for h, g in agg_df.groupby("holdout_weeks"):
            gg = g.dropna(subset=["actual_sum", "pred_sum"])
            if len(gg) >= 3:
                r2_sum = r2(gg["actual_sum"].values, gg["pred_sum"].values)
            else:
                r2_sum = np.nan

            add_rows.append({
                "holdout_weeks": int(h),
                "cut_date": pd.NaT,
                "eval_type": "agg_sum",
                "horizon": f"{int(h)}w",
                "MAPE": np.nan,
                "R2": np.nan,
                "CRPS": np.nan,
                "MAPE_sum": np.nan,
                "CRPS_sum": np.nan,
                "actual_sum": np.nan,
                "pred_sum": np.nan,
                "R2_sum_over_windows": r2_sum,
            })

        metrics_table = pd.concat([metrics_table, pd.DataFrame(add_rows)], ignore_index=True)

    # summary
    summary = (
        metrics_table
        .groupby(["holdout_weeks", "eval_type", "horizon"], dropna=False)
        .agg(
            count=("holdout_weeks", "count"),
            MAPE_mean=("MAPE", "mean"),
            MAPE_median=("MAPE", "median"),
            R2_mean=("R2", "mean"),
            R2_median=("R2", "median"),
            CRPS_mean=("CRPS", "mean"),
            CRPS_median=("CRPS", "median"),
            MAPE_sum_mean=("MAPE_sum", "mean"),
            MAPE_sum_median=("MAPE_sum", "median"),
            CRPS_sum_mean=("CRPS_sum", "mean"),
            CRPS_sum_median=("CRPS_sum", "median"),
        )
        .reset_index()
        .sort_values(["holdout_weeks", "eval_type"])
    )

    return {
        "metrics_table": metrics_table.sort_values(["holdout_weeks", "cut_date", "eval_type"]).reset_index(drop=True),
        "summary": summary,
        "details": details,
    }

def _knee_index_by_max_dist(x: np.ndarray, y: np.ndarray) -> int:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        return int(np.argmax(y))

    x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)

    p1 = np.array([x_n[0], y_n[0]])
    p2 = np.array([x_n[-1], y_n[-1]])
    v = p2 - p1
    v_norm = np.linalg.norm(v) + 1e-12

    pts = np.column_stack([x_n, y_n])
    dist = np.abs(np.cross(v, pts - p1)) / v_norm

    return int(np.argmax(dist[1:-1]) + 1)

def _get_channel_contribution_grid(mmm, *, start, stop, num):
    """
    MMM 버전 차이를 흡수하는 공용 wrapper
    (singular / plural 메서드 모두 대응)
    """
    if hasattr(mmm, "get_channel_contribution_forward_pass_grid"):
        return mmm.get_channel_contribution_forward_pass_grid(
            start=start, stop=stop, num=num
        )

    if hasattr(mmm, "get_channel_contributions_forward_pass_grid"):
        return mmm.get_channel_contributions_forward_pass_grid(
            start=start, stop=stop, num=num
        )

    raise AttributeError(
        "MMM has no channel contribution grid method. "
        "Expected one of: "
        "get_channel_contribution_forward_pass_grid / "
        "get_channel_contributions_forward_pass_grid"
    )

def get_channel_contribution_knee_df(
    mmm,
    start: float,
    stop: float,
    num: int,
    absolute_xrange: bool = False,
    hdi_prob: float = 0.94,
) -> pd.DataFrame:
    share_grid = np.linspace(start=start, stop=stop, num=num)

    contributions = _get_channel_contribution_grid(
        mmm, start=start, stop=stop, num=num
    )

    rows = []

    for channel in mmm.channel_columns:
        # (chain, draw, date, grid, ...) -> date 합쳐서 total curve
        channel_total = contributions.sel(channel=channel).sum(dim="date")

        # ✅ grid dim 찾기: chain/draw 제외한 나머지 1개가 grid여야 함
        non_sample_dims = [d for d in channel_total.dims if d not in ("chain", "draw")]
        if len(non_sample_dims) != 1:
            raise ValueError(
                f"[{channel}] grid dim을 특정할 수 없음. dims={channel_total.dims}, "
                f"non_sample_dims={non_sample_dims}"
            )
        grid_dim = non_sample_dims[0]

        # ✅ 평균 curve: 길이는 num(=12) 이어야 정상
        y_mean = channel_total.mean(dim=("chain", "draw")).values  # shape: (num,)

        # x축
        total_channel_input = float(mmm.X[channel].sum())
        x_range = total_channel_input * share_grid if absolute_xrange else share_grid

        # ✅ knee index (x,y 길이 일치)
        k_idx = _knee_index_by_max_dist(x_range, y_mean)

        # ✅ HDI curve (grid별로 2개 lower/upper)
        hdi = az.hdi(channel_total, hdi_prob=hdi_prob)  # dims: (grid_dim, hdi)
        # ArviZ 버전에 따라 hdi dim 이름이 달라질 수 있어서 안전하게 values로
        hdi_vals = hdi["x"].values  # shape: (num, 2)

        rows.append({
            "channel": channel,
            "grid_dim": grid_dim,
            "knee_idx": int(k_idx),
            "knee_share": float(share_grid[k_idx]),
            "knee_x": float(x_range[k_idx]),
            "knee_contribution_mean": float(y_mean[k_idx]),
            "knee_hdi_low": float(hdi_vals[k_idx, 0]),
            "knee_hdi_high": float(hdi_vals[k_idx, 1]),
            "current_total_input": total_channel_input,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("knee_contribution_mean", ascending=False)
        .reset_index(drop=True)
    )


import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import warnings

def granger_best_lag(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    max_lag: int = 12,
    *,
    criterion: str = "min_pvalue",   # "min_pvalue" | "first_sig"
    alpha: float = 0.05,
    test: str = "ssr_ftest",         # "ssr_ftest" | "ssr_chi2test" | "lrtest" | "params_ftest"
    add_const: bool = True,
    dropna: bool = True,
    verbose: bool = False,
) -> tuple[int, pd.DataFrame]:
    """
    Find best lag for Granger causality: does x_col Granger-cause y_col?

    Parameters
    ----------
    df : pd.DataFrame
        Time-ordered data.
    x_col : str
        Cause candidate (X).
    y_col : str
        Target variable (Y).
    max_lag : int
        Maximum lag to test.
    criterion : str
        - "min_pvalue": pick lag with smallest p-value
        - "first_sig": pick smallest lag with p < alpha (if none, fallback to min_pvalue)
    alpha : float
        Significance threshold used for "first_sig".
    test : str
        Which test statistic to use from statsmodels output.
    add_const : bool
        Whether to include constant in test regression.
    dropna : bool
        Drop rows with NaNs in x/y.
    verbose : bool
        Print statsmodels internal output.

    Returns
    -------
    best_lag : int
        Selected lag.
    summary_df : pd.DataFrame
        Columns: lag, p_value, f_stat(or chi2), df_denom, df_num
    """
    if criterion not in ("min_pvalue", "first_sig"):
        raise ValueError("criterion must be 'min_pvalue' or 'first_sig'")

    if test not in ("ssr_ftest", "ssr_chi2test", "lrtest", "params_ftest"):
        raise ValueError("test must be one of: ssr_ftest, ssr_chi2test, lrtest, params_ftest")

    data = df[[y_col, x_col]].copy()
    if dropna:
        data = data.dropna()

    if len(data) < max_lag + 5:
        raise ValueError(
            f"Not enough rows ({len(data)}) for max_lag={max_lag}. "
            "Try reducing max_lag or provide more data."
        )

    # statsmodels expects array with columns [y, x]
    results = grangercausalitytests(
        data.values,
        maxlag=max_lag,
        addconst=add_const,
        verbose=verbose,
    )

    rows = []
    for lag, res in results.items():
        test_stats = res[0][test]  # tuple like (stat, pvalue, df_denom, df_num) for F-tests
        stat = float(test_stats[0])
        pval = float(test_stats[1])

        # For chi2 / lr tests df fields can differ; best effort:
        df_denom = test_stats[2] if len(test_stats) > 2 else np.nan
        df_num = test_stats[3] if len(test_stats) > 3 else np.nan

        rows.append({
            "lag": int(lag),
            "stat": stat,
            "p_value": pval,
            "df_denom": df_denom,
            "df_num": df_num,
            "test": test,
        })

    summary_df = pd.DataFrame(rows).sort_values("lag").reset_index(drop=True)

    # Choose best lag
    if criterion == "first_sig":
        sig = summary_df[summary_df["p_value"] < alpha]
        if len(sig) > 0:
            best_lag = int(sig.iloc[0]["lag"])
        else:
            best_lag = int(summary_df.loc[summary_df["p_value"].idxmin(), "lag"])
    else:  # min_pvalue
        best_lag = int(summary_df.loc[summary_df["p_value"].idxmin(), "lag"])

    return best_lag, summary_df

def granger_best_lag_many(
    df: pd.DataFrame,
    x_cols: list[str],
    y_col: str,
    max_lag: int = 12,
    **kwargs
) -> pd.DataFrame:
    rows = []
    for x in x_cols:
        best_lag, summary = granger_best_lag(df, x, y_col, max_lag=max_lag, **kwargs)
        rows.append({
            "x_col": x,
            "y_col": y_col,
            "best_lag": best_lag,
            "best_p_value": float(summary.loc[summary["lag"] == best_lag, "p_value"].iloc[0]),
            # "min_p_value": float(summary["p_value"].min()),
        })
    return pd.DataFrame(rows).sort_values(["best_p_value"]).reset_index(drop=True)


import pandas as pd
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz

# ============================================================
# Endpoint 기호 정리 (가독성)
# ============================================================
def _endpoint_symbol(ep):
    s = str(ep)
    if "ARROW" in s:  
        return "->"
    if "TAIL" in s:   
        return "-"
    if "CIRCLE" in s:
        return "o"
    return s

# ============================================================
# GeneralGraph / CausalGraph 모두 처리
# ============================================================
def edges_from_graph_any(cg, pretty_names=None):
    # cg.G가 있으면 사용(CausalGraph), 없으면 그대로 사용(GeneralGraph)
    G = getattr(cg, "G", cg)

    nodes = G.get_nodes()
    # 표시용 이름 매핑
    if pretty_names and len(pretty_names) == len(nodes):
        name_map = {nodes[i].get_name(): pretty_names[i] for i in range(len(nodes))}
    else:
        name_map = {node.get_name(): node.get_name() for node in nodes}

    edges, seen = [], set()
    for a in nodes:
        for b in nodes:
            if a is b:
                continue

            edge = G.get_edge(a, b)  # Node 객체 기반 조회
            if edge is None:
                continue

            ep = edge.get_proximal_endpoint(b)
            key = (a.get_name(), b.get_name(), str(ep))
            if key in seen:
                continue

            seen.add(key)
            edges.append((name_map[a.get_name()], name_map[b.get_name()], _endpoint_symbol(ep)))

    return edges


# ============================================================
# 💡 PC / FCI 인과 탐색 함수
# ============================================================
def causal_discovery(df, columns, alpha=0.05):
    """
    df: pandas DataFrame
    columns: 사용할 col 리스트(channel_cols + control_cols + [target])
    alpha: 유의수준
    
    return: (pc_edges, fci_edges)
    """
    data = df[columns].to_numpy()
    names = columns

    # ---- PC ----
    pc_graph = pc(data, alpha=alpha, indep_test_func=fisherz)
    edges_pc = edges_from_graph_any(pc_graph, names)

    # ---- FCI ----
    fci_ret = fci(data, alpha=alpha, indep_test_func=fisherz)
    fci_graph = fci_ret[0] if isinstance(fci_ret, tuple) else fci_ret
    edges_fci = edges_from_graph_any(fci_graph, names)

    return edges_pc, edges_fci