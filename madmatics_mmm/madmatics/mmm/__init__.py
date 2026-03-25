#   Copyright 2022 - 2025 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Marketing Mix Models (MMM)."""

from madmatics.mmm import base, mmm_ico, preprocessing, validating
from madmatics.mmm.base import BaseValidateMMM, MMMModelBuilder
from madmatics.mmm.components.adstock import (
    AdstockTransformation,
    DelayedAdstock,
    GeometricAdstock,
    WeibullCDFAdstock,
    WeibullPDFAdstock,
    adstock_from_dict,
)
from madmatics.mmm.components.saturation import (
    HillSaturation,
    HillSaturationSigmoid,
    InverseScaledLogisticSaturation,
    LogisticSaturation,
    MichaelisMentenSaturation,
    RootSaturation,
    SaturationTransformation,
    TanhSaturation,
    TanhSaturationBaselined,
    saturation_from_dict,
)
from madmatics.mmm.fourier import MonthlyFourier, WeeklyFourier, YearlyFourier
from madmatics.mmm.hsgp import (
    HSGP,
    CovFunc,
    HSGPPeriodic,
    PeriodicCovFunc,
    SoftPlusHSGP,
    approx_hsgp_hyperparams,
    create_complexity_penalizing_prior,
    create_constrained_inverse_gamma_prior,
    create_eta_prior,
    create_m_and_L_recommendations,
)
from madmatics.mmm.linear_trend import LinearTrend
from madmatics.mmm.media_transformation import (
    MediaConfig,
    MediaConfigList,
    MediaTransformation,
)
from madmatics.mmm.mmm import MMM as MMM
from madmatics.mmm.mmm_ico import MMM as MMMICO
from madmatics.mmm.mmm_ich import MMM as MMMICH
from madmatics.mmm.mmm_is import MMM as MMMIS
from madmatics.mmm.mmm_chs import MMM as MMMCHS
from madmatics.mmm.mmm_coch import MMM as MMMCOCH
from madmatics.mmm.mmm_cos import MMM as MMMCOS
from madmatics.mmm.mmm_ico_trend import MMM as MMMICOT
from madmatics.mmm.mmm_ich_trend import MMM as MMMICHT
from madmatics.mmm.mmm_control import MMM as MMMC
from madmatics.mmm.mmm_ico_whales import MMM as MMMW
from madmatics.mmm.preprocessing import (
    preprocessing_method_X,
    preprocessing_method_y,
)
from madmatics.mmm.validating import validation_method_X, validation_method_y

__all__ = [
    "HSGP",
    "MMM",
    "AdstockTransformation",
    "BaseValidateMMM",
    "CovFunc",
    "DelayedAdstock",
    "GeometricAdstock",
    "HSGPPeriodic",
    "HillSaturation",
    "HillSaturationSigmoid",
    "InverseScaledLogisticSaturation",
    "LinearTrend",
    "LogisticSaturation",
    "MMMModelBuilder",
    "MediaConfig",
    "MediaConfigList",
    "MediaTransformation",
    "MichaelisMentenSaturation",
    "MonthlyFourier",
    "PeriodicCovFunc",
    "RootSaturation",
    "SaturationTransformation",
    "SoftPlusHSGP",
    "TanhSaturation",
    "TanhSaturationBaselined",
    "WeeklyFourier",
    "WeibullCDFAdstock",
    "WeibullPDFAdstock",
    "YearlyFourier",
    "adstock_from_dict",
    "approx_hsgp_hyperparams",
    "base",
    "create_complexity_penalizing_prior",
    "create_constrained_inverse_gamma_prior",
    "create_eta_prior",
    "create_m_and_L_recommendations",
    "mmm",
    "mmmc",
    "mmm_ico",
    "mmm_ich",
    "mmm_is",
    "mmm_chs",
    "mmm_coch",
    "mmm_cos",
    "mmm_ico_trend",
    "mmm_ico_wales",
    "preprocessing",
    "preprocessing_method_X",
    "preprocessing_method_y",
    "saturation_from_dict",
    "validating",
    "validation_method_X",
    "validation_method_y",
]
