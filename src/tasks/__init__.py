from . import common
from .downstream import InferenceTask
# from .discriminative import MetricTask
from .generative import GenerationTask, NoveltyTask
from .structure import UnsupervisedMetrics, ModularityExplicitness, SAP, IRS, DCI, MIG, FactorVAE, BetaVAE, Fairness
from . import generalization