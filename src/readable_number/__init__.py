"""
More Readable Number

A module to make sure the accuracy when calculating,
and also show the mathematical formula by using a more readable
form.
"""

from . import constants
from .basic_class import OrderMode
from .fraction import Fraction, LatexMode
from .integer import Integer
from .monomial import Monomial
from .multinomial import Multinomial
from .power import Power
from .unknown_num import UnknownNum, UnknownShowMode

__version__ = "0.0.1"

# Some useful unknows
x = UnknownNum("x")
y = UnknownNum("y")
z = UnknownNum("z")
e = UnknownNum("e")

alpha = UnknownNum("alpha", string="α", latex=r" \alpha ")
beta = UnknownNum("beta", string="β", latex=r" \beta ")
gamma = UnknownNum("gamma", string="γ", latex=r" \gamma ")
delta = UnknownNum("delta", string="δ", latex=r" \delta ")
epsilon = UnknownNum("epsilon", string="ε", latex=r" \epsilon ")
zeta = UnknownNum("zeta", string="ζ", latex=r" \zeta ")
eta = UnknownNum("eta", string="η", latex=r" \eta ")
theta = UnknownNum("theta", string="θ", latex=r" \theta ")
iota = UnknownNum("iota", string="ι", latex=r" \iota ")
kappa = UnknownNum("kappa", string="κ", latex=r" \kappa ")
lambda_ = UnknownNum("lambda", string="λ", latex=r" \lambda ")
mu = UnknownNum("mu", string="μ", latex=r" \mu ")
nu = UnknownNum("nu", string="ν", latex=r" \nu ")
xi = UnknownNum("xi", string="ξ", latex=r" \xi ")
omicron = UnknownNum("omicron", string="ο", latex=r" \omicron ")
pi = UnknownNum("pi", string="π", latex=r" \pi ")
rho = UnknownNum("rho", string="ρ", latex=r" \rho ")
sigma = UnknownNum("sigma", string="σ", latex=r" \sigma ")
tau = UnknownNum("tau", string="τ", latex=r" \tau ")
upsilon = UnknownNum("upsilon", string="υ", latex=r" \upsilon ")
phi = UnknownNum("phi", string="φ", latex=r" \phi ")
chi = UnknownNum("chi", string="χ", latex=r" \chi ")
psi = UnknownNum("psi", string="ψ", latex=r" \psi ")
omega = UnknownNum("omega", string="ω", latex=r" \omega ")

__all__ = [
    "constants",
    "Fraction",
    "Integer",
    "LatexMode",
    "Monomial",
    "OrderMode",
    "Multinomial",
    "Power",
    "UnknownNum",
    "UnknownShowMode",
    # Some useful unknows
    "x",
    "y",
    "z",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda_",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
]
