from Projects.SIL.src.Services.DataObject import *

class MultiParameters():
    alpha = None
    beta = None
    delta = None
    gamma = None
    epsilon = None
    stigma = None
    zeta = None
    eta = None
    theta = None
    iota = None
    kappa = None
    # lambda
    mu = None
    nu = None
    xi = None
    omicro = None
    # pi
    koppa = None
    rho = None
    sigma = None
    tau = None
    upsilon = None
    phi = None
    chi = None
    psi = None
    omega = None
    sampi = None

    def __init__(self, datatype, metadata=None):
        """
        Depends on the type of datatype (temperature, luminosity, ...) this will create a self object
        with different features
        Args:
            datatype: a type of "DataType" class.
            metadata: metadata to operate with values
        """

        if datatype == DataTypes().TEMPERATURE:
            # TODO (@gabvaztor) Assign/Determine values
            pass