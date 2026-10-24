#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    InvalidParameter,
    InvalidDimension,
    NonFiniteInput,
    InvalidTimeStep,
    InvalidCovariance,
    SingularInnovation,
    CapacityExceeded,
    NumericalFailure,
    BudgetExceeded,
    UnsupportedConfiguration,
}

impl Error {
    pub const fn message(self) -> &'static str {
        match self {
            Self::InvalidParameter => "invalid parameter",
            Self::InvalidDimension => "invalid dimension",
            Self::NonFiniteInput => "non-finite input",
            Self::InvalidTimeStep => "invalid time step",
            Self::InvalidCovariance => "invalid covariance",
            Self::SingularInnovation => "singular innovation covariance",
            Self::CapacityExceeded => "capacity exceeded",
            Self::NumericalFailure => "numerical failure",
            Self::BudgetExceeded => "budget exceeded",
            Self::UnsupportedConfiguration => "unsupported configuration",
        }
    }
}
