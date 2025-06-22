//! # Ratings
//! Various rating values used by this library.
//!
//! The main one you should care about is [`Rating`], as it represents the
//! values of Whole-History Rating with information about the timestep, rating value
//! and uncertainety.
//!
//! The rest are here for convenience/conversions.

/// Characterizes the rating of a player at a specific timestep.
///
/// This includes the [`WhrRating`] in itself, as well as the current uncertainety
/// regarding that rating.
#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub struct Rating {
    pub(crate) timestep: usize,
    pub(crate) rating: WhrRating,
    pub(crate) uncertainety: f64,
}
impl Rating {
    /// Creates a new rating given its initial value and timestep.
    pub(crate) fn new(timestep: usize, initial: WhrRating) -> Self {
        Self {
            timestep,
            rating: initial,
            uncertainety: 0f64,
        }
    }

    /// Returns the natural value of this rating.
    pub fn natural(&self) -> f64 {
        self.rating.0
    }

    /// Returns the equivalent Elo rating.
    pub fn elo(&self) -> f64 {
        EloRating::from(self.rating).0
    }

    /// Returns the equivalent gamma rating.
    pub fn gamma(&self) -> f64 {
        GammaRating::from(self.rating).0
    }

    /// Returns the uncertainty associated with this rating.
    pub fn uncertainty(&self) -> f64 {
        self.uncertainety
    }
}

/// Represents a gamma rating value.
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct GammaRating(pub(crate) f64);
impl GammaRating {
    pub fn new(value: f64) -> Self {
        Self(value)
    }

    /// Returns this values as a float.
    pub fn inner(&self) -> f64 {
        self.0
    }
}
impl From<f64> for GammaRating {
    fn from(value: f64) -> Self {
        Self(value)
    }
}
impl From<WhrRating> for GammaRating {
    fn from(value: WhrRating) -> Self {
        Self(value.0.exp())
    }
}
impl From<EloRating> for GammaRating {
    fn from(value: EloRating) -> Self {
        let whr = WhrRating::from(value);
        Self::from(whr)
    }
}
impl std::ops::Add for GammaRating {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}
impl std::ops::Sub for GammaRating {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}
impl std::ops::Mul for GammaRating {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}
impl std::ops::Div for GammaRating {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}
impl std::ops::Neg for GammaRating {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

/// Represents a Whole-History Rating value.
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct WhrRating(pub(crate) f64);
impl WhrRating {
    pub fn new(value: f64) -> Self {
        Self(value)
    }

    /// Returns this values as a float.
    pub fn inner(&self) -> f64 {
        self.0
    }
}
impl From<f64> for WhrRating {
    fn from(value: f64) -> Self {
        Self(value)
    }
}
impl From<GammaRating> for WhrRating {
    fn from(value: GammaRating) -> Self {
        Self(value.0.ln())
    }
}
impl From<EloRating> for WhrRating {
    fn from(value: EloRating) -> Self {
        Self(value.0 * (10f64.ln() / 400f64))
    }
}
impl std::ops::Add for WhrRating {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}
impl std::ops::Sub for WhrRating {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}
impl std::ops::Mul for WhrRating {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}
impl std::ops::Div for WhrRating {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}
impl std::ops::Neg for WhrRating {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

/// Represents an Elo rating value.
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct EloRating(pub(crate) f64);
impl EloRating {
    pub fn new(value: f64) -> Self {
        Self(value)
    }

    /// Returns this values as a float.
    pub fn inner(&self) -> f64 {
        self.0
    }
}
impl From<f64> for EloRating {
    fn from(value: f64) -> Self {
        Self(value)
    }
}
impl From<WhrRating> for EloRating {
    fn from(value: WhrRating) -> Self {
        Self(value.0 * (400f64 / 10f64.ln()))
    }
}
impl From<GammaRating> for EloRating {
    fn from(value: GammaRating) -> Self {
        let whr = WhrRating::from(value);
        Self::from(whr)
    }
}
impl std::ops::Add for EloRating {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}
impl std::ops::Sub for EloRating {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}
impl std::ops::Mul for EloRating {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}
impl std::ops::Div for EloRating {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}
impl std::ops::Neg for EloRating {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}
