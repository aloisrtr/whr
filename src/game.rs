use crate::rating::EloRating;

/// Information about a match between two players.
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct MatchRecord<P> {
    player1: P,
    player2: P,
    winner: Option<P>,
    timestep: usize,
    handicap: Option<EloRating>,
}
impl<P> MatchRecord<P>
where
    P: Eq,
{
    /// Creates a new match record.
    ///
    /// # Errors
    /// This function will return an error if `player1` and `player2` are the same,
    /// or `winner` is not one of `player1` or `player2`.
    pub fn new(
        player1: P,
        player2: P,
        winner: Option<P>,
        timestep: usize,
        handicap: Option<EloRating>,
    ) -> Result<Self, MatchRecordError> {
        if player1 == player2 {
            return Err(MatchRecordError::PlayingYourself);
        }

        if winner
            .as_ref()
            .map(|w| w != &player1 && w != &player2)
            .unwrap_or(false)
        {
            return Err(MatchRecordError::WinnerDidNotPlay);
        }

        Ok(Self {
            player1,
            player2,
            winner,
            timestep,
            handicap,
        })
    }

    /// Returns the two players of this match.
    pub fn players(&self) -> (&P, &P) {
        (&self.player1, &self.player2)
    }

    /// Returns this match's winner, or `None` if it was a draw.
    pub fn winner(&self) -> Option<&P> {
        self.winner.as_ref()
    }

    /// Returns the timestep this game occured on.
    pub fn timestep(&self) -> usize {
        self.timestep
    }

    pub(crate) fn handicap(&self) -> Option<EloRating> {
        self.handicap
    }
}

/// Errors that may happen when creating a new [`MatchRecord`].
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum MatchRecordError {
    /// Raised when the indicated winner is not one of the players.
    WinnerDidNotPlay,
    /// Raised when both players are the same.
    PlayingYourself,
}
impl std::fmt::Display for MatchRecordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::WinnerDidNotPlay => "The winner is not one of the players",
                Self::PlayingYourself => "Both players are the same",
            }
        )
    }
}
impl std::error::Error for MatchRecordError {}

/// Inner match records, using integers instead of a generic player identification
/// to allow for faster accessing of matrices and such.
#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub struct InnerMatchRecord {
    pub p1: usize,
    pub p2: usize,
    pub winner: Option<usize>,
    pub handicap: EloRating,
}
impl InnerMatchRecord {
    pub fn opponent(&self, player: usize) -> usize {
        if player == self.p1 {
            self.p2
        } else {
            self.p1
        }
    }

    pub fn handicap(&self, player: usize) -> EloRating {
        if player == self.p1 {
            -self.handicap
        } else {
            self.handicap
        }
    }
}
