#![doc = include_str!("../readme.md")]

mod game;
pub mod rating;
#[cfg(test)]
mod test;
mod timestep;

use std::{
    collections::HashMap,
    num::NonZeroU32,
    time::{Duration, Instant},
};

use game::InnerMatchRecord;
pub use game::{MatchRecord, MatchRecordError};
use rating::{EloRating, GammaRating, Rating};
use timestep::TimeStep;

/// WHR ratings for players, with additional API to access probability of winning
/// and other useful information.
#[derive(Clone)]
pub struct Whr<P> {
    ratings: HashMap<P, Vec<Rating>>,
}
impl<P> Whr<P>
where
    P: std::hash::Hash + Eq + Clone,
{
    /// Returns a player's ratings over timesteps.
    ///
    /// If the player in question has never played, return `None`.
    pub fn get_player_ratings(&self, player: &P) -> Option<&[Rating]> {
        self.ratings.get(player).map(|r| r.as_slice())
    }

    /// Returns a player's rating at a specific timestep.
    pub fn rating(&self, player: &P, time: usize) -> Option<Rating> {
        self.ratings
            .get(player)?
            .binary_search_by_key(&time, |r| r.timestep)
            .ok()
            .map(|i| self.ratings.get(player).unwrap()[i])
    }

    /// Computes the probability of winning for `p1` against `p2` at a given timestep.
    ///
    /// Returns `None` if any of the players are not registered.
    pub fn probability_of_winning(&self, p1: &P, p2: &P, time: usize) -> Option<f64> {
        let p1_rating = self.rating(p1, time)?.gamma();
        let p2_rating = self.rating(p2, time)?.gamma();
        Some(p1_rating / (p1_rating + p2_rating))
    }
}

/// Builder API for [`Whr`], allowing you to incrementally add match records and
/// compute ratings.
///
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use whr::{MatchRecord, WhrBuilder};
///
/// let mut whr = WhrBuilder::default();
/// whr
///   // Register matches, with:
///   // - two named players,
///   // - an optional winner,
///   // - a timestep,
///   // - and optional handicap (first player advantage)
///   .add_match(MatchRecord::new("alice", "bob", Some("bob"), 1, None)?)
///   .add_match(MatchRecord::new("alice", "bob", None, 2, None)?)
///   .add_match(MatchRecord::new("bob", "alice", Some("alice"), 2, None)?)
///
///   // You can even add multiple games at once from an iterator
///   .add_matches([
///     MatchRecord::new("bob", "alice", Some("alice"), 1, None)?,
///     MatchRecord::new("alice", "charlie", Some("charlie"), 4, None)?
///   ]);
/// // Build ratings up to this point
/// let ratings = whr.build();
///
/// // Then add a new match and recompute the ratings.
/// whr.add_match(MatchRecord::new("charlie", "bob", None, 2, None)?);
/// let new_ratings = whr.build();
/// # Ok(()) }
/// ```
///
/// `WhrBuilder` is generic the player's identification so that you can use whatever
/// your infrastructure uses: strings, integers, etc.
///
/// Timesteps are discrete values indicating an order for the matches. This may be matched
/// with tournaments, for example. They need to map to unsigned integers. One way
/// to do this for dates (which is the most common example) is to map them to the
/// number of days between this date and a fixed start date.
///
/// By default, the rating values are refined until their mean change is within
/// some error rate epsilon, which is `1e-3` by default. This behavior (and other
/// parameters) can be configured using the same builder pattern.
#[derive(Clone)]
pub struct WhrBuilder<P> {
    // Mapping from players to their identifiers.
    player_index: HashMap<P, usize>,

    // List of matches recorded..
    matches: Vec<InnerMatchRecord>,
    // Information about each player at each timestep.
    timesteps: Vec<Vec<TimeStep>>,
    ratings: Vec<Vec<Rating>>,

    // Parameters for the algorithms to compute ratings.
    iterations: Option<NonZeroU32>,
    epsilon: f64,
    max_duration: Option<Duration>,
    batch_size: NonZeroU32,
    w2: f64,
    virtual_games: u32,
}
impl<P> Default for WhrBuilder<P> {
    fn default() -> Self {
        Self {
            player_index: HashMap::new(),
            matches: vec![],
            timesteps: vec![],
            ratings: vec![],

            iterations: None,
            epsilon: 1e-3,
            max_duration: None,
            batch_size: unsafe { NonZeroU32::new_unchecked(10) },
            w2: 300f64 * (10f64.ln() / 400f64).powf(2f64),
            virtual_games: 2,
        }
    }
}
impl<P> WhrBuilder<P>
where
    P: std::hash::Hash + Eq + Clone,
{
    /// Creates a new default builder with no recorded matches or players.
    pub fn new() -> Self {
        Self::default()
    }

    /// Uses the current parameters to compute ratings.
    ///
    /// This does not consume the builder, so that the match records can be updated
    /// incrementally.
    pub fn build(&mut self) -> Whr<P> {
        let mut iterations = 0;
        let start = Instant::now();
        'refine: loop {
            for _ in 0..self.batch_size.get() {
                let delta = self.refine_ratings();
                if delta <= self.epsilon {
                    break 'refine;
                }
                iterations += 1;
            }

            // Stop conditions
            if let Some(max_iters) = self.iterations {
                if iterations >= max_iters.get() {
                    break;
                }
            }
            if let Some(max_duration) = self.max_duration {
                if start.elapsed() >= max_duration {
                    break;
                }
            }
        }
        self.update_uncertainety();
        Whr {
            ratings: HashMap::from_iter(
                self.player_index
                    .iter()
                    .map(|(p, &i)| (p.clone(), self.ratings[i].clone())),
            ),
        }
    }

    /// Adds a [`MatchRecord`] to the history.
    pub fn add_match(&mut self, game: MatchRecord<P>) -> &mut Self {
        let (p1, p2) = game.players();
        // Record players
        let p1_index = self.register_player(p1.clone());
        let p2_index = self.register_player(p2.clone());
        let winner_index = game.winner().map(|p| self.get_player_index(&p));

        // Record game
        let game_index = self.matches.len();
        self.matches.push(InnerMatchRecord {
            p1: p1_index,
            p2: p2_index,
            winner: winner_index,
            handicap: game.handicap().unwrap_or(EloRating(0f64)),
        });

        if let Some(winner_index) = winner_index {
            self.get_timestep(winner_index, game.timestep())
                .won_games
                .push(game_index);

            let loser_index = if winner_index == p1_index {
                p2_index
            } else {
                p1_index
            };
            self.get_timestep(loser_index, game.timestep())
                .lost_games
                .push(game_index);
        } else {
            self.get_timestep(p1_index, game.timestep())
                .drawn_games
                .push(game_index);
            self.get_timestep(p2_index, game.timestep())
                .drawn_games
                .push(game_index);
        }

        self
    }

    /// Adds multiple matches to the history at once.
    pub fn add_matches(&mut self, matches: impl IntoIterator<Item = MatchRecord<P>>) -> &mut Self {
        for m in matches {
            self.add_match(m);
        }
        self
    }

    /// Sets the number of iterations. By default, the algorithm iterates until
    /// a given precision (`1e-3` by default) is reached.
    ///
    /// If a value of 0 is passed, the algorithm considers that it can run an for
    /// an unlimited amount of iterations.
    pub fn set_iterations(&mut self, iterations: u32) -> &mut Self {
        self.iterations = NonZeroU32::new(iterations);
        self
    }

    /// Sets the error margin under which the algorithm should consider ratings to be
    /// stabilized. By default, this value is `1e-3`.
    pub fn set_epsilon(&mut self, epsilon: f64) -> &mut Self {
        self.epsilon = epsilon;
        self
    }

    /// Specifies a maximum duration for the algorithm to run.
    pub fn set_maximum_duration(&mut self, duration: Duration) -> &mut Self {
        self.max_duration = Some(duration);
        self
    }

    /// Specifies how many iterations to perform as a batch before checking for
    /// stop conditions such as time or convergence. By default, checks are performed
    /// after every batch of 10 iterations.
    ///
    /// If a value of 0 is passed, the batch size is 1 (no batching).
    pub fn set_batch_size(&mut self, size: u32) -> &mut Self {
        self.batch_size = NonZeroU32::new(size).unwrap_or(NonZeroU32::new(1).unwrap());
        self
    }

    /// Sets the `w2` parameter, responsible for the variability of ratings over
    /// time. A higher value means that ratings will fluctuate more.
    pub fn set_w2(&mut self, w2: f64) -> &mut Self {
        self.w2 = w2 * (10f64.ln() / 400f64).powf(2f64); // Converts from elo to whr
        self
    }

    /// Sets the number of virtual games to initialize ratings.
    ///
    /// Every player's initial rating is decided as if they played `2 * virtual_games`
    /// matches against an opponent of rating 0, with `virtual_games` wins and `virtual_games`
    /// draws.
    pub fn set_virtual_games(&mut self, virtual_games: u32) -> &mut Self {
        self.virtual_games = virtual_games;
        self
    }

    // HELPER FUNCTIONS

    /// Registers a player in the playerbase, returning its corresponding index.
    /// If the player was already registered, returns its index.
    fn register_player(&mut self, player: P) -> usize {
        if let Some(&i) = self.player_index.get(&player) {
            i
        } else {
            debug_assert_eq!(self.player_index.len(), self.ratings.len());
            debug_assert_eq!(self.player_index.len(), self.timesteps.len());
            let player_index = self.player_index.len();
            self.player_index.insert(player, player_index);
            self.ratings.push(vec![]);
            self.timesteps.push(vec![]);
            player_index
        }
    }

    /// Returns the index of a given player, which is known to be registered.
    ///
    /// # Panic
    /// This function panics if called with an unregistered player.
    fn get_player_index(&self, player: &P) -> usize {
        self.player_index[player]
    }

    /// Gets or inserts a new timestep for the given player.
    fn get_timestep(&mut self, player: usize, time: usize) -> &mut TimeStep {
        let i = match self.timesteps[player].binary_search_by_key(&time, |t| t.timestep) {
            Ok(i) => i,
            Err(i) => {
                self.timesteps[player].insert(i, TimeStep::new(time));
                let rating = Rating::new(
                    time,
                    if i == 0 {
                        GammaRating(1f64).into()
                    } else {
                        self.ratings[player][i - 1].rating
                    },
                );
                self.ratings[player].insert(i, rating);
                i
            }
        };
        &mut self.timesteps[player][i]
    }

    /// Iterator over all timesteps for a given player.
    fn get_timesteps(&self, player: usize) -> impl Iterator<Item = usize> + '_ {
        self.ratings[player].iter().map(|r| r.timestep)
    }

    /// Number of timesteps for a given player.
    fn get_timestep_count(&self, player: usize) -> usize {
        self.ratings[player].len()
    }

    /// Computes the normalizing terms for all sets of games/players.
    fn compute_normalizing_terms(&mut self, player: usize) {
        let adjusted_gamma_rating = |game: usize, player: usize, time: usize| {
            let game = self.matches[game];
            let opponent = game.opponent(player);
            let result = self.ratings[opponent]
                .binary_search_by_key(&time, |t| t.timestep)
                .map(|i| self.ratings[opponent][i].elo())
                .unwrap()
                + game.handicap(player).0;

            10f64.powf(result / 400f64)
        };

        // Account for virtual games
        for (i, timestep) in self.timesteps[player].iter_mut().enumerate() {
            let time = timestep.timestep;

            timestep.won_game_terms.clear();
            for &game in &timestep.won_games {
                let opponent_gamma = adjusted_gamma_rating(game, player, time);
                timestep
                    .won_game_terms
                    .push([1f64, 0f64, 1f64, opponent_gamma])
            }
            timestep.lost_game_terms.clear();
            for &game in &timestep.lost_games {
                let opponent_gamma = adjusted_gamma_rating(game, player, time);
                timestep
                    .lost_game_terms
                    .push([0f64, opponent_gamma, 1f64, opponent_gamma])
            }
            timestep.drawn_game_terms.clear();
            for &game in &timestep.drawn_games {
                let opponent_gamma = adjusted_gamma_rating(game, player, time);
                timestep
                    .drawn_game_terms
                    .push([0.5f64, 0.5 * opponent_gamma, 1f64, opponent_gamma])
            }
            if i == 0 {
                for _ in 0..self.virtual_games {
                    timestep.drawn_game_terms.push([0.5, 0.5, 1.0, 1.0])
                }
            }
        }
    }

    /// Computes the normal variance for a given player.
    fn normal_variance(&self, player: usize) -> Vec<f64> {
        let mut normal_variance = vec![];
        let mut t1 = None;
        for t2 in self.get_timesteps(player) {
            if let Some(t1) = t1 {
                normal_variance.push(t2.abs_diff(t1) as f64 * self.w2);
            }
            t1 = Some(t2);
        }
        normal_variance
    }

    /// Computes the Hessian matrix for a given player.
    fn hessian_matrix(&self, player: usize, normal_variance: &[f64]) -> Vec<f64> {
        let steps = self.get_timestep_count(player);
        let mut hessian = vec![0f64; steps * steps];
        for (row, time) in self.get_timesteps(player).enumerate() {
            for col in 0..steps {
                hessian[row * steps + col] = if row == col {
                    let mut prior = 0.0;
                    if row < steps - 1 {
                        prior += -1.0 / normal_variance[row]
                    }
                    if row > 0 {
                        prior += -1.0 / normal_variance[row - 1]
                    }
                    self.timestep_dlog2_likelihood(player, time) + prior - 0.001
                } else if col >= 1 && row == col - 1 {
                    1.0 / normal_variance[row]
                } else if row == col + 1 {
                    1.0 / normal_variance[col]
                } else {
                    0.0
                }
            }
        }
        hessian
    }

    /// Computes the gradient for each day.
    fn gradient(&self, player: usize, normal_variance: &[f64]) -> Vec<f64> {
        let ratings = &self.ratings[player];
        let steps = self.get_timestep_count(player);

        self.get_timesteps(player)
            .enumerate()
            .map(|(i, time)| {
                let mut prior = 0.0;
                if i < steps - 1 {
                    prior += (ratings[i + 1].rating.0 - ratings[i].rating.0) / normal_variance[i]
                }
                if i > 0 {
                    prior +=
                        (ratings[i - 1].rating.0 - ratings[i].rating.0) / normal_variance[i - 1]
                }
                self.timestep_dlog_likelihood(player, time) + prior
            })
            .collect()
    }

    /// Computes the log likelihood's derivative for this timestamp.
    fn timestep_dlog_likelihood(&self, player: usize, time: usize) -> f64 {
        let mut sum = 0f64;
        let (timestep, rating) = self.get_player_information(player, time);
        let gamma = rating.gamma();
        for terms in [
            &timestep.won_game_terms,
            &timestep.lost_game_terms,
            &timestep.drawn_game_terms,
        ] {
            for [_, _, c, d] in terms {
                sum += c / (c * gamma + d)
            }
        }
        timestep.won_game_terms.len() as f64 + (0.5 * timestep.drawn_game_terms.len() as f64)
            - (gamma * sum)
    }

    /// Computes the log likelihood's second derivative for this timestamp.
    fn timestep_dlog2_likelihood(&self, player: usize, time: usize) -> f64 {
        let mut sum = 0f64;
        let (timestep, rating) = self.get_player_information(player, time);
        let gamma = rating.gamma();
        for terms in [
            &timestep.won_game_terms,
            &timestep.lost_game_terms,
            &timestep.drawn_game_terms,
        ] {
            for [_, _, c, d] in terms {
                sum += (c * d) / (c * gamma + d).powf(2f64)
            }
        }
        -gamma * sum
    }

    /// Returns the timestep and rating of a given player at a certain time.
    fn get_player_information(&self, player: usize, time: usize) -> (&TimeStep, &Rating) {
        let i = self.ratings[player]
            .binary_search_by_key(&time, |t| t.timestep)
            .unwrap();
        (&self.timesteps[player][i], &self.ratings[player][i])
    }

    /// Refines the players' ratings, corresponds to one iteration of Newton's method.
    /// Returns the delta between the updated ratings and previous ones.
    fn refine_ratings(&mut self) -> f64 {
        let players = self.player_index.values().cloned().collect::<Vec<_>>();
        let mut delta = 0f64;
        let mut diffs = 1;
        for player in players {
            self.compute_normalizing_terms(player);

            let ratings_len = self.get_timestep_count(player);
            if ratings_len == 1 {
                // 1D Newton method
                let step = self.get_timesteps(player).next().unwrap();
                let dlog = self.timestep_dlog_likelihood(player, step);
                let dlog2 = self.timestep_dlog2_likelihood(player, step);
                self.ratings[player][0].rating.0 -= dlog / dlog2;
            } else if self.timesteps.len() > 1 {
                // ND Newton method
                let normal_variance = self.normal_variance(player);
                let hessian = self.hessian_matrix(player, &normal_variance);
                let gradient = self.gradient(player, &normal_variance);

                let steps = self.get_timestep_count(player);

                let mut a = vec![0f64; steps];
                let mut d = vec![0f64; steps];
                d[0] = hessian[0];
                let mut b = vec![0f64; steps];
                b[0] = hessian[1];

                for i in 1..steps {
                    a[i] = hessian[i * steps + i - 1] / d[i - 1];
                    d[i] = hessian[i * steps + i] - a[i] * b[i - 1];
                    if i < steps - 1 {
                        b[i] = hessian[i * steps + i + 1];
                    }
                }

                let mut y = vec![0f64; steps];
                y[0] = gradient[0];
                for i in 1..steps {
                    y[i] = gradient[i] - a[i] * y[i - 1]
                }

                let mut x = vec![0.0; steps];
                x[steps - 1] = y[steps - 1] / d[steps - 1];
                for i in (0..(steps - 1)).rev() {
                    x[i] = (y[i] - b[i] * x[i + 1]) / d[i]
                }

                // Update ratings
                for (rating, diff) in self.ratings[player].iter_mut().zip(x) {
                    rating.rating.0 -= diff;
                    delta += (diff.abs() - delta).abs() / diffs as f64;
                    diffs += 1;
                }
            }
        }
        delta
    }

    /// Updates the uncertainety of ratings for each player.
    fn update_uncertainety(&mut self) {
        let players = self.player_index.values().cloned().collect::<Vec<_>>();
        for player in players {
            let steps = self.get_timestep_count(player);
            if steps == 0 {
                continue;
            }

            let normal_variance = self.normal_variance(player);
            let hessian = self.hessian_matrix(player, &normal_variance);

            let mut a = vec![0f64; steps];
            let mut b = vec![0f64; steps];
            let mut d = vec![0f64; steps];
            d[0] = hessian[0];
            if steps > 1 {
                b[0] = hessian[1];
            }

            for i in 1..steps {
                a[i] = hessian[i * steps + i - 1] / d[i - 1];
                d[i] = hessian[i * steps + 1] - a[i] * b[i - 1];
                if i < steps - 1 {
                    b[i] = hessian[i * steps + i + 1];
                }
            }

            let mut ap = vec![0f64; steps];
            let mut bp = vec![0f64; steps];
            let mut dp = vec![0f64; steps];
            let mut variance = vec![0f64; steps];
            if steps > 1 {
                dp[steps - 1] = hessian[steps * steps - 1];
                bp[steps - 1] = hessian[steps * steps - 2];
                for i in (0..steps - 1).rev() {
                    ap[i] = hessian[i * steps + i + 1] / dp[i + 1];
                    dp[i] = hessian[i * steps + i] - ap[i] * bp[i + 1];
                    if i > 0 {
                        bp[i] = hessian[i * steps + i - 1];
                    }
                }
                for i in 0..steps - 1 {
                    variance[i] = dp[i + 1] / (b[i] * bp[i + 1] - d[i] * dp[i + 1]);
                }
                variance[steps - 1] = -1f64 / d[steps - 1];
            }

            let mut covariance = vec![0f64; steps * steps];
            for row in 0..steps {
                for col in 0..steps {
                    if row == col {
                        covariance[row * steps + col] = variance[row];
                    } else if col != 0 && row == col - 1 {
                        covariance[row * steps + col] = -a[col] * variance[col];
                    }
                }
            }

            for (i, rating) in self.ratings[player].iter_mut().enumerate() {
                rating.uncertainety = covariance[i * steps + i]
            }
        }
    }
}
