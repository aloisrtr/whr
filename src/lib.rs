//! # Whole-History Rating

//! Implementation of Rémi Coulom's [Whole History Rating (WHR) algorithm](https://www.remi-coulom.fr/WHR/)
//! as a Rust library.
//!
//! It notably supports:
//! - handicaps (first player advantage)
//! - draws
//!
//! WHR is used to rate players in competitive games, akin to the [Elo](https://en.wikipedia.org/wiki/Elo_rating_system)
//! or [TrueSkill](https://en.wikipedia.org/wiki/TrueSkill) systems. It can estimate
//! the probability of winning between two players even if they have never competed
//! against one another. It is more accurate than say the Elo system, at the cost
//! of requiring more computation.
//!
//! WHR is notably used in Go, Warcraft 3, Renju, and is even used in some sports!
//!
//! ## Future work
//! I hope to generalize the library further, notably by supporting:
//! - games with more than two players
//! - teams
//!
//! as was stated to be possible in Rémi Coulom's paper introducing WHR.
//!
//! ## Installation
//! The library can be used in any Cargo project by running:
//! ```sh
//! cargo add whr
//! ```
//! or by adding the following to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! whr = "0.1"
//! ```
//! ## Exemple usage
//! The library works following the [builder pattern](https://rust-unofficial.github.io/patterns/patterns/creational/builder.html).
//! ```rust
//! use whr::WhrBuilder;
//!
//! let whr = WhrBuilder::default()
//!   .with_iterations(50) // Maximum number of iterations for the algorithm to converge.
//!   .with_epsilon(1e-5) // Target stability between iterations
//!   // Register games, with:
//!   // - two named players,
//!   // - an optional winner,
//!   // - a timestep,
//!   // - and optional handicap (first player advantage)
//!   .with_game("alice", "bob", Some("bob"), 1, None)
//!   .with_game("alice", "bob", None, 2, None)
//!   .with_game("bob", "alice", Some("alice"), 2, None)
//!   .build();
//! ```
//! This returns a [`Whr`] object with ratings for each player by timestep.
mod game;
mod rating;
#[cfg(test)]
mod test;
mod timestep;

use std::{
    collections::HashMap,
    num::NonZeroU32,
    time::{Duration, Instant},
};

use game::Game;
use itertools::Itertools;
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
    /// Returns the player's ratings.
    pub fn get_player_ratings(&self, player: &P) -> Option<&[Rating]> {
        self.ratings.get(player).map(|r| r.as_slice())
    }

    /// Returns the rating of a given player at a specific time if the player
    /// played any game at this timestep.
    pub fn rating(&self, player: &P, time: usize) -> Option<Rating> {
        self.ratings
            .get(player)?
            .binary_search_by_key(&time, |r| r.timestep)
            .ok()
            .map(|i| self.ratings.get(player).unwrap()[i])
    }

    /// Computes the probability of winning for `p1` against `p2` at a given timestep.
    /// Returns `None` if any of the players are not registered.
    pub fn probability_of_winning(&self, p1: &P, p2: &P, time: usize) -> Option<f64> {
        let p1_rating = self.rating(p1, time)?.gamma();
        let p2_rating = self.rating(p2, time)?.gamma();
        Some(p1_rating / (p1_rating + p2_rating))
    }
}

/// Builder API for [`Whr`], with easy ways to add input data and set the number of
/// iterations for refinement.
#[derive(Clone)]
pub struct WhrBuilder<P> {
    // Mapping from players to identifiers
    player_index: HashMap<P, usize>,

    // Mapping from games to identifiers
    games: Vec<Game>,
    // Timesteps information for each player
    timesteps: Vec<Vec<TimeStep>>,
    ratings: Vec<Vec<Rating>>,

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
            games: vec![],
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
    /// Creates a new default builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Uses the currently inputted parameters to compute ratings.
    /// This does not consume the builder.
    pub fn build(mut self) -> Whr<P> {
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

    /// Adds a game record to the builder.
    pub fn with_game(
        mut self,
        p1: P,
        p2: P,
        winner: Option<P>,
        time: usize,
        handicap: Option<EloRating>,
    ) -> Self {
        assert!(p1 != p2, "Players cannot be the same for both sides");
        assert!(
            winner
                .as_ref()
                .map(|p| p == &p1 || p == &p2)
                .unwrap_or(true),
            "Winner must be one of the players",
        );

        // Record players
        let p1_index = self.register_player(p1);
        let p2_index = self.register_player(p2);
        let winner_index = winner.map(|p| self.get_player_index(&p));

        // Record game
        let game_index = self.games.len();
        self.games.push(Game {
            p1: p1_index,
            p2: p2_index,
            winner: winner_index,
            handicap: handicap.unwrap_or(EloRating(0f64)),
        });

        if let Some(winner_index) = winner_index {
            self.get_timestep(winner_index, time)
                .won_games
                .push(game_index);

            let loser_index = if winner_index == p1_index {
                p2_index
            } else {
                p1_index
            };
            self.get_timestep(loser_index, time)
                .lost_games
                .push(game_index);
        } else {
            self.get_timestep(p1_index, time)
                .drawn_games
                .push(game_index);
            self.get_timestep(p2_index, time)
                .drawn_games
                .push(game_index);
        }

        self
    }

    /// Adds multiple games to the builder at once.
    pub fn with_games(
        mut self,
        games: impl Iterator<Item = (P, P, Option<P>, usize, Option<EloRating>)>,
    ) -> Self {
        for (p1, p2, winner, time, handicap) in games {
            self = self.with_game(p1, p2, winner, time, handicap)
        }
        self
    }

    /// Sets the number of iterations. By default, the algorithm iterates until
    /// a given precision (1e-3 by default) is reached.
    ///
    /// Note that passing a value of 0 is the same as indicating that the algorithm
    /// should keep the default behavior.
    pub fn with_iterations(mut self, iterations: u32) -> Self {
        self.iterations = NonZeroU32::new(iterations);
        self
    }

    /// Sets the precision at which the algorithm should consider ratings to be
    /// stabilized. By default, this value is 1e-3.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Specifies a maximum duration for the algorithm to run.
    pub fn with_maximum_duration(mut self, duration: Duration) -> Self {
        self.max_duration = Some(duration);
        self
    }

    /// Specifies how many iterations to perform as a batch before checking for
    /// stop conditions such as time or convergence. By default, checks are performed
    /// after every batch of 10 iterations.
    ///
    /// Note that passing a batch size of 0 will set the batch size to 1.
    pub fn with_batch_size(mut self, size: u32) -> Self {
        self.batch_size = NonZeroU32::new(size).unwrap_or(unsafe { NonZeroU32::new_unchecked(1) });
        self
    }

    /// Sets the `w2` parameter, responsible for the variability of ratings over
    /// time.
    pub fn with_w2(mut self, w2: f64) -> Self {
        self.w2 = w2 * (10f64.ln() / 400f64).powf(2f64); // Converts from elo to whr
        self
    }

    /// Sets the number of virtual games to stabilize ratings.
    pub fn with_virtual_games(mut self, virtual_games: u32) -> Self {
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
    pub fn compute_normalizing_terms(&mut self, player: usize) {
        let adjusted_gamma_rating = |game: usize, player: usize, time: usize| {
            let game = self.games[game];
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
        self.get_timesteps(player)
            .tuple_windows()
            .map(|(t1, t2)| t2.abs_diff(t1) as f64 * self.w2)
            .collect()
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
                let dlog = self.timestep_dlog_likelihood(player, 0);
                let dlog2 = self.timestep_dlog2_likelihood(player, 0);
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
            dp[steps - 1] = hessian[steps * steps - 1];
            bp[steps - 1] = hessian[steps * steps - 2];
            for i in (0..steps - 1).rev() {
                ap[i] = hessian[i * steps + i + 1] / dp[i + 1];
                dp[i] = hessian[i * steps + i] - ap[i] * bp[i + 1];
                if i > 0 {
                    bp[i] = hessian[i * steps + i - 1];
                }
            }
            let mut variance = vec![0f64; steps];
            for i in 0..steps - 1 {
                variance[i] = dp[i + 1] / (b[i] * bp[i + 1] - d[i] * dp[i + 1]);
            }
            variance[steps - 1] = -1f64 / d[steps - 1];

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
