//! # Whole-History Rating

mod game;
mod player_info;
#[cfg(test)]
mod test;
mod timestep;

use std::{
    collections::HashMap,
    num::{NonZeroU16, NonZeroU32},
    rc::Rc,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};

use game::Game;
use player_info::PlayerInfo;
use timestep::TimeStamp;

/// Builder API for [`Whr`], with easy ways to add input data and set the number of
/// iterations for refinement.
#[derive(Clone)]
pub struct WhrBuilder<P> {
    players: HashMap<P, Arc<RwLock<PlayerInfo<P>>>>,
    games: Vec<Rc<Game<P>>>,

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
            players: HashMap::new(),
            games: vec![],

            iterations: None,
            epsilon: 1e-3,
            max_duration: None,
            batch_size: unsafe { NonZeroU32::new_unchecked(10) },
            w2: 300f64,
            virtual_games: 2,
        }
    }
}
impl<P> WhrBuilder<P>
where
    P: std::hash::Hash + Eq + Copy + std::fmt::Debug + std::fmt::Display,
{
    /// Creates a new default builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Uses the currently inputted parameters to compute [`Whr`] ratings.
    pub fn build(mut self) -> HashMap<P, Arc<RwLock<PlayerInfo<P>>>> {
        let mut iterations = 0;
        let start = Instant::now();
        loop {
            for _ in 0..self.batch_size.get() {
                for player in self.players.values_mut() {
                    player.write().unwrap().refine_ratings()
                }
            }

            // Stop conditions
            if let Some(max_iters) = self.iterations {
                iterations += 1;
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
        self.players
    }

    /// Adds a game record to the builder.
    pub fn with_game(
        mut self,
        p1: P,
        p2: P,
        winner: Option<P>,
        time: TimeStamp,
        handicap: Option<f64>,
    ) -> Self {
        log::info!("Added game between {p1} and {p2} (winner: {winner:?}) at time {time}");
        assert_ne!(p1, p2, "Players cannot be the same for both sides");
        if !self.players.contains_key(&p1) {
            log::info!("Added player {p1} to the database");
            self.players.insert(
                p1,
                Arc::new(RwLock::new(PlayerInfo::new(
                    p1,
                    self.w2,
                    self.virtual_games,
                ))),
            );
        }
        if !self.players.contains_key(&p2) {
            log::info!("Added player {p2} to the database");
            self.players.insert(
                p2,
                Arc::new(RwLock::new(PlayerInfo::new(
                    p2,
                    self.w2,
                    self.virtual_games,
                ))),
            );
        }

        let mut p1_info = self.players.get(&p1).unwrap().write().unwrap();
        let mut p2_info = self.players.get(&p2).unwrap().write().unwrap();
        let p1_timestep = p1_info.get_or_insert_timestep(time);
        let p2_timestep = p2_info.get_or_insert_timestep(time);

        let game = Rc::new(Game {
            time,
            p1,
            p2,
            p1_timestep,
            p2_timestep,
            winner,
            p2_advantage: handicap.unwrap_or(0f64),
        });

        p1_info.add_game(game.clone());
        p2_info.add_game(game.clone());
        self.games.push(game.clone());

        drop(p1_info);
        drop(p2_info);
        self
    }

    /// Adds multiple games to the builder at once.
    pub fn with_games(mut self) -> Self {
        todo!()
    }

    /// Sets the number of iterations. By default, the algorithm iterates until
    /// a given precision (1e-3 by default) is reached.
    ///
    /// Note that passing a value of 0 is the same as indicating that the algorithm
    /// should keep the default behavior.
    pub fn with_iterations(mut self, iterations: u32) -> Self {
        self.iterations = NonZeroU32::new(iterations);
        log::info!(
            "Set iterations count to {}",
            if let Some(i) = self.iterations {
                i.get().to_string()
            } else {
                "infinite".to_string()
            }
        );
        self
    }

    /// Sets the precision at which the algorithm should consider ratings to be
    /// stabilized. By default, this value is 1e-3.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        log::info!("Set expected stability to {epsilon}");
        self
    }

    /// Specifies a maximum duration for the algorithm to run.
    pub fn with_maximum_duration(mut self, duration: Duration) -> Self {
        self.max_duration = Some(duration);
        log::info!("Set maximum duration to {:?}", self.max_duration);
        self
    }

    /// Specifies how many iterations to perform as a batch before checking for
    /// stop conditions such as time or convergence. By default, checks are performed
    /// after every batch of 10 iterations.
    ///
    /// Note that passing a batch size of 0 will set the batch size to 1.
    pub fn with_batch_size(mut self, size: u32) -> Self {
        self.batch_size = NonZeroU32::new(size).unwrap_or(unsafe { NonZeroU32::new_unchecked(1) });
        log::info!("Set batch size to {:?}", self.batch_size);
        self
    }

    /// Sets the `w2` parameter, responsible for the variability of ratings over
    /// time.
    pub fn with_w2(mut self, w2: f64) -> Self {
        self.w2 = w2;
        log::info!("Set w2 to {}", self.w2);
        self
    }
}
