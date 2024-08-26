use std::{
    rc::Rc,
    sync::{Arc, RwLock},
};

use itertools::Itertools;

use crate::{
    game::Game,
    timestep::{TimeStamp, TimeStep},
};

#[derive(Clone, Debug)]
pub struct PlayerInfo<P> {
    pub player: P,
    pub timesteps: Vec<Arc<RwLock<TimeStep<P>>>>,
    w2: f64,
    virtual_games: u32,
}
impl<P> PlayerInfo<P>
where
    P: Eq + PartialEq + Copy + std::fmt::Display,
{
    pub fn new(player: P, w2: f64, virtual_games: u32) -> Self {
        Self {
            player,
            w2: w2 * (10f64.ln() / 400f64).powf(2f64), // Converts from elo to whr
            timesteps: vec![],
            virtual_games,
        }
    }

    /// Returns the player's rating at each timestep as an Elo value
    pub fn get_ratings(&self) -> impl Iterator<Item = (TimeStamp, f64)> + '_ {
        self.timesteps.iter().map(|t| {
            let t = t.read().unwrap();
            (t.time, t.elo_rating())
        })
    }

    pub fn get_or_insert_timestep(&mut self, time: TimeStamp) -> Arc<RwLock<TimeStep<P>>> {
        match self
            .timesteps
            .binary_search_by_key(&time, |t| t.read().unwrap().time)
        {
            Ok(i) => self.timesteps[i].clone(),
            Err(i) => {
                log::info!("Created timestep {time} for player {}", self.player);
                let mut t = TimeStep::new(self.player, time, self.virtual_games);
                if i == 0 {
                    log::info!(
                        "Timestep {time} is the first day for player {}",
                        self.player
                    );
                    t.is_first_day = true;
                    t.set_gamma_rating(1.0);
                    if let Some(t) = self.timesteps.first_mut() {
                        t.write().unwrap().is_first_day = false;
                    }
                }
                if i >= 1 {
                    if let Some(t_prev) = self.timesteps.get(i - 1) {
                        t.rating = t_prev.read().unwrap().rating;
                    }
                }
                self.timesteps.insert(i, Arc::new(RwLock::new(t)));
                self.timesteps[i].clone()
            }
        }
    }

    /// Adds one game to this player's history.
    pub fn add_game(&mut self, game: Rc<Game<P>>) {
        // Timesteps are kept sorted for practical purposes
        self.get_or_insert_timestep(game.time)
            .write()
            .unwrap()
            .add_game(game);
    }

    /// Refines the player's rating, corresponds to one iteration of Newton's method.
    pub fn refine_ratings(&mut self) {
        for t in &self.timesteps {
            let mut t = t.write().unwrap();
            t.clear_normalizing_terms_cache();
            t.compute_normalizing_terms();
        }

        log::info!(
            "Player {} ratings before refinement:\n{:?}",
            self.player,
            self.get_ratings().collect::<Vec<_>>()
        );

        if self.timesteps.len() == 1 {
            // 1D Newton method
            let mut t = self.timesteps[0].write().unwrap();
            t.rating -= t.dlog_likelihood() / t.dlog2_likelihood();
        } else if self.timesteps.len() > 1 {
            // ND Newton method
            let normal_variance = self.normal_variance();
            let hessian = self.hessian_matrix(&normal_variance);
            let gradient = self.gradient(&normal_variance);

            let steps = self.timesteps.len();

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
            for (t, x) in self.timesteps.iter().zip(x) {
                let mut t = t.write().unwrap();
                t.rating -= x;
            }
        }
    }

    /// Computes the normal variance for this player.
    pub fn normal_variance(&self) -> Vec<f64> {
        self.timesteps
            .iter()
            .map(|t| t.read().unwrap().time)
            .tuple_windows()
            .map(|(t1, t2)| t2.abs_diff(t1) as f64 * self.w2)
            .collect()
    }

    /// Computes the Hessian matrix for this player.
    pub fn hessian_matrix(&self, normal_variance: &[f64]) -> Vec<f64> {
        let steps = self.timesteps.len();
        let mut hessian = vec![0f64; steps * steps];
        for row in 0..steps {
            for col in 0..steps {
                hessian[row * steps + col] = if row == col {
                    let mut prior = 0.0;
                    if row < steps - 1 {
                        prior += -1.0 / normal_variance[row]
                    }
                    if row > 0 {
                        prior += -1.0 / normal_variance[row - 1]
                    }
                    self.timesteps[row].read().unwrap().dlog2_likelihood() + prior - 0.001
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
    pub fn gradient(&self, normal_variance: &[f64]) -> Vec<f64> {
        (0..self.timesteps.len())
            .map(|i| {
                let mut prior = 0.0;
                if i < self.timesteps.len() - 1 {
                    prior += (self.timesteps[i + 1].read().unwrap().rating
                        - self.timesteps[i].read().unwrap().rating)
                        / normal_variance[i]
                }
                if i > 0 {
                    prior += (self.timesteps[i - 1].read().unwrap().rating
                        - self.timesteps[i].read().unwrap().rating)
                        / normal_variance[i - 1]
                }
                self.timesteps[i].read().unwrap().dlog_likelihood() + prior
            })
            .collect()
    }
}
