use std::rc::Rc;

use crate::game::Game;

pub type TimeStamp = u32;
pub type Rating = f64;

#[derive(Clone, Debug)]
pub struct TimeStep<P> {
    pub time: TimeStamp,
    player: P,
    pub is_first_day: bool,
    pub rating: f64,
    uncertainety: f64,

    game_terms_are_init: bool,
    virtual_games: u32,
    won_games: Vec<Rc<Game<P>>>,
    won_game_terms: Vec<[f64; 4]>,
    lost_games: Vec<Rc<Game<P>>>,
    lost_game_terms: Vec<[f64; 4]>,
    drawn_games: Vec<Rc<Game<P>>>,
    drawn_game_terms: Vec<[f64; 4]>,
}
impl<P> TimeStep<P>
where
    P: Eq + PartialEq + std::fmt::Display,
{
    /// Creates a new [`TimeStep`] with no games played.
    pub fn new(player: P, time: TimeStamp, virtual_games: u32) -> Self {
        Self {
            time,
            player,
            is_first_day: false,
            rating: 0.0,
            uncertainety: 0.0,

            game_terms_are_init: false,
            virtual_games,
            won_games: vec![],
            won_game_terms: vec![],
            lost_games: vec![],
            lost_game_terms: vec![],
            drawn_games: vec![],
            drawn_game_terms: vec![],
        }
    }

    /// Adds a game to this timesteps's record.
    pub fn add_game(&mut self, game: Rc<Game<P>>) {
        match &game.winner {
            Some(p) => {
                if p == &self.player {
                    self.won_games.push(game)
                } else {
                    self.lost_games.push(game)
                }
            }
            None => self.drawn_games.push(game),
        }
    }

    /// Sets the gamma rating for this timestep.
    pub fn set_gamma_rating(&mut self, gamma: Rating) {
        self.rating = gamma.ln();
    }
    /// Returns the gamma rating of this timestep.
    pub fn gamma_rating(&self) -> Rating {
        self.rating.exp()
    }

    /// Sets the elo rating for this timestep.
    pub fn set_elo_rating(&mut self, elo: Rating) {
        self.rating = elo * (10f64.ln() / 400f64);
    }
    /// Returns the elo rating of this timestep.
    pub fn elo_rating(&self) -> Rating {
        self.rating * (400f64 / 10f64.ln())
    }

    /// Clears the normalizing terms' cache.
    pub fn clear_normalizing_terms_cache(&mut self) {
        self.game_terms_are_init = false;
        self.won_game_terms.clear();
        self.lost_game_terms.clear();
        self.drawn_game_terms.clear();
    }

    /// Computes the normalizing terms for all sets of games.
    pub fn compute_normalizing_terms(&mut self) {
        if self.game_terms_are_init {
            return;
        }

        log::info!(
            "Computing game terms for player {} at time {}",
            self.player,
            self.time
        );
        for game in &self.won_games {
            let gamma = game.opponent_adjusted_gamma_rating(&self.player);
            self.won_game_terms.push([1f64, 0f64, 1f64, gamma])
        }
        for game in &self.lost_games {
            let gamma = game.opponent_adjusted_gamma_rating(&self.player);
            self.lost_game_terms.push([0f64, gamma, 1f64, gamma])
        }
        for game in &self.drawn_games {
            let gamma = game.opponent_adjusted_gamma_rating(&self.player);
            self.drawn_game_terms
                .push([0.5f64, 0.5 * gamma, 1f64, gamma])
        }
        if self.is_first_day {
            for _ in 0..self.virtual_games {
                self.drawn_game_terms.push([0.5, 0.5, 1.0, 1.0])
            }
        }
    }

    /// Computes the log likelihood for this timestamp.
    pub fn log_likelihood(&self) -> f64 {
        let mut sum = 0f64;
        let gamma = self.gamma_rating();
        for [a, _, c, d] in &self.won_game_terms {
            sum += (a * gamma).ln();
            sum -= (c * gamma + d).ln();
        }
        for [_, b, c, d] in &self.lost_game_terms {
            sum += b.ln();
            sum -= (c * gamma + d).ln();
        }
        for [a, b, c, d] in &self.drawn_game_terms {
            sum += (a * 2f64 * gamma).ln() * 0.5;
            sum += (b * 2f64).ln() * 0.5;
            sum -= (c * gamma + d).ln()
        }
        sum
    }

    /// Computes the log likelihood's derivative for this timestamp.
    pub fn dlog_likelihood(&self) -> f64 {
        let mut sum = 0f64;
        let gamma = self.gamma_rating();
        for terms in [
            &self.won_game_terms,
            &self.lost_game_terms,
            &self.drawn_game_terms,
        ] {
            for [_, _, c, d] in terms {
                sum += c / (c * gamma + d)
            }
        }
        self.won_game_terms.len() as f64 + (0.5 * self.drawn_game_terms.len() as f64)
            - (gamma * sum)
    }

    /// Computes the log likelihood's second derivative for this timestamp.
    pub fn dlog2_likelihood(&self) -> f64 {
        let mut sum = 0f64;
        let gamma = self.gamma_rating();
        for terms in [
            &self.won_game_terms,
            &self.lost_game_terms,
            &self.drawn_game_terms,
        ] {
            for [_, _, c, d] in terms {
                sum += (c * d) / (c * gamma + d).powf(2f64)
            }
        }
        -gamma * sum
    }
}
