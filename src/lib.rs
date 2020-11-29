//! # Gymnarium Random Agents
//!
//! `gymnarium_agents_random` contains agents for the `gymnarium_libraries`
//! choosing their actions based on randomness.

extern crate gymnarium_base;
extern crate rand;
extern crate rand_chacha;

use std::fmt::Debug;

use gymnarium_base::{ActionSpace, Agent, AgentAction, EnvironmentState, Seed};

use rand::SeedableRng;

use rand_chacha::ChaCha20Rng;

use serde::{Deserialize, Serialize};

/// Possible errors occurring within this library.
///
/// Currently there are none and I don't think some will be here in the future.
#[derive(Debug)]
pub enum RandomAgentError {}

impl std::fmt::Display for RandomAgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}

impl std::error::Error for RandomAgentError {}

/// Agent which chooses his actions through random number generation.
///
/// # Example
///
/// ```
/// use gymnarium_agents_random::RandomAgent;
/// use gymnarium_base::{ActionSpace, Seed, Agent, EnvironmentState};
/// use gymnarium_base::space::{DimensionBoundaries, DimensionValue};
///
/// let mut random_agent = RandomAgent::with(ActionSpace::simple(vec![
///     DimensionBoundaries::from(1..=2),
///     DimensionBoundaries::from(2.0..=2.0)
/// ]));
/// random_agent.reseed(Some(Seed::from(0))).unwrap();
/// random_agent.reset().unwrap();
///
/// let chosen_action = random_agent.choose_action(&EnvironmentState::default()).unwrap();
///
/// assert_eq!(&vec![2], chosen_action.dimensions());
/// assert_eq!(DimensionValue::INTEGER(2), chosen_action[&[0]]);
/// assert_eq!(DimensionValue::FLOAT(2.0), chosen_action[&[1]]);
/// ```
pub struct RandomAgent {
    action_spaces: ActionSpace,
    last_seed: Seed,
    rng: ChaCha20Rng,
}

impl RandomAgent {
    /// Creates a new RandomAgent with the provided ActionSpace.
    pub fn with(action_spaces: ActionSpace) -> Self {
        let last_seed = Seed::new_random();
        Self {
            action_spaces,
            last_seed: last_seed.clone(),
            rng: ChaCha20Rng::from_seed(last_seed.into()),
        }
    }
}

impl Agent<RandomAgentError, RandomAgentStorage> for RandomAgent {
    fn reseed(&mut self, random_seed: Option<Seed>) -> Result<(), RandomAgentError> {
        if let Some(seed) = random_seed {
            self.last_seed = seed;
            self.rng = ChaCha20Rng::from_seed(self.last_seed.clone().into());
        } else {
            self.last_seed = Seed::new_random();
            self.rng = ChaCha20Rng::from_seed(self.last_seed.clone().into());
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<(), RandomAgentError> {
        Ok(())
    }

    fn choose_action(&mut self, _: &EnvironmentState) -> Result<AgentAction, RandomAgentError> {
        Ok(self.action_spaces.sample_with(&mut self.rng))
    }

    fn process_reward(
        &mut self,
        _: &EnvironmentState,
        _: &EnvironmentState,
        _: f64,
        _: bool,
    ) -> Result<(), RandomAgentError> {
        Ok(())
    }

    fn load(&mut self, data: RandomAgentStorage) -> Result<(), RandomAgentError>
    where
        Self: std::marker::Sized,
    {
        self.last_seed = data.last_seed;
        self.rng = ChaCha20Rng::from_seed(self.last_seed.clone().into());
        self.rng.set_word_pos(data.rng_word_pos);
        Ok(())
    }

    fn store(&self) -> RandomAgentStorage {
        RandomAgentStorage {
            last_seed: self.last_seed.clone(),
            rng_word_pos: self.rng.get_word_pos(),
        }
    }

    fn close(&mut self) -> Result<(), RandomAgentError> {
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
pub struct RandomAgentStorage {
    last_seed: Seed,
    rng_word_pos: u128,
}
