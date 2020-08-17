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
/// random_agent.reset(Some(Seed::from(0))).unwrap();
///
/// let chosen_action = random_agent.choose_action(&EnvironmentState::default()).unwrap();
///
/// assert_eq!(&vec![2], chosen_action.dimensions());
/// assert_eq!(Ok(&DimensionValue::INTEGER(2)), chosen_action.get_value(&vec![0]));
/// assert_eq!(Ok(&DimensionValue::FLOAT(2.0)), chosen_action.get_value(&vec![1]));
/// ```
pub struct RandomAgent {
    action_spaces: ActionSpace,
    rng: ChaCha20Rng,
}

impl RandomAgent {
    /// Creates a new RandomAgent with the provided ActionSpace.
    pub fn with(action_spaces: ActionSpace) -> Self {
        Self {
            action_spaces,
            rng: ChaCha20Rng::from_entropy(),
        }
    }
}

impl Agent<RandomAgentError> for RandomAgent {
    fn reset(&mut self, random_seed: Option<Seed>) -> Result<(), RandomAgentError> {
        if let Some(seed) = random_seed {
            self.rng = ChaCha20Rng::from_seed(seed.into());
        } else {
            self.rng = ChaCha20Rng::from_entropy();
        }
        Ok(())
    }

    fn choose_action(
        &mut self,
        _: &EnvironmentState,
    ) -> Result<AgentAction, RandomAgentError> {
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

    fn close(&mut self) -> Result<(), RandomAgentError> {
        Ok(())
    }
}
