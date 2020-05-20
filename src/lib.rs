//! # Gymnarium Random Agents
//!
//! `gymnarium_agents_random` contains agents for the `gymnarium_libraries`
//! choosing their actions based on randomness.

extern crate gymnarium_base;
extern crate rand;
extern crate rand_chacha;

use gymnarium_base::{
    ActionSpace, Agent, AgentAction, DimensionBoundaries, DimensionValue, EnvironmentState, Seed,
};

use rand::distributions::{Distribution, Uniform};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::cell::RefCell;

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
pub struct RandomAgent {
    action_spaces: ActionSpace,
    rng: RefCell<ChaCha20Rng>,
}

impl RandomAgent {
    /// Creates a new RandomAgent with the provided ActionSpace.
    pub fn with(action_spaces: ActionSpace) -> Self {
        Self {
            action_spaces,
            rng: RefCell::new(ChaCha20Rng::from_entropy()),
        }
    }
}

impl Agent<RandomAgentError> for RandomAgent {
    fn reset(&mut self, random_seed: Option<Seed>) -> Result<(), RandomAgentError> {
        if let Some(seed) = random_seed {
            self.rng = RefCell::new(ChaCha20Rng::from_seed(seed.into()));
        } else {
            self.rng = RefCell::new(ChaCha20Rng::from_entropy());
        }
        Ok(())
    }

    /// ```
    /// # use gymnarium_agents_random::RandomAgent;
    /// use gymnarium_base::{DimensionBoundaries, Seed, Agent, DimensionValue};
    /// let mut random_agent = RandomAgent::with(vec!(
    ///     DimensionBoundaries::DISCRETE { minimum: 1, maximum: 2 },
    ///     DimensionBoundaries::CONTINUOUS { minimum: 2.0, maximum: 2.0 }
    /// ));
    /// random_agent.reset(Some(Seed::from(0))).unwrap();
    /// let chosen_actions = random_agent.choose_action(&vec!()).unwrap();
    /// assert_eq!(2, chosen_actions.len());
    /// assert_eq!(DimensionValue::DISCRETE(2), chosen_actions[0]);
    /// assert_eq!(DimensionValue::CONTINUOUS(2.0), chosen_actions[1]);
    /// ```
    fn choose_action(&self, _: &EnvironmentState) -> Result<AgentAction, RandomAgentError> {
        let mut rng = self.rng.borrow_mut();
        Ok(self
            .action_spaces
            .iter()
            .map(|action_space| match *action_space {
                DimensionBoundaries::DISCRETE { minimum, maximum } => DimensionValue::DISCRETE(
                    Uniform::new_inclusive(minimum, maximum).sample(&mut (*rng)),
                ),
                DimensionBoundaries::CONTINUOUS { minimum, maximum } => DimensionValue::CONTINUOUS(
                    Uniform::new_inclusive(minimum, maximum).sample(&mut (*rng)),
                ),
            })
            .collect())
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
