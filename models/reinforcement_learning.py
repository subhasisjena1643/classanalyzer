"""
Reinforcement Learning Framework for Continuous Model Improvement
Implements RL agents for optimizing detection and engagement analysis performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque
import pickle
import random
import time
from pathlib import Path
from loguru import logger

try:
    from stable_baselines3 import PPO, A2C, SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("Stable Baselines3 not available, using custom RL implementation")

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    logger.warning("Gymnasium not available")


class RLAgent:
    """
    Reinforcement Learning agent for continuous model improvement.
    Optimizes detection thresholds, engagement parameters, and system performance.
    """
    
    def __init__(self, algorithm: str = "PPO", config: Any = None, checkpoint_manager=None):
        """
        Initialize RL agent with checkpoint support.

        Args:
            algorithm: RL algorithm ("PPO", "A2C", "SAC", "DQN")
            config: Configuration object
            checkpoint_manager: CheckpointManager instance for saving/loading states
        """
        self.algorithm = algorithm.upper()
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        
        # RL parameters
        self.learning_rate = config.get("reinforcement_learning.learning_rate", 0.0003) if config else 0.0003
        self.update_frequency = config.get("reinforcement_learning.update_frequency", 100) if config else 100
        self.buffer_size = config.get("reinforcement_learning.experience_replay.buffer_size", 10000) if config else 10000
        self.batch_size = config.get("reinforcement_learning.experience_replay.batch_size", 64) if config else 64
        
        # Reward weights
        self.reward_weights = config.get("reinforcement_learning.reward_shaping", {
            "attendance_accuracy_weight": 0.4,
            "engagement_precision_weight": 0.3,
            "latency_penalty_weight": 0.2,
            "false_positive_penalty": 0.1
        }) if config else {
            "attendance_accuracy_weight": 0.4,
            "engagement_precision_weight": 0.3,
            "latency_penalty_weight": 0.2,
            "false_positive_penalty": 0.1
        }
        
        # State and action spaces
        self.state_dim = 20  # System metrics, performance indicators
        self.action_dim = 10  # Threshold adjustments, parameter tuning
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=self.buffer_size)
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_count = 0
        self.total_steps = 0
        self.best_performance = 0.0
        self.performance_history = []

        # Checkpoint tracking
        self.last_checkpoint_performance = 0.0
        self.checkpoint_improvement_threshold = 0.02  # 2% improvement triggers checkpoint
        self.last_checkpoint_time = time.time()
        self.checkpoint_interval = 600  # Save every 10 minutes minimum

        # Initialize agent
        self.agent = self._initialize_agent()

        # Try to load from checkpoint if available
        self._load_from_checkpoint()
        
        # Current state tracking
        self.current_state = None
        self.last_action = None
        self.last_reward = 0.0
        
        logger.info(f"RL Agent initialized: {self.algorithm}")
    
    def _initialize_agent(self):
        """Initialize the RL agent based on algorithm choice."""
        if SB3_AVAILABLE and self.algorithm in ["PPO", "A2C", "SAC"]:
            return self._initialize_sb3_agent()
        else:
            return self._initialize_custom_agent()
    
    def _initialize_sb3_agent(self):
        """Initialize Stable Baselines3 agent."""
        try:
            # Create custom environment
            env = ClassroomOptimizationEnv(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                reward_weights=self.reward_weights
            )
            
            if self.algorithm == "PPO":
                agent = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=self.learning_rate,
                    verbose=1,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            elif self.algorithm == "A2C":
                agent = A2C(
                    "MlpPolicy",
                    env,
                    learning_rate=self.learning_rate,
                    verbose=1,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            elif self.algorithm == "SAC":
                agent = SAC(
                    "MlpPolicy",
                    env,
                    learning_rate=self.learning_rate,
                    verbose=1,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            
            return agent
            
        except Exception as e:
            logger.error(f"Failed to initialize SB3 agent: {e}")
            return self._initialize_custom_agent()
    
    def _initialize_custom_agent(self):
        """Initialize custom RL agent."""
        if self.algorithm == "DQN":
            return DQNAgent(self.state_dim, self.action_dim, self.learning_rate)
        else:
            # Default to simple policy gradient
            return PolicyGradientAgent(self.state_dim, self.action_dim, self.learning_rate)
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Get action from current state.
        
        Args:
            state: Current system state
            
        Returns:
            Action to take (parameter adjustments)
        """
        try:
            self.current_state = state
            
            if SB3_AVAILABLE and hasattr(self.agent, 'predict'):
                action, _ = self.agent.predict(state, deterministic=False)
            else:
                action = self.agent.get_action(state)
            
            self.last_action = action
            return action
            
        except Exception as e:
            logger.error(f"Action selection failed: {e}")
            # Return neutral action (no changes)
            return np.zeros(self.action_dim)
    
    def update_online(self, feedback: Dict[str, Any]) -> float:
        """
        Update agent with online feedback.
        
        Args:
            feedback: Real-time performance feedback
            
        Returns:
            Calculated reward
        """
        try:
            if self.current_state is None or self.last_action is None:
                return 0.0
            
            # Calculate reward from feedback
            reward = self._calculate_reward(feedback)
            self.last_reward = reward
            
            # Store experience
            if len(self.experience_buffer) > 0:
                # Get next state from feedback
                next_state = self._extract_state_from_feedback(feedback)
                
                # Store transition
                experience = {
                    'state': self.current_state,
                    'action': self.last_action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': False
                }
                
                self.experience_buffer.append(experience)
            
            # Update agent if enough experiences
            if len(self.experience_buffer) >= self.batch_size and self.total_steps % self.update_frequency == 0:
                self._update_agent()
            
            self.total_steps += 1
            return reward
            
        except Exception as e:
            logger.error(f"Online update failed: {e}")
            return 0.0
    
    def _calculate_reward(self, feedback: Dict[str, Any]) -> float:
        """Calculate reward from system feedback."""
        try:
            reward = 0.0
            
            # Attendance accuracy reward
            if 'attendance_accuracy' in feedback:
                accuracy = feedback['attendance_accuracy']
                target_accuracy = 0.98
                accuracy_reward = (accuracy - target_accuracy) * 10  # Scale reward
                reward += self.reward_weights['attendance_accuracy_weight'] * accuracy_reward
            
            # Engagement precision reward
            if 'engagement_precision' in feedback:
                precision = feedback['engagement_precision']
                target_precision = 0.70
                precision_reward = (precision - target_precision) * 10
                reward += self.reward_weights['engagement_precision_weight'] * precision_reward
            
            # Latency penalty
            if 'processing_latency_ms' in feedback:
                latency = feedback['processing_latency_ms']
                target_latency = 5000  # 5 seconds
                latency_penalty = max(0, (latency - target_latency) / 1000)  # Penalty for exceeding target
                reward -= self.reward_weights['latency_penalty_weight'] * latency_penalty
            
            # False positive penalty
            if 'false_positive_rate' in feedback:
                fp_rate = feedback['false_positive_rate']
                fp_penalty = fp_rate * 5  # Penalty for false positives
                reward -= self.reward_weights['false_positive_penalty'] * fp_penalty
            
            # Additional performance metrics
            if 'fps' in feedback:
                fps = feedback['fps']
                target_fps = 30
                fps_reward = min(1.0, fps / target_fps) * 0.5
                reward += fps_reward
            
            return reward
            
        except Exception as e:
            logger.error(f"Reward calculation failed: {e}")
            return 0.0
    
    def _extract_state_from_feedback(self, feedback: Dict[str, Any]) -> np.ndarray:
        """Extract state vector from feedback."""
        try:
            state = np.zeros(self.state_dim)
            
            # Performance metrics
            state[0] = feedback.get('attendance_accuracy', 0.0)
            state[1] = feedback.get('engagement_precision', 0.0)
            state[2] = feedback.get('processing_latency_ms', 0.0) / 10000  # Normalize
            state[3] = feedback.get('false_positive_rate', 0.0)
            state[4] = feedback.get('fps', 0.0) / 60  # Normalize
            
            # Detection metrics
            state[5] = feedback.get('face_detection_confidence', 0.0)
            state[6] = feedback.get('face_recognition_similarity', 0.0)
            state[7] = feedback.get('liveness_confidence', 0.0)
            
            # Engagement metrics
            state[8] = feedback.get('attention_score', 0.0)
            state[9] = feedback.get('participation_score', 0.0)
            state[10] = feedback.get('confusion_score', 0.0)
            
            # System metrics
            state[11] = feedback.get('cpu_usage', 0.0) / 100  # Normalize
            state[12] = feedback.get('memory_usage', 0.0) / 100  # Normalize
            state[13] = feedback.get('gpu_usage', 0.0) / 100  # Normalize
            
            # Classroom metrics
            state[14] = feedback.get('total_students', 0.0) / 50  # Normalize
            state[15] = feedback.get('engaged_students', 0.0) / 50  # Normalize
            state[16] = feedback.get('attention_rate', 0.0)
            
            # Time-based features
            state[17] = feedback.get('time_of_day', 0.0) / 24  # Normalize
            state[18] = feedback.get('session_duration', 0.0) / 3600  # Normalize
            state[19] = feedback.get('frame_number', 0.0) / 10000  # Normalize
            
            return state
            
        except Exception as e:
            logger.error(f"State extraction failed: {e}")
            return np.zeros(self.state_dim)
    
    def _update_agent(self):
        """Update the RL agent with batch of experiences."""
        try:
            if SB3_AVAILABLE and hasattr(self.agent, 'learn'):
                # For SB3 agents, learning is handled internally
                pass
            else:
                # Custom agent update
                batch = random.sample(self.experience_buffer, self.batch_size)
                self.agent.update(batch)
            
            logger.debug(f"Agent updated with {self.batch_size} experiences")
            
        except Exception as e:
            logger.error(f"Agent update failed: {e}")
    
    async def update_from_session(self, session_summary: Dict[str, Any]):
        """Update agent with complete session feedback."""
        try:
            # Calculate session-level reward
            session_reward = self._calculate_session_reward(session_summary)
            
            # Update episode tracking
            self.episode_rewards.append(session_reward)
            self.episode_count += 1
            
            # Perform batch update if using custom agent
            if not SB3_AVAILABLE:
                if len(self.experience_buffer) >= self.batch_size:
                    self._update_agent()
            
            logger.info(f"Session update completed. Episode {self.episode_count}, Reward: {session_reward:.3f}")
            
        except Exception as e:
            logger.error(f"Session update failed: {e}")
    
    def _calculate_session_reward(self, session_summary: Dict[str, Any]) -> float:
        """Calculate reward for entire session."""
        try:
            metrics = session_summary.get('metrics', {})
            
            # Base reward from session performance
            base_reward = 0.0
            
            if 'attendance_accuracy' in metrics:
                base_reward += metrics['attendance_accuracy'] * 10
            
            if 'engagement_precision' in metrics:
                base_reward += metrics['engagement_precision'] * 8
            
            if 'average_latency' in metrics:
                latency_penalty = max(0, (metrics['average_latency'] - 5000) / 1000)
                base_reward -= latency_penalty
            
            # Bonus for consistency
            if 'performance_variance' in metrics:
                consistency_bonus = max(0, 1.0 - metrics['performance_variance'])
                base_reward += consistency_bonus * 2
            
            return base_reward
            
        except Exception as e:
            logger.error(f"Session reward calculation failed: {e}")
            return 0.0
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained RL model."""
        try:
            if SB3_AVAILABLE and hasattr(self.agent, 'save'):
                self.agent.save(filepath)
            else:
                # Save custom agent
                model_data = {
                    'agent': self.agent,
                    'experience_buffer': list(self.experience_buffer),
                    'episode_rewards': self.episode_rewards,
                    'episode_count': self.episode_count,
                    'total_steps': self.total_steps
                }
                
                with open(filepath, 'wb') as f:
                    pickle.dump(model_data, f)
            
            logger.info(f"RL model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save RL model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained RL model."""
        try:
            if SB3_AVAILABLE and hasattr(self.agent, 'load'):
                self.agent = self.agent.load(filepath)
            else:
                # Load custom agent
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.agent = model_data['agent']
                self.experience_buffer = deque(model_data['experience_buffer'], maxlen=self.buffer_size)
                self.episode_rewards = model_data['episode_rewards']
                self.episode_count = model_data['episode_count']
                self.total_steps = model_data['total_steps']
            
            logger.info(f"RL model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load RL model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get RL model information."""
        return {
            'algorithm': self.algorithm,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'learning_rate': self.learning_rate,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'buffer_size': len(self.experience_buffer),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'reward_weights': self.reward_weights
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get RL performance statistics."""
        if not self.episode_rewards:
            return {}
        
        return {
            'total_episodes': len(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards),
            'best_reward': np.max(self.episode_rewards),
            'worst_reward': np.min(self.episode_rewards),
            'reward_std': np.std(self.episode_rewards),
            'recent_average': np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
        }

    def _load_from_checkpoint(self):
        """Load RL agent state from checkpoint if available."""
        if not self.checkpoint_manager:
            return

        try:
            checkpoint_data = self.checkpoint_manager.load_latest_checkpoint()

            if not checkpoint_data:
                logger.info("📂 No RL checkpoint found - starting fresh")
                return

            # Load RL-specific state
            rl_state = checkpoint_data.get('models', {}).get('rl_agent', {})

            if rl_state:
                # Restore performance tracking
                self.episode_rewards = rl_state.get('episode_rewards', [])
                self.episode_count = rl_state.get('episode_count', 0)
                self.total_steps = rl_state.get('total_steps', 0)
                self.best_performance = rl_state.get('best_performance', 0.0)
                self.performance_history = rl_state.get('performance_history', [])
                self.last_checkpoint_performance = rl_state.get('last_checkpoint_performance', 0.0)

                # Restore experience buffer
                if 'experience_buffer' in rl_state:
                    self.experience_buffer = deque(rl_state['experience_buffer'], maxlen=self.buffer_size)

                # Load agent model if available
                if hasattr(self.agent, 'load') and 'agent_model' in rl_state:
                    try:
                        # For Stable Baselines3 agents
                        model_path = Path("temp_rl_model.zip")
                        with open(model_path, 'wb') as f:
                            f.write(rl_state['agent_model'])
                        self.agent.load(str(model_path))
                        model_path.unlink()  # Clean up temp file
                        logger.info("🔄 RL agent model loaded from checkpoint")
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to load RL agent model: {e}")

                logger.info(f"📂 RL state loaded from checkpoint")
                logger.info(f"   🔄 Episodes: {self.episode_count}")
                logger.info(f"   📈 Best performance: {self.best_performance:.3f}")
                logger.info(f"   💾 Experience buffer: {len(self.experience_buffer)} samples")

        except Exception as e:
            logger.error(f"❌ Failed to load RL checkpoint: {e}")

    def _should_save_checkpoint(self, current_performance: float) -> bool:
        """Determine if checkpoint should be saved based on performance improvement."""
        # Check for performance improvement
        improvement = current_performance - self.last_checkpoint_performance

        if improvement >= self.checkpoint_improvement_threshold:
            logger.info(f"📈 RL Performance improvement detected: {improvement:.3f}")
            return True

        # Check time-based saving
        time_since_last = time.time() - self.last_checkpoint_time
        if time_since_last >= self.checkpoint_interval:
            logger.info(f"⏰ Time-based RL checkpoint triggered: {time_since_last:.0f}s")
            return True

        return False

    def save_checkpoint(self, current_performance: float, force: bool = False):
        """Save RL agent checkpoint if improvement detected."""
        if not self.checkpoint_manager:
            return

        try:
            # Check if we should save
            if not force and not self._should_save_checkpoint(current_performance):
                return

            # Prepare RL state for checkpoint
            rl_state = {
                'episode_rewards': list(self.episode_rewards),
                'episode_count': self.episode_count,
                'total_steps': self.total_steps,
                'best_performance': self.best_performance,
                'performance_history': self.performance_history,
                'last_checkpoint_performance': current_performance,
                'experience_buffer': list(self.experience_buffer),
                'algorithm': self.algorithm,
                'learning_rate': self.learning_rate,
                'reward_weights': self.reward_weights
            }

            # Save agent model if possible
            if hasattr(self.agent, 'save'):
                try:
                    model_path = Path("temp_rl_model.zip")
                    self.agent.save(str(model_path))
                    with open(model_path, 'rb') as f:
                        rl_state['agent_model'] = f.read()
                    model_path.unlink()  # Clean up temp file
                except Exception as e:
                    logger.warning(f"⚠️  Failed to save RL agent model: {e}")

            # Create checkpoint data
            models = {'rl_agent': rl_state}
            performance_metrics = {
                'rl_performance': current_performance,
                'rl_episodes': self.episode_count,
                'rl_average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0
            }

            # Save checkpoint
            success = self.checkpoint_manager.save_checkpoint(
                models=models,
                performance_metrics=performance_metrics,
                force=force,
                checkpoint_type="rl_improvement" if not force else "rl_manual"
            )

            if success:
                self.last_checkpoint_performance = current_performance
                self.last_checkpoint_time = time.time()
                logger.info(f"💾 RL checkpoint saved - Performance: {current_performance:.3f}")

        except Exception as e:
            logger.error(f"❌ Failed to save RL checkpoint: {e}")


class PolicyGradientAgent:
    """Simple policy gradient agent for custom RL implementation."""
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Simple neural network policy
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from policy network."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.policy_net(state_tensor)
        return action.squeeze(0).numpy()
    
    def update(self, batch: List[Dict]):
        """Update policy with batch of experiences."""
        # Simple policy gradient update
        states = torch.FloatTensor([exp['state'] for exp in batch])
        actions = torch.FloatTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        
        # Forward pass
        predicted_actions = self.policy_net(states)
        
        # Calculate loss (negative log likelihood weighted by rewards)
        loss = -torch.mean(rewards * torch.sum((predicted_actions - actions) ** 2, dim=1))
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DQNAgent:
    """Deep Q-Network agent for discrete action spaces."""
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.epsilon = 0.1  # Exploration rate
        
        # Q-network
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            # Random action
            return np.random.uniform(-1, 1, self.action_dim)
        else:
            # Greedy action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            # Convert Q-values to continuous actions
            return torch.tanh(q_values).squeeze(0).numpy()
    
    def update(self, batch: List[Dict]):
        """Update Q-network with batch of experiences."""
        states = torch.FloatTensor([exp['state'] for exp in batch])
        actions = torch.FloatTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch])
        
        # Current Q-values
        current_q = self.q_net(states)
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.q_net(next_states)
            target_q = rewards.unsqueeze(1) + 0.99 * next_q  # Gamma = 0.99
        
        # Loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if GYM_AVAILABLE:
    class ClassroomOptimizationEnv(gym.Env):
        """Custom Gymnasium environment for classroom optimization."""
        
        def __init__(self, state_dim: int, action_dim: int, reward_weights: Dict):
            super().__init__()
            
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.reward_weights = reward_weights
            
            # Define action and observation spaces
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
            )
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
            )
            
            self.current_state = np.zeros(state_dim)
            
        def reset(self, seed=None, options=None):
            """Reset environment."""
            super().reset(seed=seed)
            self.current_state = np.random.uniform(0, 1, self.state_dim)
            return self.current_state, {}
        
        def step(self, action):
            """Take environment step."""
            # Simulate state transition
            self.current_state += np.random.normal(0, 0.01, self.state_dim)
            self.current_state = np.clip(self.current_state, 0, 1)
            
            # Calculate reward (simplified)
            reward = np.sum(self.current_state * action) / self.action_dim
            
            # Check if done (simplified)
            done = False
            
            return self.current_state, reward, done, False, {}
        
        def render(self):
            """Render environment (not implemented)."""
            pass
