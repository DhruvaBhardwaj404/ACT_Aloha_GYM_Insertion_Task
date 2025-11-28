import imageio.v2 as imageio
import gymnasium as gym
import gym_aloha  # Needed for the environment registration
import numpy as np
import sys
import torch
import os
import json
import draccus
from collections import deque
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.processor_act import PolicyProcessorPipeline
from lerobot.policies.utils import PolicyAction
from typing import Dict, Any

# --- Configuration ---
POLICY_PATH = "migrated_output"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def to_device(data: Any, device: str) -> Any:
    """Recursively moves tensors in a structure to the specified device."""
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    return data


def act_policy_forward_inference(self: ACTPolicy, batch: Dict[str, Any]):
    """
    Hacked forward method to bypass KL divergence loss calculation during inference.
    """
    actions_hat, _ = self.model(batch)
    return (actions_hat,)


class TemporalEnsembler:
    """
    Manages the history of action chunks and computes the temporally ensembled action.
    The logic is correct: for the current time step t, it takes the prediction for t
    from the newest chunk (index 0), the second newest chunk (index 1), etc.
    """

    def __init__(self, chunk_size: int, action_dim: int, decay_coef: float = 0.02):
        self.chunk_history = deque(maxlen=chunk_size)
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.decay_coef = decay_coef

        # Pre-calculate and normalize the full set of weights
        # weights[0] is for the newest prediction (chunk_history[0])
        self.weights = np.exp(-self.decay_coef * np.arange(chunk_size))
        self.weights = self.weights / np.sum(self.weights)  # Normalize

    def add_chunk(self, action_chunk_np: np.ndarray):
        """Adds a new action chunk (C, D) to the history."""
        # action_chunk_np should be (chunk_size, action_dim)
        self.chunk_history.append(action_chunk_np)

    def get_ensembled_action(self) -> np.ndarray:
        """
        Calculates the temporally ensembled action for the current time step t.
        """
        if not self.chunk_history:
            return np.zeros(self.action_dim, dtype=np.float32)

        current_time_predictions = []

        # Iterate through the chunks in history (most recent first)
        for i, chunk in enumerate(self.chunk_history):
            # The prediction for the current time step 't' is at index 'i' in the 'i'-th newest chunk
            if i < self.chunk_size:
                prediction_for_t = chunk[i]
                current_time_predictions.append(prediction_for_t)
            else:
                break  # Should be caught by deque's maxlen, but good for safety

        # 1. Convert to numpy array for weighted sum
        pred_array = np.array(current_time_predictions)

        # 2. Select the corresponding weights (only for the available predictions)
        active_weights = self.weights[:len(current_time_predictions)]

        # 3. Re-normalize active weights in case history is not full
        active_weights = active_weights / np.sum(active_weights)

        # 4. Calculate the weighted average: sum(w_i * pred_i) / sum(w_i)
        # active_weights[:, None] broadcasts the weights across the action dimension
        ensembled_action = np.sum(pred_array * active_weights[:, None], axis=0)

        return ensembled_action


def run_pretrained():
    """Runs the inference loop using the migrated ACT policy with temporal ensembling."""

    print(f"Using device: {DEVICE}")

    # 1. CONFIG LOADING
    print("Manually loading config file...")
    config_file_path = os.path.join(POLICY_PATH, "config.json")
    if not os.path.exists(config_file_path):
        print(f"Error: Could not find config file at {config_file_path}")
        # Use google to check for POLICY_PATH content
        return
    with open(config_file_path, 'r') as f:
        raw_config_data = json.load(f)
    if 'type' in raw_config_data:
        # Remove the 'type' key which can cause issues with draccus decoding
        del raw_config_data['type']
    try:
        config: ACTConfig = draccus.decode(ACTConfig, raw_config_data)
        print("ACTConfig instantiated successfully.")
    except draccus.utils.DecodingError as e:
        print(f"Failed to decode config: {e}")
        return
    print("-" * 30)

    # 2. POLICY AND PROCESSOR LOADING
    print("Loading weights from local directory...")
    policy = ACTPolicy.from_pretrained(POLICY_PATH).to(DEVICE).eval()
    print(f"Policy model loaded: {type(policy).__name__}")

    # Inject the hacked forward method for inference-only
    policy.forward = act_policy_forward_inference.__get__(policy, ACTPolicy)

    # Explicitly set kl_weight to 0.0 to ensure no loss calculation is attempted
    policy.config.kl_weight = 0.0
    if hasattr(policy, 'model') and hasattr(policy.model, 'config'):
        policy.model.config.kl_weight = 0.0

    print("Loading PolicyProcessorPipelines...")
    policy_preprocessor = PolicyProcessorPipeline.from_pretrained(
        POLICY_PATH,
        config_filename="policy_preprocessor.json",
        policy_config=config,
        device=DEVICE,
        local_files_only=True
    )
    policy_postprocessor = PolicyProcessorPipeline.from_pretrained(
        POLICY_PATH,
        config_filename="policy_postprocessor.json",
        policy_config=config,
        device=DEVICE,
        local_files_only=True
    )
    print("-" * 30)

    # 3. PARAMETER SETUP
    # Use config value, default to 100 if missing
    CHUNK_SIZE = getattr(config, 'chunk_size', 100)
    DECAY_COEF = 0.02
    # Ensure action_dim is correctly inferred or set
    ACTION_DIM = 14  # Aloha insertion environment has 14 DOFs

    # 4. EPISODE LOOP
    for e in range(3):  # Reduced episodes for quicker test
        print(f"\n🚀 Starting Episode {e}")
        # Create environment
        env = gym.make("gym_aloha/AlohaInsertion-v0", obs_type="pixels_agent_pos")
        obs, info = env.reset()
        frames = []
        last_action = np.zeros(ACTION_DIM, dtype=np.float32)

        ensembler = TemporalEnsembler(
            chunk_size=CHUNK_SIZE,
            action_dim=ACTION_DIM,
            decay_coef=DECAY_COEF
        )

        # 5. INFERENCE LOOP
        for i in range(1000):
            # 5a. Prepare raw observation
            image_raw_hwc = obs["pixels"]["top"].astype(np.float32) / 255.0
            # Get the agent's current joint positions (qpos)
            qpos_raw = obs["agent_pos"][:ACTION_DIM].astype(np.float32)

            # Transpose image (C, H, W) and add Batch (B=1, C, H, W)
            image_raw_chw = np.transpose(image_raw_hwc, (2, 0, 1))
            image_raw_seq = np.expand_dims(image_raw_chw, axis=0)  # (1, C, H, W)

            # State: (D,) -> (B=1, D)
            qpos_raw_seq = np.expand_dims(qpos_raw, axis=0)  # (1, D)

            # Action: (D,) -> (B=1, D)
            last_action_seq = np.expand_dims(last_action, axis=0)

            # Instantiate PolicyAction object for the preprocessor
            policy_action = PolicyAction(data=last_action_seq)
            policy_action.is_pad = np.array([False], dtype=np.bool_).reshape(1)

            raw_policy_input = {
                "observation.images.top": image_raw_seq,
                "observation.state": qpos_raw_seq,
                "action": policy_action,
                "action_is_pad": policy_action.is_pad
            }

            # 5b. Pre-process and move to device
            obs_for_policy = policy_preprocessor(raw_policy_input)
            obs_for_policy = to_device(obs_for_policy, DEVICE)

            # Handle key remapping required by the ACT model's forward pass
            if isinstance(obs_for_policy, dict):
                # Ensure 'observation.environment_state' exists, even if it's a copy of 'observation.state'
                if 'observation.state' in obs_for_policy and 'observation.environment_state' not in obs_for_policy:
                    obs_for_policy['observation.environment_state'] = obs_for_policy['observation.state']

                # Ensure 'observation.images' is a tuple of image tensors
                if 'observation.images.top' in obs_for_policy and 'observation.images' not in obs_for_policy:
                    image_list = [obs_for_policy['observation.images.top']]
                    obs_for_policy['observation.images'] = tuple(image_list)

                # Remove loss-related keys if present (for robustness)
                obs_for_policy.pop('action_mu', None)
                obs_for_policy.pop('action_log_sigma_x2', None)

            # 5c. Inference
            with torch.no_grad():
                action_chunk_output = policy(obs_for_policy)

            # The output is a tuple containing the action tensor: (1, C, D)
            predicted_action_tensor = action_chunk_output[0]

            # 5d. Post-process the action chunk
            action_dict_for_postprocessor = {"action": predicted_action_tensor}
            action_chunk_dict = policy_postprocessor(action_dict_for_postprocessor)
            action_chunk_np_with_batch = action_chunk_dict['action'].cpu().numpy()
            action_chunk_np = action_chunk_np_with_batch.squeeze(axis=0)  # (C, D)

            # 5e. Temporal Ensembling
            ensembler.add_chunk(action_chunk_np)
            action_np = ensembler.get_ensembled_action()

            # Input validation check
            assert action_np.ndim == 1, f"Action dim error: expected 1D, got {action_np.ndim}"
            assert action_np.shape[
                       0] == ACTION_DIM, f"Action shape error: expected {ACTION_DIM}, got {action_np.shape[0]}"

            # 5f. Environment step
            last_action = action_np  # Store the executed action for the next policy input
            obs, reward, terminated, truncated, info = env.step(action_np)

            # Rendering/Saving Frames
            rgb_array = env.render()
            frames.append(rgb_array)

            # if terminated or truncated:
            #     print(f"Episode finished after {i + 1} steps.")
            #     break

        # 6. Save video and cleanup
        if frames:
            print("Saving video...")
            imageio.mimsave(f"aloha_inference_{e}.mp4", frames, fps=60)

        env.close()

    print("\nInference run complete.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py [pretrained|train]")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "pretrained":
        run_pretrained()
    else:
        print(f"Unknown mode: {mode}")