import time
import bittensor as bt
import asyncio # Ensure asyncio is imported if you're using it in the validator's main loop or other parts

from qubittensor.validator.reward import get_rewards
from qubittensor.utils.uids import get_random_uids

# Import your custom synapses from your protocol.py file
from qubittensor.protocol import ChallengeCircuits, CompletedCircuits # <-- New import!


async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network with circuit challenges and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # --- 1. Select miners to query ---
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    bt.logging.debug(f"Selected miner UIDs for query: {miner_uids}")

    # --- 2. Generate a circuit challenge ---
    # In a real scenario, this would involve logic to create a unique and solvable circuit.
    generated_circuit_data = "{'circuit_type': 'logic_gate', 'operation': 'XOR', 'inputs': [1,0]}"
    correct_solution = "1" # Expected result of XOR(1,0)
    correct_solution_hash = bt.hash(correct_solution).hexdigest()
    difficulty = 2 # Example difficulty

    # --- 3. Create and sign the ChallengeCircuits synapse ---
    challenge_synapse = ChallengeCircuits(
        circuit_data=generated_circuit_data,
        solution_hash=correct_solution_hash, # This is the hash of the *correct* solution
        difficulty_level=difficulty
    )

    # Explicitly sign the challenge with the validator's hotkey
    challenge_content_hash = challenge_synapse.get_serializable_hash()
    validator_signature = self.wallet.hotkey.sign(
        challenge_content_hash.encode('utf-8')
    ).hex()
    challenge_synapse.validator_signature = validator_signature

    bt.logging.info(f"Created ChallengeCircuits for step {self.step}: Difficulty={difficulty}")
    bt.logging.debug(f"Challenge content hash: {challenge_content_hash}, Signature: {validator_signature[:10]}...")

    # --- 4. Query the network with your ChallengeCircuits synapse ---
    # The dendrite client queries the network.
    responses = await self.dendrite(
        # Send the query to selected miner axons in the network.
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        # Send your ChallengeCircuits synapse.
        synapse=challenge_synapse,
        # We set deserialize=False because we want the raw CompletedCircuits
        # synapse back to manually verify its fields, including the signature and solution hash.
        deserialize=False,
        timeout=12 # Set a reasonable timeout for miners to respond
    )

    bt.logging.info(f"Received {len(responses)} responses from miners.")

    # --- 5. Process and score responses ---
    # Initialize rewards for all miners to 0
    all_rewards = [0.0] * len(miner_uids)
    
    # Keep track of which miners actually responded
    responded_uids = []
    
    for i, response in enumerate(responses):
        miner_uid = miner_uids[i] # Get the UID of the miner corresponding to this response
        axon = self.metagraph.axons[miner_uid] # Get the axon for logging

        if response is None:
            bt.logging.warning(f"No response from miner {miner_uid} ({axon.hotkey}).")
            continue # Skip to the next response

        if not isinstance(response, CompletedCircuits):
            bt.logging.warning(f"Invalid synapse type received from miner {miner_uid} ({axon.hotkey}). Expected CompletedCircuits, got {type(response)}.")
            continue # Skip to the next response

        bt.logging.info(f"Processing response from miner {miner_uid} ({axon.hotkey}).")
        bt.logging.debug(f"Miner {miner_uid} response: {response}")

        # --- Verification steps ---
        # 5a. Verify the validator_signature returned by the miner
        # This checks if the miner returned the original signature we sent.
        if response.validator_signature != validator_signature:
            bt.logging.warning(f"Miner {miner_uid} returned mismatched validator_signature.")
            # Penalize or do not reward if signature is tampered with
            continue 
        
        # You could also add a full cryptographic verification here if `response.validator_signature`
        # is meant to be cryptographically verified against the validator's hotkey directly,
        # rather than just matching the string sent. For this, you'd use bt.verify_signature
        # as shown in the previous validator example, passing the validator's hotkey and the original challenge hash.

        # 5b. Check if the miner's submitted solution hash matches the correct one
        if response.solution_hash == correct_solution_hash:
            bt.logging.info(f"Miner {miner_uid} submitted correct solution hash.")
            # Reward for correct solution
            all_rewards[i] = 1.0 # Full reward
            responded_uids.append(miner_uid) # Add to list of successful responders
        else:
            bt.logging.warning(f"Miner {miner_uid} submitted incorrect solution hash. Expected {correct_solution_hash}, got {response.solution_hash}.")
            # No reward for incorrect solution
            all_rewards[i] = 0.0

    bt.logging.info(f"Calculated rewards for this step: {all_rewards}")

    # --- 6. Update scores and set weights ---
    # This part leverages the BaseValidatorNeuron's existing mechanisms.
    # The `update_scores` function expects rewards corresponding to the `miner_uids` queried.
    # Miners that didn't respond or provided invalid responses will have 0.0 rewards.
    self.update_scores(all_rewards, miner_uids)

    # In a real subnet, you'd also want a mechanism for miners to submit
    # the *actual* solved circuits (the `CompletedCircuits` with a valid solution_hash)
    # when they've actually solved them, not just in immediate response to a challenge.
    # This could involve a separate protocol or a periodic "submission" query from the validator.
    
    # You might remove the sleep(5) here, as the BaseValidatorNeuron's main loop
    # typically handles the timing between forward calls.
    # time.sleep(5)