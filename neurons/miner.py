import time
import typing
import bittensor as bt

import qubittensor
from qubittensor.base.miner import BaseMinerNeuron
from qubittensor.protocol import ChallengeCircuits, CompletedCircuits


class Miner(BaseMinerNeuron):
    """
    Receive circuit challenges, solve them, and return the completed circuit.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("Miner initialized and ready to receive circuit challenges.")


    async def forward(
        self, synapse: ChallengeCircuits
    ) -> CompletedCircuits:
        """
        Processes the incoming 'ChallengeCircuits' synapse.
        The miner will:
        1. Verify the validator's signature on the challenge.
        2. Solve the circuit provided in `synapse.circuit_data`.
        3. Compute the solution hash.
        4. Populate a `CompletedCircuits` synapse with the results and return it.

        Args:
            synapse (ChallengeCircuits): The incoming synapse containing the circuit challenge.

        Returns:
            CompletedCircuits: The synapse object with the solved circuit details.
        """
        bt.logging.info(f"Received ChallengeCircuits from validator: {synapse.dendrite.hotkey}")
        bt.logging.debug(f"Challenge details: {synapse.circuit_data}, Difficulty: {synapse.difficulty_level}")

        # --- 1. Verify the validator's signature on the challenge ---
        validator_hotkey_address = synapse.dendrite.hotkey

        # Re-create the hash of the challenge content as the validator would have done.
        temp_challenge_for_hash = ChallengeCircuits(
            circuit_data=synapse.circuit_data,
            solution_hash=synapse.solution_hash,
            difficulty_level=synapse.difficulty_level,
            # Do NOT include validator_signature when hashing for verification
            validator_signature=None
        )
        challenge_content_hash_for_verification = temp_challenge_for_hash.get_serializable_hash()

        received_validator_signature = synapse.validator_signature

        signature_is_valid = False
        if received_validator_signature:
            try:
                signature_is_valid = bt.verify_signature(
                    ss58_address=validator_hotkey_address,
                    message_hash=challenge_content_hash_for_verification,
                    signature=bytes.fromhex(received_validator_signature)
                )
            except Exception as e:
                bt.logging.error(f"Error verifying validator signature: {e}")
                signature_is_valid = False
        else:
            bt.logging.warning("No validator signature provided in the challenge.")

        if not signature_is_valid:
            bt.logging.warning(f"Invalid or missing validator signature from {validator_hotkey_address}. Rejecting challenge.")
            # If the signature is invalid, the miner should ideally not process the challenge
            # or return an empty/error response. For this example, we'll return a default
            # CompletedCircuits indicating failure.
            return CompletedCircuits(
                validator_signature=received_validator_signature, # Return the invalid signature for validator to check
                circuit_data=synapse.circuit_data,
                solution_hash="", # Indicate no solution was produced
                difficulty_level=synapse.difficulty_level
            )
        
        bt.logging.info("Validator signature verified successfully.")

        # --- 2. Solve the circuit ---
        # TODO(developer): Replace this with your actual circuit solving logic.
        # This is where your miner's "work" happens.
        bt.logging.info(f"Solving circuit: {synapse.circuit_data}...")
        try:
            # Simulate solving the circuit
            time.sleep(1) # Simulate computation time
            solved_result = "8" # This should be the actual output of your circuit solver
            
            # --- 3. Compute the solution hash ---
            miner_solution_hash = bt.hash(solved_result).hexdigest()
            bt.logging.info(f"Circuit solved. Miner's solution hash: {miner_solution_hash}")

        except Exception as e:
            bt.logging.error(f"Error solving circuit: {e}")
            # If solving fails, return an empty solution hash
            miner_solution_hash = ""

        # --- 4. Populate a CompletedCircuits synapse and return it ---
        # The miner fills in the empty fields of the CompletedCircuits synapse.
        # Importantly, the miner must return the *original* validator_signature
        # received in the challenge, as this allows the validator to trace the
        # solved circuit back to its origin and perform further verification.
        response_synapse = CompletedCircuits(
            validator_signature=synapse.validator_signature, # Return the validator's original signature
            circuit_data=synapse.circuit_data,              # Return the original circuit data for context
            solution_hash=miner_solution_hash,              # The hash of the miner's solution
            difficulty_level=synapse.difficulty_level       # Return the original difficulty
        )

        bt.logging.info("CompletedCircuits synapse prepared and returned.")
        return response_synapse

    async def blacklist(
        self, synapse: ChallengeCircuits
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted.
        This is crucial for security and resource management.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        # Get the UID of the sender
        try:
            # This can raise ValueError if the hotkey is not found in metagraph
            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        except ValueError:
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        # Check if the sender is a validator (optional, but common for challenge-response)
        if not self.metagraph.validator_permit[uid]:
            bt.logging.warning(
                f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Non-validator hotkey"

        # You might also add logic here to blacklist if the validator's stake is too low,
        # or if the `synapse.difficulty_level` is outside an acceptable range.
        # e.g., if synapse.difficulty_level > MAX_ACCEPTABLE_DIFFICULTY: return True, "Too difficult"

        bt.logging.trace(
            f"Not blacklisting recognized and permitted hotkey {synapse.dendrite.hotkey} (UID: {uid})"
        )
        return False, "Hotkey recognized and permitted!"

    async def priority(self, synapse: ChallengeCircuits) -> float:
        """
        The priority function determines the order in which requests are handled.
        Higher values indicate that the request should be processed first.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0

        try:
            caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        except ValueError:
            # Should ideally be caught by blacklist, but a fallback for unregistered hotkeys
            return 0.0

        # Prioritize based on the validator's stake. Higher stake = higher priority.
        priority = float(self.metagraph.S[caller_uid])
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)