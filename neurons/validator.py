import time
import bittensor as bt
from qubittensor.base.validator import BaseValidatorNeuron
import asyncio
import hashlib
import random

from qubittensor.protocol import ChallengeCircuits, CompletedCircuits

class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        bt.logging.info("load_state()")
        self.load_state()

    async def forward(self):
        """
        Generates circuit challenges, signs them with the validator's hotkey,
        sends them to miners, and evaluates responses, including signature verification.
        """
        try:
            bt.logging.info("Generating circuit challenge...")

            # --- 1. Generate a circuit and its solution ---
            generated_circuit_data = "{'circuit_type': '', 'inputs': [5, 3]}"
            correct_solution = "8" # The actual solution to the circuit
            correct_solution_hash = hashlib.sha256(correct_solution.encode('utf-8')).hexdigest()
            difficulty = 3 # Example difficulty level

            bt.logging.info(f"Circuit generated: {generated_circuit_data}")
            bt.logging.info(f"Expected solution hash: {correct_solution_hash}")

            # --- 2. Create the ChallengeCircuits synapse ---
            challenge_synapse = ChallengeCircuits(
                circuit_data=generated_circuit_data,
                solution_hash=correct_solution_hash,
                difficulty_level=difficulty,
            )

            # --- Explicitly Sign the synapse with the validator's hotkey ---
            challenge_content_to_sign = f"{challenge_synapse.circuit_data}" \
                                      f"{challenge_synapse.solution_hash}" \
                                      f"{challenge_synapse.difficulty_level}" \
                                      f"{self.wallet.hotkey.ss58_address}"

            challenge_content_hash = hashlib.sha256(challenge_content_to_sign.encode('utf-8')).hexdigest()

            validator_signature = self.wallet.hotkey.sign(
                challenge_content_hash.encode('utf-8')
            ).hex()

            challenge_synapse.validator_signature = validator_signature

            bt.logging.info(f"Challenge signed with validator hotkey: {self.wallet.hotkey.ss58_address}")
            bt.logging.trace(f"Validator signature: {validator_signature}")

            bt.logging.info("Querying miners with ChallengeCircuits...")

            # --- 3. Query miners ---
            # CHANGE MADE HERE: Simply cast self.metagraph.n to an int
            num_miners = int(self.metagraph.n)
            # --------------------------------------------------------

            sample_size = min(self.config.neuron.sample_size, num_miners)
            
            all_uids = list(range(num_miners))
            
            miner_uids = random.sample(all_uids, k=sample_size)

            queried_axons = [self.metagraph.axons[uid] for uid in miner_uids]
            
            responses = await self.dendrite(
                axons=queried_axons,
                synapse=challenge_synapse,
                deserialize=False,
                timeout=12,
            )

            bt.logging.info(f"Received {len(responses)} responses from miners.")

            # --- 4. Process and score responses ---
            all_rewards = [0.0] * len(miner_uids) 

            for i, response in enumerate(responses):
                miner_uid = miner_uids[i]
                axon_info = queried_axons[i] 

                bt.logging.debug(f"AXON INFO for response {i}: {axon_info}")

                if response is not None and isinstance(response, CompletedCircuits):
                    bt.logging.info(f"Received CompletedCircuits from miner {miner_uid}")
                    bt.logging.debug(f"Miner {miner_uid} response: {response}")

                    received_validator_signature = response.validator_signature

                    miner_received_challenge_content_to_verify = f"{response.circuit_data}" \
                                                               f"{response.solution_hash}" \
                                                               f"{response.difficulty_level}" \
                                                               f"{self.wallet.hotkey.ss58_address}"

                    miner_received_content_hash_for_verification = hashlib.sha256(
                        miner_received_challenge_content_to_verify.encode('utf-8')
                    ).hexdigest()

                    signature_valid = False
                    if received_validator_signature:
                        try:
                            signature_valid = bt.verify_signature(
                                ss58_address=self.wallet.hotkey.ss58_address,
                                message_hash=miner_received_content_hash_for_verification,
                                signature=bytes.fromhex(received_validator_signature)
                            )
                        except Exception as verify_e:
                            bt.logging.warning(f"Signature verification failed for miner {miner_uid}: {verify_e}")
                            signature_valid = False
                    else:
                        bt.logging.warning(f"No validator signature provided by miner {miner_uid}.")

                    if not signature_valid:
                        bt.logging.warning(f"Miner {miner_uid} returned an invalid or unverified validator_signature.")
                        all_rewards[i] = 0.0
                    elif response.solution_hash == correct_solution_hash:
                        bt.logging.info(f"Miner {miner_uid} submitted correct solution hash with valid signature.")
                        all_rewards[i] = 1.0
                    else:
                        bt.logging.warning(f"Miner {miner_uid} submitted incorrect solution hash.")
                        all_rewards[i] = 0.0
                else:
                    bt.logging.warning(f"No valid response or incorrect synapse type from miner {miner_uid}.")
                    all_rewards[i] = 0.0

            bt.logging.info(f"Calculated rewards: {all_rewards}")

            self.update_scores(all_rewards, miner_uids)

            return all_rewards

        except Exception as e:
            bt.logging.error(f"Error in validator forward pass: {e}")
            return []


if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)