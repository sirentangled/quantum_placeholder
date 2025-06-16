from lib.circuit_gen import CircuitParams

def main():
    # example, please change this to whatever you need to do this in a nice
    # automated fashion
    difficulty_level = 1.0 # min
    seed = 1054 # whatever
    circuit_params = CircuitParams.from_difficulty(difficulty_level)
    print(f"qubits: {circuit_params.nqubits}")
    print(f"rqc depth: {circuit_params.rqc_depth}")
    print(f"pqc depth: {circuit_params.pqc_depth}")
    circuit = circuit_params.compute_circuit(seed)
    qasm_out = circuit.to_qasm()
    print(qasm_out)
    print("Target state")
    print(circuit.target_state)

if __name__ == "__main__":
    main()


# 0.0: 001101001111
# 1.0: 0011010011110000011111
