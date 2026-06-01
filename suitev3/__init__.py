"""
suitev3 - Does the quantum part matter in a HYBRID (circuit + classical head) QCNN?

Single focused test on Fashion-MNIST 4-class (inherited from suitev2):
  - flat_no_pool quantum ansatz (1 conv layer, no pooling) + v1-style classical head
  - amplitude encoding WITHOUT norm injection
  - four arms x three seeds, paired on the data split:
      A "trained"  : quantum circuit trained  + head trained
      B "frozen"   : quantum circuit FROZEN at a meaningful random init (angles ~ U(0,2pi))
                     + head trained  (the "quantum random features" null)
      C1 "logistic": logistic regression on the 256 raw pixels
      C2 "randfeat": fixed random classical projection -> cos -> trainable head
                     (classical analogue of the frozen-quantum arm)

Reading:
  A vs B   -> value of TRAINING the circuit (A>>B => real quantum learning; A~B => head does the work)
  B vs C2  -> value of the quantum STRUCTURE  (B>>C2 => quantum random features beat classical random)
  A vs C1  -> quantum vs a trivial classical model

Hyperparameters (optimizer, epochs, split, weight-decay, scheduler, label smoothing)
are inherited verbatim from suitev2.config.BASELINE.
"""
