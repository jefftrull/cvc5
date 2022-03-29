/******************************************************************************
 * Top contributors (to current version):
 *   Gereon Kremer
 *
 * This file is part of the cvc5 project.
 *
 * Copyright (c) 2009-2021 by the authors listed in the file AUTHORS
 * in the top-level source directory and their institutional affiliations.
 * All rights reserved.  See the file COPYING in the top-level source
 * directory for licensing information.
 * ****************************************************************************
 *
 * Proof checker utility for transcendental solver.
 */

#include "cvc5_private.h"

#ifndef CVC5__THEORY__ARITH__NL__TRANSCENDENTAL__PROOF_CHECKER_H
#define CVC5__THEORY__ARITH__NL__TRANSCENDENTAL__PROOF_CHECKER_H

#include "expr/node.h"
#include "proof/proof_checker.h"
#include "proof/proof_node.h"

namespace cvc5::internal {
namespace theory {
namespace arith {
namespace nl {
namespace transcendental {

/**
 * A checker for NlExt proofs
 *
 * This proof checker takes care of all proofs for lemmas from the
 * transcendental subsolver.
 */
class TranscendentalProofRuleChecker : public ProofRuleChecker
{
 public:
  TranscendentalProofRuleChecker() = default;
  ~TranscendentalProofRuleChecker() = default;

  /** Register all rules owned by this rule checker in pc. */
  void registerTo(ProofChecker* pc) override;

 protected:
  /** Return the conclusion of the given proof step, or null if it is invalid */
  Node checkInternal(PfRule id,
                     const std::vector<Node>& children,
                     const std::vector<Node>& args) override;
};

}  // namespace transcendental
}  // namespace nl
}  // namespace arith
}  // namespace theory
}  // namespace cvc5::internal

#endif /* CVC5__THEORY__STRINGS__PROOF_CHECKER_H */
