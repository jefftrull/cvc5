/******************************************************************************
 * Top contributors (to current version):
 *   Haniel Barbosa, Andrew Reynolds, Mathias Preiner
 *
 * This file is part of the cvc5 project.
 *
 * Copyright (c) 2009-2021 by the authors listed in the file AUTHORS
 * in the top-level source directory and their institutional affiliations.
 * All rights reserved.  See the file COPYING in the top-level source
 * directory for licensing information.
 * ****************************************************************************
 *
 * sygus_unif_rl
 */

#include "cvc5_private.h"

#ifndef CVC5__THEORY__QUANTIFIERS__SYGUS_UNIF_RL_H
#define CVC5__THEORY__QUANTIFIERS__SYGUS_UNIF_RL_H

#include <map>

#include "options/main_options.h"
#include "theory/quantifiers/lazy_trie.h"
#include "theory/quantifiers/sygus/sygus_unif.h"
#include "util/bool.h"

namespace cvc5::internal {
namespace theory {
namespace quantifiers {

using BoolNodePair = std::pair<bool, Node>;
using BoolNodePairHashFunction =
    PairHashFunction<bool, Node, BoolHashFunction, std::hash<Node>>;
using BoolNodePairMap =
    std::unordered_map<BoolNodePair, Node, BoolNodePairHashFunction>;
using NodePairMap = std::unordered_map<Node, Node>;
using NodePair = std::pair<Node, Node>;

class SynthConjecture;

/** Sygus unification Refinement Lemmas utility
 *
 * This class implement synthesis-by-unification, where the specification is a
 * set of refinement lemmas. With respect to SygusUnif, it's main interface
 * function is addExample, which adds a refinement lemma to the specification.
 */
class SygusUnifRl : public SygusUnif
{
 public:
  SygusUnifRl(Env& env, SynthConjecture* p);
  ~SygusUnifRl();

  /** initialize */
  void initializeCandidate(
      TermDbSygus* tds,
      Node f,
      std::vector<Node>& enums,
      std::map<Node, std::vector<Node>>& strategy_lemmas) override;

  /** Notify enumeration (unused) */
  void notifyEnumeration(Node e, Node v, std::vector<Node>& lemmas) override;
  /** Construct solution */
  bool constructSolution(std::vector<Node>& sols,
                         std::vector<Node>& lemmas) override;
  /** add refinement lemma
   *
   * This adds a lemma to the specification. It returns the purified form
   * of the lemma based on the method purifyLemma below. The method adds the
   * head of "evaluation points" to the map eval_hds, where an evaluation point
   * is a term of the form:
   *   ev( e1, c1...cn )
   * where ev is an evaluation function for the sygus deep embedding, e1 is of
   * sygus datatype type (the "head" of the evaluation point), and c1...cn are
   * constants. If ev( e1, c1...cn ) is the purified form of ev( e, t1...tn ),
   * then e1 is added to eval_hds[e]. We add evaluation points to eval_hds only
   * for those terms that are newly generated by this call (and have not been
   * returned by a previous call to this method).
   */
  Node addRefLemma(Node lemma, std::map<Node, std::vector<Node>>& eval_hds);
  /**
   * whether f is being synthesized with unification strategies. This can be
   * checked through wehether f has conditional or point enumerators (we use the
   * former)
   */
  bool usingUnif(Node f) const;
  /** get condition for evaluation point
   *
   * Returns the strategy point corresponding to the condition of the strategy
   * point e.
   */
  Node getConditionForEvaluationPoint(Node e) const;
  /** set conditional enumerators
   *
   * This informs this class that the current set of conditions for evaluation
   * point e are enumerated by "enums" and have values "conds"; "guard" is
   * Boolean variable whose semantics correspond to "there is a solution using
   * at most enums.size() conditions."
   */
  void setConditions(Node e,
                     Node guard,
                     const std::vector<Node>& enums,
                     const std::vector<Node>& conds);

  /** retrieve the head of evaluation points for candidate c, if any */
  std::vector<Node> getEvalPointHeads(Node c);

  /**
   * Whether we are using condition pool enumeration (Section 4 of Barbosa et al
   * FMCAD 2019). This is determined by option::sygusUnifPi().
   */
  bool usingConditionPool() const;
  /** Whether we are additionally using information gain.  */
  bool usingConditionPoolInfoGain() const;

 protected:
  /** reference to the parent conjecture */
  SynthConjecture* d_parent;
  /** Whether we are using condition pool enumeration */
  bool d_useCondPool;
  /** Whether we are additionally using information gain heuristics */
  bool d_useCondPoolIGain;
  /* Functions-to-synthesize (a.k.a. candidates) with unification strategies */
  std::unordered_set<Node> d_unif_candidates;
  /** construct sol */
  Node constructSol(Node f,
                    Node e,
                    NodeRole nrole,
                    int ind,
                    std::vector<Node>& lemmas) override;
  /** collects data from refinement lemmas to drive solution construction
   *
   * In particular it rebuilds d_app_to_pt whenever d_prev_rlemmas is different
   * from d_rlemmas, in which case we may have added or removed data points
   */
  void initializeConstructSol() override;
  /** initialize construction solution for function-to-synthesize f */
  void initializeConstructSolFor(Node f) override;
  /** maps unif functions-to~synhesize to their last built solutions */
  std::map<Node, Node> d_cand_to_sol;
  /*
    --------------------------------------------------------------
        Purification
    --------------------------------------------------------------
  */
  /**
   * maps heads of applications of a unif function-to-synthesize to their tuple
   * of arguments (which constitute a "data point" aka an "evaluation point")
   */
  std::map<Node, std::vector<Node>> d_hd_to_pt;
  /** maps unif candidates to heads of their evaluation points */
  std::map<Node, std::vector<Node>> d_cand_to_eval_hds;
  /**
   * maps applications of unif functions-to-synthesize to the result of their
   * purification */
  std::map<Node, Node> d_app_to_purified;
  /** maps unif functions-to-synthesize to counters of heads of evaluation
   * points */
  std::map<Node, unsigned> d_cand_to_hd_count;
  /**
   * This is called on the refinement lemma and will rewrite applications of
   * functions-to-synthesize to their respective purified form, i.e. such that
   * all unification functions are applied over concrete values. Moreover
   * unification functions are also rewritten such that every different tuple of
   * arguments has a fresh function symbol applied to it.
   *
   * Non-unification functions are also equated to their model values when they
   * occur as arguments of unification functions.
   *
   * A vector of guards with the (negated) equalities between the original
   * arguments and their model values is populated accordingly.
   *
   * When the traversal encounters an application of a unification
   * function-to-synthesize it will proceed to ensure that the arguments of that
   * function application are constants (ensureConst becomes "true"). If an
   * applicatin of a non-unif function-to-synthesize is reached, the requirement
   * is lifted (ensureConst becomes "false"). This avoides introducing spurious
   * equalities in model_guards.
   *
   * For example if "f" is being synthesized with a unification strategy and "g"
   * is not then the application
   *   f(g(f(g(0))))=1
   * would be purified into
   *   g(0) = c1 ^ g(f1(c1)) = c3 => f2(c3)
   *
   * Similarly
   *   f(+(0,f(g(0))))
   * would be purified into
   *   g(0) = c1 ^ f1(c1) = c2 => f2(+(0,c2))
   *
   * This function also populates the maps between candidates, heads and
   * evaluation points
   */
  Node purifyLemma(Node n,
                   bool ensureConst,
                   std::vector<Node>& model_guards,
                   BoolNodePairMap& cache);
  /*
    --------------------------------------------------------------
        Strategy information
    --------------------------------------------------------------
  */
  /**
   * This class stores the necessary information for building a decision tree
   * for a particular node in the strategy tree of a candidate variable f.
   */
  class DecisionTreeInfo
  {
   public:
    DecisionTreeInfo()
        : d_unif(nullptr), d_strategy(nullptr), d_strategy_index(0)
    {
    }
    ~DecisionTreeInfo() {}
    /** initializes this class */
    void initialize(Node cond_enum,
                    SygusUnifRl* unif,
                    SygusUnifStrategy* strategy,
                    unsigned strategy_index);
    /** returns index of strategy information of strategy node for this DT */
    unsigned getStrategyIndex() const;
    /** builds solution, if possible, using the given constructor
     *
     * A solution is possible when all different valued heads can be separated,
     * i.e. the current set of conditions separates them in a decision tree
     */
    Node buildSol(Node cons, std::vector<Node>& lemmas);
    /** bulids a solution by considering all condition values ever enumerated */
    Node buildSolAllCond(Node cons, std::vector<Node>& lemmas);
    /** builds a solution by incrementally adding points and conditions to DT
     *
     * Differently from the above method, here a condition is only added to the
     * DT when it's necessary for resolving a separation conflict (i.e. heads
     * with different values in the same leaf of the DT). Only one value per
     * condition enumerated is considered.
     *
     * If a solution cannot be built, then there are more conflicts to be
     * resolved than condition enumerators. A conflict lemma is added to lemmas
     * that forces a new assigment in which the conflict is removed (heads are
     * made equal) or a new condition is enumerated to try to separate them.
     */
    Node buildSolMinCond(Node cons, std::vector<Node>& lemmas);
    /** reference to parent unif util */
    SygusUnifRl* d_unif;
    /** enumerator template (if no templates, nodes in pair are Node::null()) */
    NodePair d_template;
    /** enumerated condition values, this is set by setConditions(...). */
    std::vector<Node> d_conds;
    /** gathered evaluation point heads */
    std::vector<Node> d_hds;
    /** all enumerated model values for conditions */
    std::unordered_set<Node> d_cond_mvs;
    /** get condition enumerator */
    Node getConditionEnumerator() const { return d_cond_enum; }
    /** set conditions */
    void setConditions(Node guard,
                       const std::vector<Node>& enums,
                       const std::vector<Node>& conds);

   private:
    /** true and false nodes */
    Node d_true;
    Node d_false;
    /** Accumulates solutions built when considering all enumerated condition
     * values (which may generate repeated solutions) */
    std::unordered_set<Node> d_sols;
    /**
     * Conditional enumerator variables corresponding to the condition values in
     * d_conds. These are used for generating separation lemmas during
     * buildSol. This is set by setConditions(...).
     */
    std::vector<Node> d_enums;
    /**
     * The guard literal whose semantics is "we need at most d_enums.size()
     * conditions in our solution. This is set by setConditions(...).
     */
    Node d_guard;
    /**
     * reference to inferred strategy for the function-to-synthesize this DT is
     * associated with
     */
    SygusUnifStrategy* d_strategy;
    /** index of strategy information of strategy node this DT is based on
     *
     * this is the index of the strategy (d_strats[index]) in the strategy node
     * to which this decision tree corresponds, which can be accessed through
     * the above strategy reference
     */
    unsigned d_strategy_index;
    /**
     * The enumerator in the strategy tree that stores conditions of the
     * decision tree.
     */
    Node d_cond_enum;
    /** extracts solution from decision tree built
     *
     * Depending on the active options, the decision tree might be rebuilt
     * before a solution is extracted, for example to optimize size (smaller
     * DTs) or chance of having a general solution (information gain heuristics)
     */
    Node extractSol(Node cons, std::map<Node, Node>& hd_mv);

    /** rebuild decision tree using information gain heuristic
     *
     * In a scenario in which the decision tree potentially contains more
     * conditions than necessary, it is beneficial to rebuild it in a way that
     * "better" conditions occurr closer to the top.
     *
     * The information gain heuristic selects conditions that lead to a
     * greater reduction of the Shannon entropy in the set of points
     */
    void recomputeSolHeuristically(std::map<Node, Node>& hd_mv);
    /** recursively select (best) conditions to split heads
     *
     * At each call picks the best condition based on the information gain
     * heuristic and splits the set of heads accordingly, then recurses on
     * them.
     *
     * The base case is a set being fully classified (i.e. all heads have the
     * same value)
     *
     * hds is the set of evaluation point heads we must classify with the
     * values in conds. The classification is guided by how a condition value
     * splits the heads through its evaluation on the points associated with
     * the heads. The metric is based on the model values of the heads (hd_mv)
     * in the resulting splits.
     *
     * ind is the current level of indentation (for debugging)
     */
    void buildDtInfoGain(std::vector<Node>& hds,
                         std::vector<Node> conds,
                         std::map<Node, Node>& hd_mv,
                         int ind);
    /** computes the Shannon entropy of a set of heads
     *
     * The entropy depends on how many positive and negative heads are in the
     * set and in their distribution. The polarity of the evaluation heads is
     * queried from their model values in hd_mv.
     *
     * ind is the current level of indentation (for debugging)
     */
    double getEntropy(const std::vector<Node>& hds,
                      std::map<Node, Node>& hd_mv,
                      int ind);
    /** evaluates a condition on a set of points
     *
     * The result is two sets of points: those on which the condition holds
     * and those on which it does not
     */
    std::pair<std::vector<Node>, std::vector<Node>> evaluateCond(
        std::vector<Node>& pts, Node cond);
    /** Classifies evaluation points according to enumerated condition values
     *
     * Maintains the invariant that points evaluated in the same way in the
     * current condition values are kept in the same "separation class."
     */
    class PointSeparator : public LazyTrieEvaluator
    {
     public:
      PointSeparator() : d_dt(nullptr) {}
      /** initializes this class */
      void initialize(DecisionTreeInfo* dt);
      /**
       * evaluates the respective evaluation point of the head n on the index-th
       * condition
       */
      Node evaluate(Node n, unsigned index) override;

      /** the lazy trie for building the separation classes */
      LazyTrieMulti d_trie;
      /** extracts solution from decision tree built */
      Node extractSol(Node cons, std::map<Node, Node>& hd_mv);
      /** computes the result of applying cond on the respective point of hd
       *
       * If for example cond is (\lambda xy. x < y) and hd is an evaluation head
       * in point (hd 0 1) this function will result in true, since
       *   (\lambda xy. x < y) 0 1 evaluates to true
       */
      Node computeCond(Node cond, Node hd);

     private:
      /** reference to parent unif util */
      DecisionTreeInfo* d_dt;
      /** cache of conditions evaluations on heads
       *
       * If for example cond is (\lambda xy. x < y) and hd is an evaluation head
       * in point (hd 0 1), then after invoking computeCond(cond, hd) this map
       * will contain d_eval_cond_hd[<cond, hd>] = true, since
       *
       *   (\lambda xy. x < y) 0 1 evaluates to true
       */
      std::map<std::pair<Node, Node>, Node> d_eval_cond_hd;
    };
    /**
     * Utility for determining how evaluation points are separated by currently
     * enumerated condiotion values
     */
    PointSeparator d_pt_sep;
  };
  /** maps strategy points in the strategy tree to the above data */
  std::map<Node, DecisionTreeInfo> d_stratpt_to_dt;
  /** maps conditional enumerators to strategy points in which they occur */
  std::map<Node, std::vector<Node>> d_cenum_to_stratpt;
  /** maps unif candidates to their conditional enumerators */
  std::map<Node, std::vector<Node>> d_cand_cenums;
  /** all conditional enumerators */
  std::vector<Node> d_cond_enums;
  /** register strategy
   *
   * Initialize the above data for the relevant enumerators in the strategy tree
   * of candidate variable f. For each strategy point e which there is a
   * decision tree strategy, we add e to enums. For each strategy with index
   * i in an strategy point e, if we are not using the strategy, we add i to
   * unused_strats[e]. This map is later passed to
   * SygusUnifStrategy::staticLearnRedundantOps.
   */
  void registerStrategy(
      Node f,
      std::vector<Node>& enums,
      std::map<Node, std::unordered_set<unsigned>>& unused_strats);
  /** register strategy node
   *
   * Called while traversing the strategy tree of f. The arguments e and nrole
   * indicate the current node in the tree we are traversing, and visited
   * indicates the nodes we have already visited. The arguments enums and
   * unused_strats are modified as described above.
   */
  void registerStrategyNode(
      Node f,
      Node e,
      NodeRole nrole,
      std::map<Node, std::map<NodeRole, bool>>& visited,
      std::vector<Node>& enums,
      std::map<Node, std::unordered_set<unsigned>>& unused_strats);
  /** register conditional enumerator
   *
   * Registers that cond is a conditional enumerator for building a (recursive)
   * decision tree at strategy node e within the strategy tree of f.
   */
  void registerConditionalEnumerator(Node f,
                                     Node e,
                                     Node cond,
                                     unsigned strategy_index);
};

}  // namespace quantifiers
}  // namespace theory
}  // namespace cvc5::internal

#endif /* CVC5__THEORY__QUANTIFIERS__SYGUS_UNIF_RL_H */
