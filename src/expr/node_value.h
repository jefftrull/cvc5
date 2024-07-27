/******************************************************************************
 * Top contributors (to current version):
 *   Morgan Deters, Aina Niemetz, Andres Noetzli
 *
 * This file is part of the cvc5 project.
 *
 * Copyright (c) 2009-2024 by the authors listed in the file AUTHORS
 * in the top-level source directory and their institutional affiliations.
 * All rights reserved.  See the file COPYING in the top-level source
 * directory for licensing information.
 * ****************************************************************************
 *
 * A node value.
 *
 * The actual node implementation.
 * Instances of this class are generally referenced through cvc5::internal::Node
 * rather than by pointer. Note that cvc5::internal::Node maintains the
 * reference count on NodeValue instances.
 */

#include "cvc5_private.h"

#ifndef CVC5__EXPR__NODE_VALUE_H
#define CVC5__EXPR__NODE_VALUE_H

#include <array>
#include <iterator>
#include <string>
#include <vector>

#include "expr/kind.h"
#include "expr/metakind.h"
#include "options/language.h"

namespace cvc5::internal {

template <bool ref_count> class NodeTemplate;
class TypeNode;
class NodeBuilder;
class NodeManager;

namespace expr {
  class NodeValue;
}

namespace kind {
  namespace metakind {

  template <cvc5::internal::Kind k, class T, bool pool>
  struct NodeValueConstCompare;

  struct NodeValueCompare;

  }  // namespace metakind
  }  // namespace kind

namespace expr {

/**
 * This is a NodeValue, a pure virtual base for implementations of different sizes and holding different things
 */
class CVC5_EXPORT NodeValue
{
  template <bool>
  friend class cvc5::internal::NodeTemplate;
  friend class cvc5::internal::TypeNode;
  friend class cvc5::internal::NodeBuilder;
  friend class cvc5::internal::NodeManager;

  template <Kind k, class T, bool pool>
  friend struct kind::metakind::NodeValueConstCompare;

  friend struct kind::metakind::NodeValueCompare;

  friend void kind::metakind::nodeValueConstantToStream(std::ostream& out,
                                                        const NodeValue* nv);
  friend void kind::metakind::deleteNodeValueConstant(NodeValue* nv);

  friend class RefCountGuard;

  /* ------------------------------------------------------------------------ */
 public:
  /* ------------------------------------------------------------------------ */

  virtual ~NodeValue() = 0;

  using nv_iterator = NodeValue**;
  using const_nv_iterator = NodeValue const* const*;

  template <class T>
  class iterator
  {
   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using reference = T;

    iterator() : d_i(nullptr) {}
    explicit iterator(const_nv_iterator i) : d_i(i) {}

    /** Conversion of a TNode iterator to a Node iterator. */
    inline operator NodeValue::iterator<NodeTemplate<true> >()
    {
      return iterator<NodeTemplate<true> >(d_i);
    }

    T operator*() const { return T(*d_i); }

    bool operator==(const iterator& i) const { return d_i == i.d_i; }

    bool operator!=(const iterator& i) const { return d_i != i.d_i; }

    iterator& operator++()
    {
      ++d_i;
      return *this;
    }

    iterator operator++(int) { return iterator(d_i++); }

    iterator& operator--()
    {
      --d_i;
      return *this;
    }

    iterator operator--(int) { return iterator(d_i--); }

    iterator& operator+=(difference_type p)
    {
      d_i += p;
      return *this;
    }

    iterator& operator-=(difference_type p)
    {
      d_i -= p;
      return *this;
    }

    iterator operator+(difference_type p) { return iterator(d_i + p); }

    iterator operator-(difference_type p) { return iterator(d_i - p); }

    difference_type operator-(iterator i) { return d_i - i.d_i; }

   private:
    const_nv_iterator d_i;

  }; /* class NodeValue::iterator<T> */

  friend inline void intrusive_ptr_add_ref(NodeValue* nv) {
    nv->inc();
  }

  friend inline void intrusive_ptr_release(NodeValue* nv) {
    nv->dec();
  }

  uint64_t getId() const { return d_id; }

  Kind getKind() const { return dKindToKind(d_kind); }

  kind::MetaKind getMetaKind() const { return kind::metaKindOf(getKind()); }

  // Derived classes must provide contiguous storage for NodeValue* children
  // Sometimes other things are stored there :(
  virtual NodeValue* const * getChildEntries() const = 0;
  virtual NodeValue* * getChildEntries() = 0;
  virtual uint32_t getNumChildEntries() const = 0;  // amount of storage
  virtual NodeValue* getChildEntry(int i) const = 0;

  inline uint32_t getNumChildren() const
  {
    return (getMetaKind() == kind::metakind::PARAMETERIZED) ? getNumChildEntries() - 1
                                                            : getNumChildEntries();
  }

  /* ------------------------------ Header ---------------------------------- */
  /** Number of bits reserved for reference counting. */
  static constexpr uint32_t NBITS_REFCOUNT = 20;
  /** Number of bits reserved for node kind. */
  static constexpr uint32_t NBITS_KIND = 10;
  /** Number of bits reserved for node id. */
  static constexpr uint32_t NBITS_ID = 40;
  /** Number of bits reserved for number of children. */
  static const uint32_t NBITS_NCHILDREN = 26;
  static_assert(NBITS_REFCOUNT + NBITS_KIND + NBITS_ID + NBITS_NCHILDREN == 96,
                "NodeValue header bit assignment does not sum to 96 !");
  /* ------------------- This header fits into 96 bits ---------------------- */

  /** Maximum number of children possible. */
  static constexpr uint32_t MAX_CHILDREN =
      (static_cast<uint32_t>(1) << NBITS_NCHILDREN) - 1;

  uint32_t getRefCount() const { return d_rc; }

  NodeValue* getOperator() const;
  inline NodeValue* getChild(int i) const;


  /** If this is a CONST_* Node, extract the constant from it.  */
  template <class T>
  const T& getConst() const;

  static inline NodeValue& null();

  /**
   * Hash this NodeValue.  For hash_maps, hash_sets, etc.. but this is
   * for expr package internal use only at present!  This is likely to
   * be POORLY PERFORMING for other uses!  For example, this gives
   * collisions for all VARIABLEs.
   * @return the hash value of this expression.
   */
  size_t poolHash() const
  {
    if (getMetaKind() == kind::metakind::CONSTANT)
    {
      return kind::metakind::NodeValueCompare::constHash(this);
    }

    size_t hash = d_kind;
    const_nv_iterator i = nv_begin();
    const_nv_iterator i_end = nv_end();
    while (i != i_end)
    {
      hash ^= (*i)->d_id + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      ++i;
    }
    return hash;
  }

  static inline uint32_t kindToDKind(Kind k)
  {
    return ((uint32_t)k) & kindMask;
  }

  static inline Kind dKindToKind(uint32_t d)
  {
    return (d == kindMask) ? Kind::UNDEFINED_KIND : Kind(d);
  }

  std::string toString() const;

  void toStream(std::ostream& out) const;

  void printAst(std::ostream& out, int indent = 0) const;

  template <typename T>
  inline iterator<T> begin() const;
  template <typename T>
  inline iterator<T> end() const;

  /* ------------------------------------------------------------------------ */
 private:
  /* ------------------------------------------------------------------------ */

  /**
   * RAII guard that increases the reference count if the reference count to be
   * > 0.  Otherwise, this does nothing. This does not just increment the
   * reference count to avoid maxing out the d_rc field. This is only for low
   * level functions.
   */
  class RefCountGuard
  {
   public:
    RefCountGuard(const NodeValue* nv) : d_nv(const_cast<NodeValue*>(nv))
    {
      d_increased = (d_nv->d_rc == 0);
      if (d_increased)
      {
        d_nv->d_rc = 1;
      }
    }
    ~RefCountGuard()
    {
      // dec() without marking for deletion: we don't want to garbage
      // collect this NodeValue if ours is the last reference to it.
      // E.g., this can happen when debugging code calls the print
      // routines below.  As RefCountGuards are scoped on the stack,
      // this should be fine---but not in multithreaded contexts!
      if (d_increased)
      {
        --d_nv->d_rc;
      }
    }

   private:
    NodeValue* d_nv;
    bool d_increased;
  }; /* NodeValue::RefCountGuard */

  /** A mask for d_kind */
  static constexpr uint32_t kindMask =
      (static_cast<uint32_t>(1) << NBITS_KIND) - 1;

protected:
  /** Maximum reference count possible.  Used for sticky
   *  reference-counting.  Should be (1 << num_bits(d_rc)) - 1 */
  static constexpr uint32_t MAX_RC =
      (static_cast<uint32_t>(1) << NBITS_REFCOUNT) - 1;

  /** Uninitializing constructor for NodeBuilder's use.  */
  NodeValue()
  { /* do not initialize! */
  }
  /** Private constructor for initializing the base class members to default values */
  NodeValue(int);

  /** Private constructor for specific values of base class members */
  NodeValue(uint64_t id, uint32_t rc, uint32_t kind);


private:
  void inc()
  {
    if (__builtin_expect((d_rc < MAX_RC - 1), true))
    {
      ++d_rc;
    }
    else if (__builtin_expect((d_rc == MAX_RC - 1), false))
    {
      ++d_rc;
      markRefCountMaxedOut();
    }
  }

  void dec()
  {
    // FIXME multithreading
    if (__builtin_expect((d_rc < MAX_RC), true))
    {
      --d_rc;
      if (__builtin_expect((d_rc == 0), false))
      {
        markForDeletion();
      }
    }
  }

  void markRefCountMaxedOut();
  void markForDeletion();

  /** Decrement ref counts of children */
  inline void decrRefCounts();

  /** Returns true if the reference count is maximized. */
  inline bool HasMaximizedReferenceCount() { return d_rc == MAX_RC; }

  nv_iterator nv_begin();
  nv_iterator nv_end();

  const_nv_iterator nv_begin() const;
  const_nv_iterator nv_end() const;

  /**
   * Indents the given stream a given amount of spaces.
   * @param out the stream to indent
   * @param indent the numer of spaces
   */
  static inline void indent(std::ostream& out, int indent)
  {
    for (int i = 0; i < indent; i++)
    {
      out << ' ';
    }
  }

  /** The ID (0 is reserved for the null value) */
  uint64_t d_id : NBITS_ID;

  /** The expression's reference count. */
  uint32_t d_rc : NBITS_REFCOUNT;

  /** Kind of the expression */
  uint32_t d_kind : NBITS_KIND;
}; /* class NodeValue */

class CVC5_EXPORT NodeValueClassic : public NodeValue {

  friend class cvc5::internal::NodeBuilder;
  friend class cvc5::internal::NodeManager;

  // define required member functions

  inline NodeValue* const * getChildEntries() const final;
  inline NodeValue* * getChildEntries() final;
  inline uint32_t getNumChildEntries() const final;
  inline NodeValue* getChildEntry(int i) const final;

private:
  friend class NodeValue;

  /** Uninitializing constructor for NodeBuilder's use.  */
  NodeValueClassic()
  { /* do not initialize! */
  }

  /** Private constructor for the null value. */
  NodeValueClassic(int);

  /** Number of children */
  uint32_t d_nchildren : NBITS_NCHILDREN;

  /** Variable number of child nodes */
  NodeValue* d_children[0];
};

template<uint32_t NChildren>
class CVC5_EXPORT NodeValueFixed : public NodeValue {

  static_assert(NChildren <= MAX_CHILDREN);

  friend class cvc5::internal::NodeBuilder;

  // define required member functions

  inline NodeValue* const * getChildEntries() const final;
  inline NodeValue* * getChildEntries() final;
  inline uint32_t getNumChildEntries() const final;
  inline NodeValue* getChildEntry(int i) const final;

private:
  friend class NodeValue;

  template<typename... NVPtr>
  NodeValueFixed(uint64_t id, uint32_t rc, uint32_t kind,
                 NVPtr... children);

  /** Helper functions for creating NodeValueFixed objects */
  template<std::size_t... Is>
  friend NodeValueFixed<sizeof...(Is)>*
  mkNodeValueFixedImpl(uint64_t id, uint32_t rc, uint32_t kind,
                       std::index_sequence<Is...>, NodeValue** start);

  /** Fixed number of child nodes */
  std::array<NodeValue*, NChildren> d_children;
};

class CVC5_EXPORT NodeValueVariable : public NodeValue {

  friend class cvc5::internal::NodeBuilder;

  // define required member functions

  inline NodeValue* const * getChildEntries() const final;
  inline NodeValue* * getChildEntries() final;
  inline uint32_t getNumChildEntries() const final;
  inline NodeValue* getChildEntry(int i) const final;

private:
  friend class NodeValue;

  template<typename NVIter>
  NodeValueVariable(uint64_t id, uint32_t rc, uint32_t kind,
                    NVIter child_start, NVIter child_end);

  /** Variable number of child nodes */
  std::vector<NodeValue*> d_children;
};

template<typename T>
class CVC5_EXPORT NodeValueConst : public NodeValue {

  friend class cvc5::internal::NodeManager;

  // define required member functions

public:
  inline NodeValue* const * getChildEntries() const final;
  inline NodeValue* * getChildEntries() final;
  inline uint32_t getNumChildEntries() const final;
  inline NodeValue* getChildEntry(int i) const final;

  const T& getConst() const;

private:
  friend class NodeValue;

  NodeValueConst(uint64_t id, uint32_t rc, uint32_t kind, T val);

  T d_val;
};

inline NodeValue& NodeValue::null()
{
  static NodeValue* s_null = new NodeValueClassic(0);
  return *s_null;
}

/**
 * Provides a symmetric addition operator to that already defined in
 * the iterator class.
 */
NodeValue::iterator<NodeTemplate<true> > operator+(
    NodeValue::iterator<NodeTemplate<true> >::difference_type p,
    NodeValue::iterator<NodeTemplate<true> > i);

/**
 * Provides a symmetric addition operator to that already defined in
 * the iterator class.
 */
NodeValue::iterator<NodeTemplate<false> > operator+(
    NodeValue::iterator<NodeTemplate<false> >::difference_type p,
    NodeValue::iterator<NodeTemplate<false> > i);

/**
 * For hash_maps, hash_sets, etc.. but this is for expr package
 * internal use only at present!  This is likely to be POORLY
 * PERFORMING for other uses!  NodeValue::poolHash() will lead to
 * collisions for all VARIABLEs.
 */
struct NodeValuePoolHashFunction {
  inline size_t operator()(const NodeValue* nv) const {
    return (size_t) nv->poolHash();
  }
};/* struct NodeValuePoolHashFunction */

/**
 * For hash_maps, hash_sets, etc.
 */
struct NodeValueIDHashFunction {
  inline size_t operator()(const NodeValue* nv) const {
    return (size_t) nv->getId();
  }
};/* struct NodeValueIDHashFunction */


/**
 * An equality predicate that is applicable between pointers to fully
 * constructed NodeValues.
 */
struct NodeValueIDEquality {
  inline bool operator()(const NodeValue* a, const NodeValue* b) const {
    return a->getId() == b->getId();
  }
};

std::ostream& operator<<(std::ostream& out, const NodeValue& nv);

inline NodeValueClassic::NodeValueClassic(int)
  : NodeValue(0), d_nchildren(0)
{
}

inline void NodeValue::decrRefCounts() {
  for(nv_iterator i = nv_begin(); i != nv_end(); ++i) {
    (*i)->dec();
  }
}

inline NodeValue::nv_iterator NodeValue::nv_begin() {
  return getChildEntries();
}

inline NodeValue::nv_iterator NodeValue::nv_end() {
  return getChildEntries() + getNumChildEntries();
}

inline NodeValue::const_nv_iterator NodeValue::nv_begin() const {
  return getChildEntries();
}

inline NodeValue::const_nv_iterator NodeValue::nv_end() const {
  return getChildEntries() + getNumChildEntries();
}

template <typename T>
inline NodeValue::iterator<T> NodeValue::begin() const {
  NodeValue* const* firstChild = getChildEntries();
  if(getMetaKind() == kind::metakind::PARAMETERIZED) {
    ++firstChild;
  }
  return iterator<T>(firstChild);
}

template <typename T>
inline NodeValue::iterator<T> NodeValue::end() const {
  return iterator<T>(getChildEntries() + getNumChildEntries());
}

inline NodeValue* NodeValue::getOperator() const {
  Assert(getMetaKind() == kind::metakind::PARAMETERIZED);
  return getChildEntry(0);
}

inline NodeValue* const * NodeValueClassic::getChildEntries() const {
  return d_children;
}

inline NodeValue* * NodeValueClassic::getChildEntries() {
  return d_children;
}

inline uint32_t NodeValueClassic::getNumChildEntries() const {
  return d_nchildren;
}

inline NodeValue* NodeValueClassic::getChildEntry(int i) const {
  return d_children[i];
}

inline NodeValue* NodeValue::getChild(int i) const {
  if(getMetaKind() == kind::metakind::PARAMETERIZED) {
    ++i;
  }

  Assert(i >= 0 && unsigned(i) < getNumChildEntries());
  return getChildEntry(i);
}

template<uint32_t NChildren>
template<typename... NVPtr>
NodeValueFixed<NChildren>::NodeValueFixed(uint64_t id, uint32_t rc, uint32_t kind,
                                          NVPtr... children)
  : NodeValue(id, rc, kind), d_children{children...}
{
}

template<uint32_t NChildren>
inline NodeValue* const * NodeValueFixed<NChildren>::getChildEntries() const {
  return d_children.data();
}

template<uint32_t NChildren>
inline NodeValue* * NodeValueFixed<NChildren>::getChildEntries() {
  return d_children.data();
}

template<uint32_t NChildren>
inline uint32_t NodeValueFixed<NChildren>::getNumChildEntries() const {
  return NChildren;
}

template<uint32_t NChildren>
inline NodeValue* NodeValueFixed<NChildren>::getChildEntry(int i) const {
  return d_children[i];
}

// helper function for creating NodeValueFixed from a pointer and a count
template<std::size_t... Is>
inline NodeValueFixed<sizeof...(Is)>*
mkNodeValueFixedImpl(uint64_t id, uint32_t rc, uint32_t kind,
                     std::index_sequence<Is...>, NodeValue** start) {
  return new NodeValueFixed<sizeof...(Is)>(id, rc, kind, start[Is]...);
}

template<std::size_t N>
inline NodeValueFixed<N>*
mkNodeValueFixed(uint64_t id, uint32_t rc, uint32_t kind,
                 NodeValue** start) {
  return mkNodeValueFixedImpl(id, rc, kind, std::make_index_sequence<N>(), start);
}

// NodeValueVariable member functions
template<typename NVIter>
NodeValueVariable::NodeValueVariable(uint64_t id, uint32_t rc, uint32_t kind,
                                                NVIter children_start, NVIter children_end)
  : NodeValue(id, rc, kind), d_children(children_start, children_end)
{
}

inline NodeValue* const * NodeValueVariable::getChildEntries() const {
  return d_children.data();
}

inline NodeValue* * NodeValueVariable::getChildEntries() {
  return d_children.data();
}

inline uint32_t NodeValueVariable::getNumChildEntries() const {
  return d_children.size();
}

inline NodeValue* NodeValueVariable::getChildEntry(int i) const {
  return d_children[i];
}

// NodeValueConst member functions
template<typename T>
NodeValueConst<T>::NodeValueConst(uint64_t id, uint32_t rc, uint32_t kind, T val)
  : NodeValue(id, rc, kind), d_val(std::move(val))
{
}

template<typename T>
inline NodeValue* const * NodeValueConst<T>::getChildEntries() const {
  return nullptr;   // Is this OK?
}

template<typename T>
inline NodeValue* * NodeValueConst<T>::getChildEntries() {
  return nullptr;
}

template<typename T>
inline uint32_t NodeValueConst<T>::getNumChildEntries() const {
  return 0;
}

template<typename T>
inline NodeValue* NodeValueConst<T>::getChildEntry(int i) const {
  Assert(false);
  return nullptr;
}

template<typename T>
inline const T& NodeValueConst<T>::getConst() const {
  return d_val;
}


}  // namespace expr
}  // namespace cvc5::internal

#endif /* CVC5__EXPR__NODE_VALUE_H */
