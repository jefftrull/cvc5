/******************************************************************************
 * Top contributors (to current version):
 *   Morgan Deters, Tim King, Andres Noetzli
 *
 * This file is part of the cvc5 project.
 *
 * Copyright (c) 2009-2024 by the authors listed in the file AUTHORS
 * in the top-level source directory and their institutional affiliations.
 * All rights reserved.  See the file COPYING in the top-level source
 * directory for licensing information.
 * ****************************************************************************
 *
 * Node attributes' internals.
 */

#include <algorithm>
#include <numeric>

#include "cvc5_private.h"

#ifndef CVC5_ATTRIBUTE_H__INCLUDING__ATTRIBUTE_INTERNALS_H
#  error expr/attribute_internals.h should only be included by expr/attribute.h
#endif /* CVC5_ATTRIBUTE_H__INCLUDING__ATTRIBUTE_INTERNALS_H */

#ifndef CVC5__EXPR__ATTRIBUTE_INTERNALS_H
#define CVC5__EXPR__ATTRIBUTE_INTERNALS_H

#include <unordered_map>

namespace cvc5::internal {
namespace expr {

// ATTRIBUTE HASH FUNCTIONS ====================================================

namespace attr {

/**
 * A hash function for attribute table keys.  Attribute table keys are
 * pairs, (unique-attribute-id, Node).
 */
struct AttrHashFunction {
  enum { LARGE_PRIME = 32452843ul };
  std::size_t operator()(const std::pair<uint64_t, NodeValue*>& p) const {
    return p.first * LARGE_PRIME + p.second->getId();
  }
};/* struct AttrHashFunction */

/**
 * A hash function for boolean-valued attribute table keys; here we
 * don't have to store a pair as the key, because we use a known bit
 * in [0..63] for each attribute
 */
struct AttrBoolHashFunction {
  std::size_t operator()(NodeValue* nv) const {
    return (size_t)nv->getId();
  }
};/* struct AttrBoolHashFunction */

}  // namespace attr

// ATTRIBUTE TYPE MAPPINGS =====================================================

namespace attr {

/**
 * KindValueToTableValueMapping is a compile-time-only mechanism to
 * convert "attribute value types" into "table value types" and back
 * again.
 *
 * Each instantiation < T > is expected to have three members:
 *
 *   typename table_value_type
 *
 *      A type representing the underlying table's value_type for
 *      attribute value type (T).  It may be different from T, e.g. T
 *      could be a pointer-to-Foo, but the underlying table value_type
 *      might be pointer-to-void.
 *
 *   static [convertible-to-table_value_type] convert([convertible-from-T])
 *
 *      Converts a T into a table_value_type.  Used to convert an
 *      attribute value on setting it (and assigning it into the
 *      underlying table).  See notes on specializations of
 *      KindValueToTableValueMapping, below.
 *
 *   static [convertible-to-T] convertBack([convertible-from-table_value_type])
 *
 *      Converts a table_value_type back into a T.  Used to convert an
 *      underlying table value back into the attribute's expected type
 *      when retrieving it from the table.  See notes on
 *      specializations of KindValueToTableValueMapping, below.
 *
 * This general (non-specialized) implementation of the template does
 * nothing.
 *
 * The `Enable` template parameter is used to instantiate the template
 * conditionally: If the template substitution of Enable fails (e.g. when using
 * `std::enable_if` in the template parameter and the condition is false), the
 * instantiation is ignored due to the SFINAE rule.
 */
template <class T, class Enable = void>
struct KindValueToTableValueMapping
{
  /** Simple case: T == table_value_type */
  typedef T table_value_type;
  /** No conversion necessary */
  inline static T convert(const T& t) { return t; }
  /** No conversion necessary */
  inline static T convertBack(const T& t) { return t; }
};

/**
 * This converts arbitrary unsigned integers (up to 64-bit) to and from 64-bit
 * integers s.t. they can be stored in the hash table for integral-valued
 * attributes.
 */
template <class T>
struct KindValueToTableValueMapping<
    T,
    // Use this specialization only for unsigned integers
    typename std::enable_if<std::is_unsigned<T>::value>::type>
{
  typedef uint64_t table_value_type;
  /** Convert from unsigned integer to uint64_t */
  static uint64_t convert(const T& t)
  {
    static_assert(sizeof(T) <= sizeof(uint64_t),
                  "Cannot store integer attributes of a bit-width that is "
                  "greater than 64-bits");
    return static_cast<uint64_t>(t);
  }
  /** Convert from uint64_t to unsigned integer */
  static T convertBack(const uint64_t& t) { return static_cast<T>(t); }
};

}  // namespace attr

// ATTRIBUTE HASH TABLES =======================================================

namespace attr {

// Returns a 64 bit integer with a single `bit` set when `bit` < 64.
// Avoids problems in (1 << x) when sizeof(x) <= sizeof(uint64_t).
inline uint64_t GetBitSet(uint64_t bit)
{
  constexpr uint64_t kOne = 1;
  return kOne << bit;
}

/**
 * An "AttrHash<V>"---the hash table underlying
 * attributes---is a mapping of pair<unique-attribute-id, Node>
 * to V using a two-level hash+flat_map structure. The top level
 * uses NodeValue* as its key, allowing rapid deletion of matching
 * collections of entries, while the second level, keyed on Ids
 * and implemented with a sorted vector, optimizes for size and
 * speed for small collections.
 */
template <class V>
class AttrHash
{

  // Second level flat map uint64_t -> V
  struct IdMap {
    typedef std::pair<uint64_t, V> value_type;
    typedef std::vector<value_type> Container;
    typedef typename Container::iterator iterator;
    typedef typename Container::const_iterator const_iterator;

    // only the methods required by AttrHash<V>:

    const_iterator begin() const { return d_contents.begin(); }
    const_iterator end() const { return d_contents.end(); }

    iterator begin() { return d_contents.begin(); }
    iterator end() { return d_contents.end(); }

    std::size_t size() const { return d_contents.size(); }

    void reserve(std::size_t sz) { d_contents.reserve(sz); }

    std::pair<iterator, bool> emplace(uint64_t k, V v) {
      auto p = std::make_pair(k, std::move(v));
      auto range = std::equal_range(d_contents.begin(), d_contents.end(),
                                    p,
                                    [](const value_type & a, const value_type & b) {
                                      return a.first < b.first;
                                    });
      if (range.first != range.second) {
        // key already present, don't insert
        return std::make_pair(iterator{}, false);
      }

      return std::make_pair(d_contents.insert(range.first, std::move(p)), true);
    }

    const_iterator find(uint64_t key) const {
      auto range = std::equal_range(d_contents.begin(), d_contents.end(),
                                    std::make_pair(key, V{}),
                                    [](const value_type & a, const value_type & b) {
                                      return a.first < b.first;
                                    });
      if (range.first == range.second) {
        // not in map
        return d_contents.end();
      } else {
        return range.first;
      }
    }

    iterator find(uint64_t key) {
      auto range = std::equal_range(d_contents.begin(), d_contents.end(),
                                    std::make_pair(key, V{}),
                                    [](const value_type & a, const value_type & b) {
                                      return a.first < b.first;
                                    });
      if (range.first == range.second) {
        return d_contents.end();
      } else {
        return range.first;
      }
    }

    iterator erase(iterator pos) {
      return d_contents.erase(pos);
    }

    V & operator[](uint64_t key) {
      auto it = std::lower_bound(d_contents.begin(), d_contents.end(),
                                 std::make_pair(key, V{}),
                                 [](const value_type & a, const value_type & b) {
                                   return a.first < b.first;
                                 });
      if ((it == d_contents.end()) || (it->first != key)) {
        // not in map
        it = d_contents.insert(it, std::make_pair(key, V{}));
      }
      return (*it).second;
    }

    // range insert
    template<typename Iter>
    void insert(Iter beg, Iter end) {
      for (Iter it = beg; it != end; ++it) {
        auto found_it = std::lower_bound(d_contents.begin(), d_contents.end(),
                                         it->first,
                                         [](const value_type & a, const value_type & b) {
                                           return a.first < b.first;
                                         });
        if ((found_it != d_contents.end()) && (it->first == found_it->first)) {
          // this key is already present in the map. replace it:
          found_it->second = it->second;
        } else {
          d_contents.insert(found_it, *it);
        }
      }
    }

  private:
    Container d_contents;
  };

  typedef std::unordered_map<NodeValue*, IdMap, AttrBoolHashFunction> Storage;

  Storage d_storage;

public:

  template<typename Parent, typename L1It, typename L2It>
  struct Iterator {
    // requirements for ForwardIterator
    typedef std::forward_iterator_tag iterator_category;
    typedef std::pair<std::pair<uint64_t, NodeValue*>, V> value_type;
    typedef value_type reference;    // we don't supply a true reference
    typedef value_type* pointer;
    typedef std::ptrdiff_t difference_type;

    // default constructor
    Iterator() : d_atEnd{true} {}

    Iterator(Parent * parent) : d_parent{parent}, d_l1It(parent->d_storage.begin()) {
      d_atEnd = (d_l1It == parent->d_storage.end());
      if (!d_atEnd) {
        d_l2It = d_l1It->second.begin();
        legalize();   // L2 map may be empty
      }
    }

    // prerequisite: l1_it and l2_it are valid iterators
    Iterator(Parent * parent, L1It l1_it, L2It l2_it)
      : d_atEnd(l1_it == parent->d_storage.end()), d_parent(parent), d_l1It(l1_it), d_l2It(l2_it) {}

    // increment
    Iterator & operator++() {  // pre
      increment();
      return *this;
    }

    Iterator operator++(int) {  // post
      Iterator tmp = *this;
      increment();
      return tmp;
    }

    // dereference
    value_type operator*() const {
      return std::make_pair(std::make_pair(d_l2It->first, d_l1It->first),
                            d_l2It->second);
    }

    // comparison
    bool operator==(Iterator const & other) const {
      return (d_atEnd && other.d_atEnd) ||
        (!d_atEnd && !other.d_atEnd &&
         (d_l1It == other.d_l1It) && (d_l2It == other.d_l2It));
    }
    bool operator!=(Iterator const & other) const { return !(*this == other); }

    // this is kinda weird... ideally would be private but a friend of AttrHash
    Iterator erase() {
      auto result = *this;
      result.increment();

      d_l1It->second.erase(d_l2It);

      return result;
    }

  private:

    bool d_atEnd;
    Parent* d_parent;
    L1It d_l1It;
    L2It d_l2It;

    void increment() {
      if (d_atEnd)
        return;

      ++d_l2It;

      legalize();
    }

    void legalize() {
      // move forward to next valid entry
      while (d_l2It == d_l1It->second.end()) {
        ++d_l1It;
        if (d_l1It == d_parent->d_storage.end()) {
          d_atEnd = true;
          return;
        }
        d_l2It = d_l1It->second.begin();
      }
    }
  };

  typedef Iterator<AttrHash<V>, typename Storage::iterator, typename IdMap::iterator> iterator;
  typedef Iterator<const AttrHash<V>, typename Storage::const_iterator, typename IdMap::const_iterator> const_iterator;

  std::size_t size() const {
    return std::accumulate(d_storage.begin(), d_storage.end(), 0u,
                           [](std::size_t sum, auto const & l2) {
                             return sum + l2.second.size();
                           });
  }

  auto begin() { return iterator(this); }
  auto end() { return iterator(); }

  auto begin() const { return const_iterator(const_cast<AttrHash<V>*>(this)); }
  auto end() const { return const_iterator(); }

  iterator erase(iterator it) {
    return it.erase();
  }

  std::size_t erase(std::pair<uint64_t, NodeValue*> p) {
    typename Storage::iterator it1 = d_storage.find(p.second);
    if (it1 == d_storage.end())
      return 0;

    typename IdMap::iterator it2 = it1->second.find(p.first);
    if (it2 == it1->second.end())
      return 0;

    it1->second.erase(it2);
    if (it1->second.empty())
      d_storage.erase(it1);

    return 1;
  }

  // only used for "reconstruction", which reinserts everything into another table and swaps
  // seems like that should just call rehash() or something
  template<typename Iter>
  void insert(Iter beg, Iter end) {
    std::vector<typename iterator::value_type> entries(beg, end);

    // sort by second (NodeValue*) then first (uint64_t)
    std::sort(entries.begin(), entries.end(),
              [](auto const & a, auto const & b) {
                return (a.first.second < b.first.second) ||
                  ((a.first.second == b.first.second) &&
                   (a.first.first < b.first.first));
              });

    auto find_different_nv = [](auto nv, auto first, auto last) {
      return std::find_if(first, last, [nv](auto const & a){ return a.first.second != nv; });
    };

    std::size_t l1_unique_count = 0;
    for (auto it = entries.begin(); it != entries.end();
         it = find_different_nv(it->first.second, it, entries.end())) {
      ++l1_unique_count;
    }
    d_storage.reserve(d_storage.size() + l1_unique_count);

    for (auto it = entries.begin(); it != entries.end();) {
      auto chunk_end = std::find_if(it, entries.end(),
                                    [it](auto const & v) { return v.first.second != it->first.second; });
      // add to l2 map
      auto & l2 = d_storage[it->first.second];
      l2.reserve(l2.size() + std::distance(it, chunk_end));
      for (;it != chunk_end; ++it) {
        l2.emplace(it->first.first, it->second);
      }
    }
  }

  void swap(AttrHash & other) {
    std::swap(d_storage, other.d_storage);
  }

  // the main way we get new entries.
  // AttributeManager::setAttribute is called with a NodeValue and a new attribute.
  // it generates a fresh id and inserts the attribute
  V & operator[](std::pair<uint64_t, NodeValue*> p) {
    return d_storage[p.second][p.first];
  }

  void clear() {
    d_storage.clear();
  }

  const_iterator find(std::pair<std::uint64_t, NodeValue*> p) const {
    typename Storage::const_iterator it1 = d_storage.find(p.second);
    if (it1 == d_storage.end())
      return const_iterator();

    typename IdMap::const_iterator it2 = it1->second.find(p.first);
    if (it2 == it1->second.end())
      return const_iterator();

    return const_iterator(this, it1, it2);
  }

  void delByNodeValue(NodeValue* nv) {
    d_storage.erase(nv);
  }

};
// };/* class AttrHash<> */

/**
 * In the case of Boolean-valued attributes we have a special
 * "AttrHash<bool>" to pack bits together in words.
 */
template <>
class AttrHash<bool> :
    protected std::unordered_map<NodeValue*,
                                  uint64_t,
                                  AttrBoolHashFunction> {

  /** A "super" type, like in Java, for easy reference below. */
  typedef std::unordered_map<NodeValue*, uint64_t, AttrBoolHashFunction> super;

  /**
   * BitAccessor allows us to return a bit "by reference."  Of course,
   * we don't require bit-addressibility supported by the system, we
   * do it with a complex type.
   */
  class BitAccessor {

    uint64_t& d_word;

    uint64_t d_bit;

   public:
    BitAccessor(uint64_t& word, uint64_t bit) : d_word(word), d_bit(bit) {}

    BitAccessor& operator=(bool b) {
      if(b) {
        // set the bit
        d_word |= GetBitSet(d_bit);
      } else {
        // clear the bit
        d_word &= ~GetBitSet(d_bit);
      }

      return *this;
    }

    operator bool() const { return (d_word & GetBitSet(d_bit)) ? true : false; }
  };/* class AttrHash<bool>::BitAccessor */

  /**
   * A (somewhat degenerate) iterator over boolean-valued attributes.
   * This iterator doesn't support anything except comparison and
   * dereference.  It's intended just for the result of find() on the
   * table.
   */
  class BitIterator {

    std::pair<NodeValue* const, uint64_t>* d_entry;

    uint64_t d_bit;

   public:

    BitIterator() :
      d_entry(NULL),
      d_bit(0) {
    }

    BitIterator(std::pair<NodeValue* const, uint64_t>& entry, uint64_t bit)
        : d_entry(&entry), d_bit(bit)
    {
    }

    std::pair<NodeValue* const, BitAccessor> operator*() {
      return std::make_pair(d_entry->first,
                            BitAccessor(d_entry->second, d_bit));
    }

    bool operator==(const BitIterator& b) {
      return d_entry == b.d_entry && d_bit == b.d_bit;
    }
  };/* class AttrHash<bool>::BitIterator */

  /**
   * A (somewhat degenerate) const_iterator over boolean-valued
   * attributes.  This const_iterator doesn't support anything except
   * comparison and dereference.  It's intended just for the result of
   * find() on the table.
   */
  class ConstBitIterator {

    const std::pair<NodeValue* const, uint64_t>* d_entry;

    uint64_t d_bit;

   public:

    ConstBitIterator() :
      d_entry(NULL),
      d_bit(0) {
    }

    ConstBitIterator(const std::pair<NodeValue* const, uint64_t>& entry,
                     uint64_t bit)
        : d_entry(&entry), d_bit(bit)
    {
    }

    std::pair<NodeValue* const, bool> operator*()
    {
      return std::make_pair(
          d_entry->first, (d_entry->second & GetBitSet(d_bit)) ? true : false);
    }

    bool operator==(const ConstBitIterator& b) {
      return d_entry == b.d_entry && d_bit == b.d_bit;
    }
  };/* class AttrHash<bool>::ConstBitIterator */

public:

  typedef std::pair<uint64_t, NodeValue*> key_type;
  typedef bool data_type;
  typedef std::pair<const key_type, data_type> value_type;

  /** an iterator type; see above for limitations */
  typedef BitIterator iterator;
  /** a const_iterator type; see above for limitations */
  typedef ConstBitIterator const_iterator;

  /**
   * Find the boolean value in the hash table.  Returns something ==
   * end() if not found.
   */
  BitIterator find(const std::pair<uint64_t, NodeValue*>& k) {
    super::iterator i = super::find(k.second);
    if(i == super::end()) {
      return BitIterator();
    }
    /*
    Trace.printf("boolattr",
                 "underlying word at 0x%p looks like 0x%016llx, bit is %u\n",
                 &(*i).second,
                 (uint64_t)((*i).second),
                 uint64_t(k.first));
    */
    return BitIterator(*i, k.first);
  }

  /** The "off the end" iterator */
  BitIterator end() {
    return BitIterator();
  }

  /**
   * Find the boolean value in the hash table.  Returns something ==
   * end() if not found.
   */
  ConstBitIterator find(const std::pair<uint64_t, NodeValue*>& k) const {
    super::const_iterator i = super::find(k.second);
    if(i == super::end()) {
      return ConstBitIterator();
    }
    /*
    Trace.printf("boolattr",
                 "underlying word at 0x%p looks like 0x%016llx, bit is %u\n",
                 &(*i).second,
                 (uint64_t)((*i).second),
                 uint64_t(k.first));
    */
    return ConstBitIterator(*i, k.first);
  }

  /** The "off the end" const_iterator */
  ConstBitIterator end() const {
    return ConstBitIterator();
  }

  /**
   * Access the hash table using the underlying operator[].  Inserts
   * the key into the table (associated to default value) if it's not
   * already there.
   */
  BitAccessor operator[](const std::pair<uint64_t, NodeValue*>& k) {
    uint64_t& word = super::operator[](k.second);
    return BitAccessor(word, k.first);
  }

  /**
   * Delete all flags from the given node.
   */
  void erase(NodeValue* nv) {
    super::erase(nv);
  }

  /**
   * Clear the hash table.
   */
  void clear() {
    super::clear();
  }

  /** Is the hash table empty? */
  bool empty() const {
    return super::empty();
  }

  /** This is currently very misleading! */
  size_t size() const {
    return super::size();
  }
};/* class AttrHash<bool> */

}  // namespace attr

// ATTRIBUTE IDENTIFIER ASSIGNMENT TEMPLATE ====================================

namespace attr {

/**
 * This is the last-attribute-assigner.  IDs are not globally
 * unique; rather, they are unique for each table_value_type.
 */
template <class T>
struct LastAttributeId
{
 public:
  static uint64_t getNextId() {
    uint64_t* id = raw_id();
    const uint64_t next_id = *id;
    ++(*id);
    return next_id;
  }
  static uint64_t getId() {
    return *raw_id();
  }
 private:
  static uint64_t* raw_id()
  {
    static uint64_t s_id = 0;
    return &s_id;
  }
};

}  // namespace attr

// ATTRIBUTE DEFINITION ========================================================

/**
 * An "attribute type" structure.
 *
 * @param T the tag for the attribute kind.
 *
 * @param value_t the underlying value_type for the attribute kind
 */
template <class T, class value_t>
class Attribute
{
  /**
   * The unique ID associated to this attribute.  Assigned statically,
   * at load time.
   */
  static const uint64_t s_id;

public:

  /** The value type for this attribute. */
  typedef value_t value_type;

  /** Get the unique ID associated to this attribute. */
  static inline uint64_t getId() { return s_id; }

  /**
   * This attribute does not have a default value: calling
   * hasAttribute() for a Node that hasn't had this attribute set will
   * return false, and getAttribute() for the Node will return a
   * default-constructed value_type.
   */
  static const bool has_default_value = false;

  /**
   * Register this attribute kind and check that the ID is a valid ID
   * for bool-valued attributes.  Fail an assert if not.  Otherwise
   * return the id.
   */
  static inline uint64_t registerAttribute() {
    typedef typename attr::KindValueToTableValueMapping<value_t>::
                     table_value_type table_value_type;
    return attr::LastAttributeId<table_value_type>::getNextId();
  }
};/* class Attribute<> */

/**
 * An "attribute type" structure for boolean flags (special).
 */
template <class T>
class Attribute<T, bool>
{
  /** IDs for bool-valued attributes are actually bit assignments. */
  static const uint64_t s_id;

public:

  /** The value type for this attribute; here, bool. */
  typedef bool value_type;

  /** Get the unique ID associated to this attribute. */
  static inline uint64_t getId() { return s_id; }

  /**
   * Such bool-valued attributes ("flags") have a default value: they
   * are false for all nodes on entry.  Calling hasAttribute() for a
   * Node that hasn't had this attribute set will return true, and
   * getAttribute() for the Node will return the default_value below.
   */
  static const bool has_default_value = true;

  /**
   * Default value of the attribute for Nodes without one explicitly
   * set.
   */
  static const bool default_value = false;

  /**
   * Register this attribute kind and check that the ID is a valid ID
   * for bool-valued attributes.  Fail an assert if not.  Otherwise
   * return the id.
   */
  static inline uint64_t registerAttribute() {
    const uint64_t id = attr::LastAttributeId<bool>::getNextId();
    AlwaysAssert(id <= 63) << "Too many boolean node attributes registered "
                              "during initialization !";
    return id;
  }
};/* class Attribute<..., bool, ...> */

// ATTRIBUTE IDENTIFIER ASSIGNMENT =============================================

/** Assign unique IDs to attributes at load time. */
template <class T, class value_t>
const uint64_t Attribute<T, value_t>::s_id =
    Attribute<T, value_t>::registerAttribute();

/** Assign unique IDs to attributes at load time. */
template <class T>
const uint64_t Attribute<T, bool>::s_id =
    Attribute<T, bool>::registerAttribute();

}  // namespace expr
}  // namespace cvc5::internal

#endif /* CVC5__EXPR__ATTRIBUTE_INTERNALS_H */
