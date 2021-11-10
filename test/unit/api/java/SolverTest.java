/******************************************************************************
 * Top contributors (to current version):
 *   Aina Niemetz, Mudathir Mohamed, Andrew Reynolds
 *
 * This file is part of the cvc5 project.
 *
 * Copyright (c) 2009-2021 by the authors listed in the file AUTHORS
 * in the top-level source directory and their institutional affiliations.
 * All rights reserved.  See the file COPYING in the top-level source
 * directory for licensing information.
 * ****************************************************************************
 *
 * Black box testing of the Solver class of the Java API.
 */

package tests;

import static io.github.cvc5.api.Kind.*;
import static io.github.cvc5.api.RoundingMode.*;
import static org.junit.jupiter.api.Assertions.*;

import io.github.cvc5.api.*;
import java.math.BigInteger;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.function.Executable;

class SolverTest
{
  private Solver d_solver;

  @BeforeEach void setUp()
  {
    d_solver = new Solver();
  }

  @AfterEach void tearDown()
  {
    d_solver.close();
  }

  @Test void recoverableException() throws CVC5ApiException
  {
    d_solver.setOption("produce-models", "true");
    Term x = d_solver.mkConst(d_solver.getBooleanSort(), "x");
    d_solver.assertFormula(x.eqTerm(x).notTerm());
    assertThrows(CVC5ApiRecoverableException.class, () -> d_solver.getValue(x));
  }

  @Test void supportsFloatingPoint() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.mkRoundingMode(ROUND_NEAREST_TIES_TO_EVEN));
  }

  @Test void getBooleanSort() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.getBooleanSort());
  }

  @Test void getIntegerSort()
  {
    assertDoesNotThrow(() -> d_solver.getIntegerSort());
  }

  @Test void getNullSort() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.getNullSort());
  }

  @Test void getRealSort() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.getRealSort());
  }

  @Test void getRegExpSort() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.getRegExpSort());
  }

  @Test void getStringSort() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.getStringSort());
  }

  @Test void getRoundingModeSort() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.getRoundingModeSort());
  }

  @Test void mkArraySort() throws CVC5ApiException
  {
    Sort boolSort = d_solver.getBooleanSort();
    Sort intSort = d_solver.getIntegerSort();
    Sort realSort = d_solver.getRealSort();
    Sort bvSort = d_solver.mkBitVectorSort(32);
    assertDoesNotThrow(() -> d_solver.mkArraySort(boolSort, boolSort));
    assertDoesNotThrow(() -> d_solver.mkArraySort(intSort, intSort));
    assertDoesNotThrow(() -> d_solver.mkArraySort(realSort, realSort));
    assertDoesNotThrow(() -> d_solver.mkArraySort(bvSort, bvSort));
    assertDoesNotThrow(() -> d_solver.mkArraySort(boolSort, intSort));
    assertDoesNotThrow(() -> d_solver.mkArraySort(realSort, bvSort));

    Sort fpSort = d_solver.mkFloatingPointSort(3, 5);
    assertDoesNotThrow(() -> d_solver.mkArraySort(fpSort, fpSort));
    assertDoesNotThrow(() -> d_solver.mkArraySort(bvSort, fpSort));

    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.mkArraySort(boolSort, boolSort));
    slv.close();
  }

  @Test void mkBitVectorSort() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.mkBitVectorSort(32));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkBitVectorSort(0));
  }

  @Test void mkFloatingPointSort() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.mkFloatingPointSort(4, 8));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkFloatingPointSort(0, 8));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkFloatingPointSort(4, 0));
  }

  @Test void mkDatatypeSort() throws CVC5ApiException
  {
    DatatypeDecl dtypeSpec = d_solver.mkDatatypeDecl("list");
    DatatypeConstructorDecl cons = d_solver.mkDatatypeConstructorDecl("cons");
    cons.addSelector("head", d_solver.getIntegerSort());
    dtypeSpec.addConstructor(cons);
    DatatypeConstructorDecl nil = d_solver.mkDatatypeConstructorDecl("nil");
    dtypeSpec.addConstructor(nil);
    assertDoesNotThrow(() -> d_solver.mkDatatypeSort(dtypeSpec));

    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.mkDatatypeSort(dtypeSpec));

    DatatypeDecl throwsDtypeSpec = d_solver.mkDatatypeDecl("list");
    assertThrows(CVC5ApiException.class, () -> d_solver.mkDatatypeSort(throwsDtypeSpec));
    slv.close();
  }

  @Test void mkDatatypeSorts() throws CVC5ApiException
  {
    Solver slv = new Solver();

    DatatypeDecl dtypeSpec1 = d_solver.mkDatatypeDecl("list1");
    DatatypeConstructorDecl cons1 = d_solver.mkDatatypeConstructorDecl("cons1");
    cons1.addSelector("head1", d_solver.getIntegerSort());
    dtypeSpec1.addConstructor(cons1);
    DatatypeConstructorDecl nil1 = d_solver.mkDatatypeConstructorDecl("nil1");
    dtypeSpec1.addConstructor(nil1);
    DatatypeDecl dtypeSpec2 = d_solver.mkDatatypeDecl("list2");
    DatatypeConstructorDecl cons2 = d_solver.mkDatatypeConstructorDecl("cons2");
    cons2.addSelector("head2", d_solver.getIntegerSort());
    dtypeSpec2.addConstructor(cons2);
    DatatypeConstructorDecl nil2 = d_solver.mkDatatypeConstructorDecl("nil2");
    dtypeSpec2.addConstructor(nil2);
    DatatypeDecl[] decls = {dtypeSpec1, dtypeSpec2};
    assertDoesNotThrow(() -> d_solver.mkDatatypeSorts(decls));

    assertThrows(CVC5ApiException.class, () -> slv.mkDatatypeSorts(decls));

    DatatypeDecl throwsDtypeSpec = d_solver.mkDatatypeDecl("list");
    DatatypeDecl[] throwsDecls = new DatatypeDecl[] {throwsDtypeSpec};
    assertThrows(CVC5ApiException.class, () -> d_solver.mkDatatypeSorts(throwsDecls));

    /* with unresolved sorts */
    Sort unresList = d_solver.mkUninterpretedSort("ulist");
    Set<Sort> unresSorts = new HashSet<>();
    unresSorts.add(unresList);
    DatatypeDecl ulist = d_solver.mkDatatypeDecl("ulist");
    DatatypeConstructorDecl ucons = d_solver.mkDatatypeConstructorDecl("ucons");
    ucons.addSelector("car", unresList);
    ucons.addSelector("cdr", unresList);
    ulist.addConstructor(ucons);
    DatatypeConstructorDecl unil = d_solver.mkDatatypeConstructorDecl("unil");
    ulist.addConstructor(unil);
    DatatypeDecl[] udecls = new DatatypeDecl[] {ulist};
    assertDoesNotThrow(() -> d_solver.mkDatatypeSorts(Arrays.asList(udecls), unresSorts));

    assertThrows(
        CVC5ApiException.class, () -> slv.mkDatatypeSorts(Arrays.asList(udecls), unresSorts));
    slv.close();

    /* Note: More tests are in datatype_api_black. */
  }

  @Test void mkFunctionSort() throws CVC5ApiException
  {
    assertDoesNotThrow(()
                           -> d_solver.mkFunctionSort(
                               d_solver.mkUninterpretedSort("u"), d_solver.getIntegerSort()));
    Sort funSort =
        d_solver.mkFunctionSort(d_solver.mkUninterpretedSort("u"), d_solver.getIntegerSort());
    // function arguments are allowed
    assertDoesNotThrow(() -> d_solver.mkFunctionSort(funSort, d_solver.getIntegerSort()));
    // non-first-class arguments are not allowed
    Sort reSort = d_solver.getRegExpSort();
    assertThrows(
        CVC5ApiException.class, () -> d_solver.mkFunctionSort(reSort, d_solver.getIntegerSort()));

    assertThrows(
        CVC5ApiException.class, () -> d_solver.mkFunctionSort(d_solver.getIntegerSort(), funSort));

    assertDoesNotThrow(()
                           -> d_solver.mkFunctionSort(new Sort[] {d_solver.mkUninterpretedSort("u"),
                                                          d_solver.getIntegerSort()},
                               d_solver.getIntegerSort()));
    Sort funSort2 =
        d_solver.mkFunctionSort(d_solver.mkUninterpretedSort("u"), d_solver.getIntegerSort());
    // function arguments are allowed
    assertDoesNotThrow(
        ()
            -> d_solver.mkFunctionSort(new Sort[] {funSort2, d_solver.mkUninterpretedSort("u")},
                d_solver.getIntegerSort()));
    assertThrows(CVC5ApiException.class,
        ()
            -> d_solver.mkFunctionSort(
                new Sort[] {d_solver.getIntegerSort(), d_solver.mkUninterpretedSort("u")},
                funSort2));

    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class,
        () -> slv.mkFunctionSort(d_solver.mkUninterpretedSort("u"), d_solver.getIntegerSort()));

    assertThrows(CVC5ApiException.class,
        () -> slv.mkFunctionSort(slv.mkUninterpretedSort("u"), d_solver.getIntegerSort()));

    Sort[] sorts1 =
        new Sort[] {d_solver.getBooleanSort(), slv.getIntegerSort(), d_solver.getIntegerSort()};
    Sort[] sorts2 = new Sort[] {slv.getBooleanSort(), slv.getIntegerSort()};
    assertDoesNotThrow(() -> slv.mkFunctionSort(sorts2, slv.getIntegerSort()));
    assertThrows(CVC5ApiException.class, () -> slv.mkFunctionSort(sorts1, slv.getIntegerSort()));
    assertThrows(
        CVC5ApiException.class, () -> slv.mkFunctionSort(sorts2, d_solver.getIntegerSort()));
    slv.close();
  }

  @Test void mkParamSort() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.mkParamSort("T"));
    assertDoesNotThrow(() -> d_solver.mkParamSort(""));
  }

  @Test void mkPredicateSort()
  {
    assertDoesNotThrow(() -> d_solver.mkPredicateSort(new Sort[] {d_solver.getIntegerSort()}));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkPredicateSort(new Sort[] {}));
    Sort funSort =
        d_solver.mkFunctionSort(d_solver.mkUninterpretedSort("u"), d_solver.getIntegerSort());
    // functions as arguments are allowed
    assertDoesNotThrow(
        () -> d_solver.mkPredicateSort(new Sort[] {d_solver.getIntegerSort(), funSort}));

    Solver slv = new Solver();
    assertThrows(
        CVC5ApiException.class, () -> slv.mkPredicateSort(new Sort[] {d_solver.getIntegerSort()}));
    slv.close();
  }

  @Test void mkRecordSort() throws CVC5ApiException
  {
    Pair<String, Sort>[] fields = new Pair[] {new Pair<>("b", d_solver.getBooleanSort()),
        new Pair<>("bv", d_solver.mkBitVectorSort(8)),
        new Pair<>("i", d_solver.getIntegerSort())};
    Pair<String, Sort>[] empty = new Pair[0];
    assertDoesNotThrow(() -> d_solver.mkRecordSort(fields));
    assertDoesNotThrow(() -> d_solver.mkRecordSort(empty));
    Sort recSort = d_solver.mkRecordSort(fields);
    assertDoesNotThrow(() -> recSort.getDatatype());

    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.mkRecordSort(fields));
    slv.close();
  }

  @Test void mkSetSort() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.mkSetSort(d_solver.getBooleanSort()));
    assertDoesNotThrow(() -> d_solver.mkSetSort(d_solver.getIntegerSort()));
    assertDoesNotThrow(() -> d_solver.mkSetSort(d_solver.mkBitVectorSort(4)));
    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.mkSetSort(d_solver.mkBitVectorSort(4)));
    slv.close();
  }

  @Test void mkBagSort() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.mkBagSort(d_solver.getBooleanSort()));
    assertDoesNotThrow(() -> d_solver.mkBagSort(d_solver.getIntegerSort()));
    assertDoesNotThrow(() -> d_solver.mkBagSort(d_solver.mkBitVectorSort(4)));
    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.mkBagSort(d_solver.mkBitVectorSort(4)));
    slv.close();
  }

  @Test void mkSequenceSort() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.mkSequenceSort(d_solver.getBooleanSort()));
    assertDoesNotThrow(
        () -> d_solver.mkSequenceSort(d_solver.mkSequenceSort(d_solver.getIntegerSort())));
    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.mkSequenceSort(d_solver.getIntegerSort()));
    slv.close();
  }

  @Test void mkUninterpretedSort() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.mkUninterpretedSort("u"));
    assertDoesNotThrow(() -> d_solver.mkUninterpretedSort(""));
  }

  @Test void mkSortConstructorSort() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.mkSortConstructorSort("s", 2));
    assertDoesNotThrow(() -> d_solver.mkSortConstructorSort("", 2));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkSortConstructorSort("", 0));
  }

  @Test void mkTupleSort() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.mkTupleSort(new Sort[] {d_solver.getIntegerSort()}));
    Sort funSort =
        d_solver.mkFunctionSort(d_solver.mkUninterpretedSort("u"), d_solver.getIntegerSort());
    assertThrows(CVC5ApiException.class,
        () -> d_solver.mkTupleSort(new Sort[] {d_solver.getIntegerSort(), funSort}));

    Solver slv = new Solver();
    assertThrows(
        CVC5ApiException.class, () -> slv.mkTupleSort(new Sort[] {d_solver.getIntegerSort()}));
    slv.close();
  }

  @Test void mkBitVector() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.mkBitVector(8, 2));
    assertDoesNotThrow(() -> d_solver.mkBitVector(32, 2));
    assertDoesNotThrow(() -> d_solver.mkBitVector(8, "-1111111", 2));
    assertDoesNotThrow(() -> d_solver.mkBitVector(8, "0101", 2));
    assertDoesNotThrow(() -> d_solver.mkBitVector(8, "00000101", 2));
    assertDoesNotThrow(() -> d_solver.mkBitVector(8, "-127", 10));
    assertDoesNotThrow(() -> d_solver.mkBitVector(8, "255", 10));
    assertDoesNotThrow(() -> d_solver.mkBitVector(8, "-7f", 16));
    assertDoesNotThrow(() -> d_solver.mkBitVector(8, "a0", 16));

    assertThrows(CVC5ApiException.class, () -> d_solver.mkBitVector(0, 2));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkBitVector(0, "-127", 10));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkBitVector(0, "a0", 16));

    assertThrows(CVC5ApiException.class, () -> d_solver.mkBitVector(8, "", 2));

    assertThrows(CVC5ApiException.class, () -> d_solver.mkBitVector(8, "101", 5));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkBitVector(8, "128", 11));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkBitVector(8, "a0", 21));

    assertThrows(CVC5ApiException.class, () -> d_solver.mkBitVector(8, "-11111111", 2));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkBitVector(8, "101010101", 2));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkBitVector(8, "-256", 10));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkBitVector(8, "257", 10));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkBitVector(8, "-a0", 16));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkBitVector(8, "fffff", 16));

    assertThrows(CVC5ApiException.class, () -> d_solver.mkBitVector(8, "10201010", 2));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkBitVector(8, "-25x", 10));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkBitVector(8, "2x7", 10));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkBitVector(8, "fzff", 16));

    assertEquals(d_solver.mkBitVector(8, "0101", 2), d_solver.mkBitVector(8, "00000101", 2));
    assertEquals(d_solver.mkBitVector(4, "-1", 2), d_solver.mkBitVector(4, "1111", 2));
    assertEquals(d_solver.mkBitVector(4, "-1", 16), d_solver.mkBitVector(4, "1111", 2));
    assertEquals(d_solver.mkBitVector(4, "-1", 10), d_solver.mkBitVector(4, "1111", 2));
    assertEquals(d_solver.mkBitVector(8, "01010101", 2).toString(), "#b01010101");
    assertEquals(d_solver.mkBitVector(8, "F", 16).toString(), "#b00001111");
    assertEquals(d_solver.mkBitVector(8, "-1", 10), d_solver.mkBitVector(8, "FF", 16));
  }

  @Test void mkVar() throws CVC5ApiException
  {
    Sort boolSort = d_solver.getBooleanSort();
    Sort intSort = d_solver.getIntegerSort();
    Sort funSort = d_solver.mkFunctionSort(intSort, boolSort);
    assertDoesNotThrow(() -> d_solver.mkVar(boolSort));
    assertDoesNotThrow(() -> d_solver.mkVar(funSort));
    assertDoesNotThrow(() -> d_solver.mkVar(boolSort, ("b")));
    assertDoesNotThrow(() -> d_solver.mkVar(funSort, ""));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkVar(d_solver.getNullSort()));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkVar(d_solver.getNullSort(), "a"));
    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.mkVar(boolSort, "x"));
    slv.close();
  }

  @Test void mkBoolean() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.mkBoolean(true));
    assertDoesNotThrow(() -> d_solver.mkBoolean(false));
  }

  @Test void mkRoundingMode() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.mkRoundingMode(RoundingMode.ROUND_TOWARD_ZERO));
  }

  @Test void mkAbstractValue() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.mkAbstractValue(("1")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkAbstractValue(("0")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkAbstractValue(("-1")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkAbstractValue(("1.2")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkAbstractValue("1/2"));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkAbstractValue("asdf"));

    assertDoesNotThrow(() -> d_solver.mkAbstractValue((int) 1));
    assertDoesNotThrow(() -> d_solver.mkAbstractValue((int) 1));
    assertDoesNotThrow(() -> d_solver.mkAbstractValue((long) 1));
    assertDoesNotThrow(() -> d_solver.mkAbstractValue((long) 1));
    // java does not have specific types for unsigned integers, therefore the two lines below do not
    // make sense in java. assertDoesNotThrow(() -> d_solver.mkAbstractValue((int)-1));
    // assertDoesNotThrow(() -> d_solver.mkAbstractValue((long)-1));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkAbstractValue(0));
  }

  @Test void mkFloatingPoint() throws CVC5ApiException
  {
    Term t1 = d_solver.mkBitVector(8);
    Term t2 = d_solver.mkBitVector(4);
    Term t3 = d_solver.mkInteger(2);
    assertDoesNotThrow(() -> d_solver.mkFloatingPoint(3, 5, t1));

    assertThrows(
        CVC5ApiException.class, () -> d_solver.mkFloatingPoint(0, 5, d_solver.getNullTerm()));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkFloatingPoint(0, 5, t1));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkFloatingPoint(3, 0, t1));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkFloatingPoint(3, 5, t2));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkFloatingPoint(3, 5, t2));

    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.mkFloatingPoint(3, 5, t1));
    slv.close();
  }

  @Test void mkCardinalityConstraint() throws CVC5ApiException
  {
    Sort su = d_solver.mkUninterpretedSort("u");
    Sort si = d_solver.getIntegerSort();
    assertDoesNotThrow(() -> d_solver.mkCardinalityConstraint(su, 3));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkCardinalityConstraint(si, 3));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkCardinalityConstraint(su, 0));
    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.mkCardinalityConstraint(su, 3));
    slv.close();
  }

  @Test void mkEmptySet() throws CVC5ApiException
  {
    Solver slv = new Solver();
    Sort s = d_solver.mkSetSort(d_solver.getBooleanSort());
    assertDoesNotThrow(() -> d_solver.mkEmptySet(d_solver.getNullSort()));
    assertDoesNotThrow(() -> d_solver.mkEmptySet(s));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkEmptySet(d_solver.getBooleanSort()));
    assertThrows(CVC5ApiException.class, () -> slv.mkEmptySet(s));
    slv.close();
  }

  @Test void mkEmptyBag() throws CVC5ApiException
  {
    Solver slv = new Solver();
    Sort s = d_solver.mkBagSort(d_solver.getBooleanSort());
    assertDoesNotThrow(() -> d_solver.mkEmptyBag(d_solver.getNullSort()));
    assertDoesNotThrow(() -> d_solver.mkEmptyBag(s));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkEmptyBag(d_solver.getBooleanSort()));

    assertThrows(CVC5ApiException.class, () -> slv.mkEmptyBag(s));
    slv.close();
  }

  @Test void mkEmptySequence() throws CVC5ApiException
  {
    Solver slv = new Solver();
    Sort s = d_solver.mkSequenceSort(d_solver.getBooleanSort());
    assertDoesNotThrow(() -> d_solver.mkEmptySequence(s));
    assertDoesNotThrow(() -> d_solver.mkEmptySequence(d_solver.getBooleanSort()));
    assertThrows(CVC5ApiException.class, () -> slv.mkEmptySequence(s));
    slv.close();
  }

  @Test void mkFalse() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.mkFalse());
    assertDoesNotThrow(() -> d_solver.mkFalse());
  }

  @Test void mkNaN() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.mkNaN(3, 5));
  }

  @Test void mkNegZero() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.mkNegZero(3, 5));
  }

  @Test void mkNegInf()
  {
    assertDoesNotThrow(() -> d_solver.mkNegInf(3, 5));
  }

  @Test void mkPosInf()
  {
    assertDoesNotThrow(() -> d_solver.mkPosInf(3, 5));
  }

  @Test void mkPosZero()
  {
    assertDoesNotThrow(() -> d_solver.mkPosZero(3, 5));
  }

  @Test void mkOp()
  {
    // Unlike c++,  mkOp(Kind kind, Kind k) is a type error in java
    // assertThrows(CVC5ApiException.class, () -> d_solver.mkOp(BITVECTOR_EXTRACT, EQUAL));

    // mkOp(Kind kind, const std::string& arg)
    assertDoesNotThrow(() -> d_solver.mkOp(DIVISIBLE, "2147483648"));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkOp(BITVECTOR_EXTRACT, "asdf"));

    // mkOp(Kind kind, int arg)
    assertDoesNotThrow(() -> d_solver.mkOp(DIVISIBLE, 1));
    assertDoesNotThrow(() -> d_solver.mkOp(BITVECTOR_ROTATE_LEFT, 1));
    assertDoesNotThrow(() -> d_solver.mkOp(BITVECTOR_ROTATE_RIGHT, 1));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkOp(BITVECTOR_EXTRACT, 1));

    // mkOp(Kind kind, int arg1, int arg2)
    assertDoesNotThrow(() -> d_solver.mkOp(BITVECTOR_EXTRACT, 1, 1));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkOp(DIVISIBLE, 1, 2));

    // mkOp(Kind kind, int[] args)
    int[] args = new int[] {1, 2, 2};
    assertDoesNotThrow(() -> d_solver.mkOp(TUPLE_PROJECT, args));
  }

  @Test void mkPi()
  {
    assertDoesNotThrow(() -> d_solver.mkPi());
  }

  @Test void mkInteger()
  {
    assertDoesNotThrow(() -> d_solver.mkInteger("123"));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger("1.23"));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger("1/23"));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger("12/3"));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger(".2"));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger("2."));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger(""));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger("asdf"));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger("1.2/3"));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger("."));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger("/"));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger("2/"));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger("/2"));

    assertDoesNotThrow(() -> d_solver.mkReal(("123")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger(("1.23")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger(("1/23")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger(("12/3")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger((".2")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger(("2.")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger(("")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger(("asdf")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger(("1.2/3")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger((".")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger(("/")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger(("2/")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkInteger(("/2")));

    int val1 = 1;
    long val2 = -1;
    int val3 = 1;
    long val4 = -1;
    assertDoesNotThrow(() -> d_solver.mkInteger(val1));
    assertDoesNotThrow(() -> d_solver.mkInteger(val2));
    assertDoesNotThrow(() -> d_solver.mkInteger(val3));
    assertDoesNotThrow(() -> d_solver.mkInteger(val4));
    assertDoesNotThrow(() -> d_solver.mkInteger(val4));
  }

  @Test void mkReal()
  {
    assertDoesNotThrow(() -> d_solver.mkReal("123"));
    assertDoesNotThrow(() -> d_solver.mkReal("1.23"));
    assertDoesNotThrow(() -> d_solver.mkReal("1/23"));
    assertDoesNotThrow(() -> d_solver.mkReal("12/3"));
    assertDoesNotThrow(() -> d_solver.mkReal(".2"));
    assertDoesNotThrow(() -> d_solver.mkReal("2."));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkReal(""));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkReal("asdf"));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkReal("1.2/3"));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkReal("."));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkReal("/"));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkReal("2/"));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkReal("/2"));

    assertDoesNotThrow(() -> d_solver.mkReal(("123")));
    assertDoesNotThrow(() -> d_solver.mkReal(("1.23")));
    assertDoesNotThrow(() -> d_solver.mkReal(("1/23")));
    assertDoesNotThrow(() -> d_solver.mkReal(("12/3")));
    assertDoesNotThrow(() -> d_solver.mkReal((".2")));
    assertDoesNotThrow(() -> d_solver.mkReal(("2.")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkReal(("")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkReal(("asdf")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkReal(("1.2/3")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkReal((".")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkReal(("/")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkReal(("2/")));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkReal(("/2")));

    int val1 = 1;
    long val2 = -1;
    int val3 = 1;
    long val4 = -1;
    assertDoesNotThrow(() -> d_solver.mkReal(val1));
    assertDoesNotThrow(() -> d_solver.mkReal(val2));
    assertDoesNotThrow(() -> d_solver.mkReal(val3));
    assertDoesNotThrow(() -> d_solver.mkReal(val4));
    assertDoesNotThrow(() -> d_solver.mkReal(val4));
    assertDoesNotThrow(() -> d_solver.mkReal(val1, val1));
    assertDoesNotThrow(() -> d_solver.mkReal(val2, val2));
    assertDoesNotThrow(() -> d_solver.mkReal(val3, val3));
    assertDoesNotThrow(() -> d_solver.mkReal(val4, val4));
  }

  @Test void mkRegexpNone()
  {
    Sort strSort = d_solver.getStringSort();
    Term s = d_solver.mkConst(strSort, "s");
    assertDoesNotThrow(() -> d_solver.mkTerm(STRING_IN_REGEXP, s, d_solver.mkRegexpNone()));
  }

  @Test void mkRegexpAll()
  {
    Sort strSort = d_solver.getStringSort();
    Term s = d_solver.mkConst(strSort, "s");
    assertDoesNotThrow(() -> d_solver.mkTerm(STRING_IN_REGEXP, s, d_solver.mkRegexpAll()));
  }

  @Test void mkRegexpAllchar()
  {
    Sort strSort = d_solver.getStringSort();
    Term s = d_solver.mkConst(strSort, "s");
    assertDoesNotThrow(() -> d_solver.mkTerm(STRING_IN_REGEXP, s, d_solver.mkRegexpAllchar()));
  }

  @Test void mkSepEmp()
  {
    assertDoesNotThrow(() -> d_solver.mkSepEmp());
  }

  @Test void mkSepNil()
  {
    assertDoesNotThrow(() -> d_solver.mkSepNil(d_solver.getBooleanSort()));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkSepNil(d_solver.getNullSort()));
    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.mkSepNil(d_solver.getIntegerSort()));
    slv.close();
  }

  @Test void mkString()
  {
    assertDoesNotThrow(() -> d_solver.mkString(""));
    assertDoesNotThrow(() -> d_solver.mkString("asdfasdf"));
    assertEquals(d_solver.mkString("asdf\\nasdf").toString(), "\"asdf\\u{5c}nasdf\"");
    assertEquals(d_solver.mkString("asdf\\u{005c}nasdf", true).toString(), "\"asdf\\u{5c}nasdf\"");
  }

  @Test void mkTerm() throws CVC5ApiException
  {
    Sort bv32 = d_solver.mkBitVectorSort(32);
    Term a = d_solver.mkConst(bv32, "a");
    Term b = d_solver.mkConst(bv32, "b");
    Term[] v1 = new Term[] {a, b};
    Term[] v2 = new Term[] {a, d_solver.getNullTerm()};
    Term[] v3 = new Term[] {a, d_solver.mkTrue()};
    Term[] v4 = new Term[] {d_solver.mkInteger(1), d_solver.mkInteger(2)};
    Term[] v5 = new Term[] {d_solver.mkInteger(1), d_solver.getNullTerm()};
    Term[] v6 = new Term[] {};
    Solver slv = new Solver();

    // mkTerm(Kind kind) const
    assertDoesNotThrow(() -> d_solver.mkTerm(PI));
    assertDoesNotThrow(() -> d_solver.mkTerm(REGEXP_NONE));
    assertDoesNotThrow(() -> d_solver.mkTerm(REGEXP_ALLCHAR));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(CONST_BITVECTOR));

    // mkTerm(Kind kind, Term child) const
    assertDoesNotThrow(() -> d_solver.mkTerm(NOT, d_solver.mkTrue()));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(NOT, d_solver.getNullTerm()));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(NOT, a));
    assertThrows(CVC5ApiException.class, () -> slv.mkTerm(NOT, d_solver.mkTrue()));

    // mkTerm(Kind kind, Term child1, Term child2) const
    assertDoesNotThrow(() -> d_solver.mkTerm(EQUAL, a, b));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(EQUAL, d_solver.getNullTerm(), b));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(EQUAL, a, d_solver.getNullTerm()));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(EQUAL, a, d_solver.mkTrue()));
    assertThrows(CVC5ApiException.class, () -> slv.mkTerm(EQUAL, a, b));

    // mkTerm(Kind kind, Term child1, Term child2, Term child3) const
    assertDoesNotThrow(
        () -> d_solver.mkTerm(ITE, d_solver.mkTrue(), d_solver.mkTrue(), d_solver.mkTrue()));
    assertThrows(CVC5ApiException.class,
        () -> d_solver.mkTerm(ITE, d_solver.getNullTerm(), d_solver.mkTrue(), d_solver.mkTrue()));

    assertThrows(CVC5ApiException.class,
        () -> d_solver.mkTerm(ITE, d_solver.mkTrue(), d_solver.getNullTerm(), d_solver.mkTrue()));

    assertThrows(CVC5ApiException.class,
        () -> d_solver.mkTerm(ITE, d_solver.mkTrue(), d_solver.mkTrue(), d_solver.getNullTerm()));

    assertThrows(CVC5ApiException.class,
        () -> d_solver.mkTerm(ITE, d_solver.mkTrue(), d_solver.mkTrue(), b));

    assertThrows(CVC5ApiException.class,
        () -> slv.mkTerm(ITE, d_solver.mkTrue(), d_solver.mkTrue(), d_solver.mkTrue()));

    // mkTerm(Kind kind, const Term[]& children) const
    assertDoesNotThrow(() -> d_solver.mkTerm(EQUAL, v1));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(EQUAL, v2));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(EQUAL, v3));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(DISTINCT, v6));
    slv.close();
  }

  @Test void mkTermFromOp() throws CVC5ApiException
  {
    Sort bv32 = d_solver.mkBitVectorSort(32);
    Term a = d_solver.mkConst(bv32, "a");
    Term b = d_solver.mkConst(bv32, "b");
    Term[] v1 = new Term[] {d_solver.mkInteger(1), d_solver.mkInteger(2)};
    Term[] v2 = new Term[] {d_solver.mkInteger(1), d_solver.getNullTerm()};
    Term[] v3 = new Term[] {};
    Term[] v4 = new Term[] {d_solver.mkInteger(5)};
    Solver slv = new Solver();

    // simple operator terms
    Op opterm1 = d_solver.mkOp(BITVECTOR_EXTRACT, 2, 1);
    Op opterm2 = d_solver.mkOp(DIVISIBLE, 1);

    // list datatype
    Sort sort = d_solver.mkParamSort("T");
    DatatypeDecl listDecl = d_solver.mkDatatypeDecl("paramlist", sort);
    DatatypeConstructorDecl cons = d_solver.mkDatatypeConstructorDecl("cons");
    DatatypeConstructorDecl nil = d_solver.mkDatatypeConstructorDecl("nil");
    cons.addSelector("head", sort);
    cons.addSelectorSelf("tail");
    listDecl.addConstructor(cons);
    listDecl.addConstructor(nil);
    Sort listSort = d_solver.mkDatatypeSort(listDecl);
    Sort intListSort = listSort.instantiate(new Sort[] {d_solver.getIntegerSort()});
    Term c = d_solver.mkConst(intListSort, "c");
    Datatype list = listSort.getDatatype();

    // list datatype constructor and selector operator terms
    Term consTerm1 = list.getConstructorTerm("cons");
    Term consTerm2 = list.getConstructor("cons").getConstructorTerm();
    Term nilTerm1 = list.getConstructorTerm("nil");
    Term nilTerm2 = list.getConstructor("nil").getConstructorTerm();
    Term headTerm1 = list.getConstructor("cons").getSelectorTerm("head");
    Term headTerm2 = list.getConstructor("cons").getSelector("head").getSelectorTerm();
    Term tailTerm1 = list.getConstructor("cons").getSelectorTerm("tail");
    Term tailTerm2 = list.getConstructor("cons").getSelector("tail").getSelectorTerm();

    // mkTerm(Op op, Term term) const
    assertDoesNotThrow(() -> d_solver.mkTerm(APPLY_CONSTRUCTOR, nilTerm1));
    assertDoesNotThrow(() -> d_solver.mkTerm(APPLY_CONSTRUCTOR, nilTerm2));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(APPLY_SELECTOR, nilTerm1));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(APPLY_SELECTOR, consTerm1));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(APPLY_CONSTRUCTOR, consTerm2));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(opterm1));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(APPLY_SELECTOR, headTerm1));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(opterm1));
    assertThrows(CVC5ApiException.class, () -> slv.mkTerm(APPLY_CONSTRUCTOR, nilTerm1));

    // mkTerm(Op op, Term child) const
    assertDoesNotThrow(() -> d_solver.mkTerm(opterm1, a));
    assertDoesNotThrow(() -> d_solver.mkTerm(opterm2, d_solver.mkInteger(1)));
    assertDoesNotThrow(() -> d_solver.mkTerm(APPLY_SELECTOR, headTerm1, c));
    assertDoesNotThrow(() -> d_solver.mkTerm(APPLY_SELECTOR, tailTerm2, c));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(opterm2, a));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(opterm1, d_solver.getNullTerm()));
    assertThrows(CVC5ApiException.class,
        () -> d_solver.mkTerm(APPLY_CONSTRUCTOR, consTerm1, d_solver.mkInteger(0)));

    assertThrows(CVC5ApiException.class, () -> slv.mkTerm(opterm1, a));

    // mkTerm(Op op, Term child1, Term child2) const
    assertDoesNotThrow(()
                           -> d_solver.mkTerm(APPLY_CONSTRUCTOR,
                               consTerm1,
                               d_solver.mkInteger(0),
                               d_solver.mkTerm(APPLY_CONSTRUCTOR, nilTerm1)));
    assertThrows(CVC5ApiException.class,
        () -> d_solver.mkTerm(opterm2, d_solver.mkInteger(1), d_solver.mkInteger(2)));

    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(opterm1, a, b));
    assertThrows(CVC5ApiException.class,
        () -> d_solver.mkTerm(opterm2, d_solver.mkInteger(1), d_solver.getNullTerm()));

    assertThrows(CVC5ApiException.class,
        () -> d_solver.mkTerm(opterm2, d_solver.getNullTerm(), d_solver.mkInteger(1)));

    assertThrows(CVC5ApiException.class,
        ()
            -> slv.mkTerm(APPLY_CONSTRUCTOR,
                consTerm1,
                d_solver.mkInteger(0),
                d_solver.mkTerm(APPLY_CONSTRUCTOR, nilTerm1)));

    // mkTerm(Op op, Term child1, Term child2, Term child3) const
    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(opterm1, a, b, a));
    assertThrows(CVC5ApiException.class,
        ()
            -> d_solver.mkTerm(
                opterm2, d_solver.mkInteger(1), d_solver.mkInteger(1), d_solver.getNullTerm()));

    // mkTerm(Op op, Term[] children)
    assertDoesNotThrow(() -> d_solver.mkTerm(opterm2, v4));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(opterm2, v1));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(opterm2, v2));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkTerm(opterm2, v3));
    assertThrows(CVC5ApiException.class, () -> slv.mkTerm(opterm2, v4));
    slv.close();
  }

  @Test void mkTrue()
  {
    assertDoesNotThrow(() -> d_solver.mkTrue());
    assertDoesNotThrow(() -> d_solver.mkTrue());
  }

  @Test void mkTuple()
  {
    assertDoesNotThrow(()
                           -> d_solver.mkTuple(new Sort[] {d_solver.mkBitVectorSort(3)},
                               new Term[] {d_solver.mkBitVector(3, "101", 2)}));
    assertDoesNotThrow(()
                           -> d_solver.mkTuple(new Sort[] {d_solver.getRealSort()},
                               new Term[] {d_solver.mkInteger("5")}));

    assertThrows(CVC5ApiException.class,
        () -> d_solver.mkTuple(new Sort[] {}, new Term[] {d_solver.mkBitVector(3, "101", 2)}));

    assertThrows(CVC5ApiException.class,
        ()
            -> d_solver.mkTuple(new Sort[] {d_solver.mkBitVectorSort(4)},
                new Term[] {d_solver.mkBitVector(3, "101", 2)}));

    assertThrows(CVC5ApiException.class,
        ()
            -> d_solver.mkTuple(
                new Sort[] {d_solver.getIntegerSort()}, new Term[] {d_solver.mkReal("5.3")}));

    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class,
        ()
            -> slv.mkTuple(new Sort[] {d_solver.mkBitVectorSort(3)},
                new Term[] {slv.mkBitVector(3, "101", 2)}));

    assertThrows(CVC5ApiException.class,
        ()
            -> slv.mkTuple(new Sort[] {slv.mkBitVectorSort(3)},
                new Term[] {d_solver.mkBitVector(3, "101", 2)}));
    slv.close();
  }

  @Test void mkUniverseSet()
  {
    assertDoesNotThrow(() -> d_solver.mkUniverseSet(d_solver.getBooleanSort()));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkUniverseSet(d_solver.getNullSort()));
    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.mkUniverseSet(d_solver.getBooleanSort()));
    slv.close();
  }

  @Test void mkConst()
  {
    Sort boolSort = d_solver.getBooleanSort();
    Sort intSort = d_solver.getIntegerSort();
    Sort funSort = d_solver.mkFunctionSort(intSort, boolSort);
    assertDoesNotThrow(() -> d_solver.mkConst(boolSort));
    assertDoesNotThrow(() -> d_solver.mkConst(funSort));
    assertDoesNotThrow(() -> d_solver.mkConst(boolSort, ("b")));
    assertDoesNotThrow(() -> d_solver.mkConst(intSort, ("i")));
    assertDoesNotThrow(() -> d_solver.mkConst(funSort, "f"));
    assertDoesNotThrow(() -> d_solver.mkConst(funSort, ""));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkConst(d_solver.getNullSort()));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkConst(d_solver.getNullSort(), "a"));

    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.mkConst(boolSort));
    slv.close();
  }

  @Test void mkConstArray()
  {
    Sort intSort = d_solver.getIntegerSort();
    Sort arrSort = d_solver.mkArraySort(intSort, intSort);
    Term zero = d_solver.mkInteger(0);
    Term constArr = d_solver.mkConstArray(arrSort, zero);

    assertDoesNotThrow(() -> d_solver.mkConstArray(arrSort, zero));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkConstArray(d_solver.getNullSort(), zero));
    assertThrows(
        CVC5ApiException.class, () -> d_solver.mkConstArray(arrSort, d_solver.getNullTerm()));
    assertThrows(
        CVC5ApiException.class, () -> d_solver.mkConstArray(arrSort, d_solver.mkBitVector(1, 1)));

    assertThrows(CVC5ApiException.class, () -> d_solver.mkConstArray(intSort, zero));
    Solver slv = new Solver();
    Term zero2 = slv.mkInteger(0);
    Sort arrSort2 = slv.mkArraySort(slv.getIntegerSort(), slv.getIntegerSort());
    assertThrows(CVC5ApiException.class, () -> slv.mkConstArray(arrSort2, zero));
    assertThrows(CVC5ApiException.class, () -> slv.mkConstArray(arrSort, zero2));
    slv.close();
  }

  @Test void declareDatatype()
  {
    DatatypeConstructorDecl nil = d_solver.mkDatatypeConstructorDecl("nil");
    DatatypeConstructorDecl[] ctors1 = new DatatypeConstructorDecl[] {nil};
    assertDoesNotThrow(() -> d_solver.declareDatatype(("a"), ctors1));
    DatatypeConstructorDecl cons = d_solver.mkDatatypeConstructorDecl("cons");
    DatatypeConstructorDecl nil2 = d_solver.mkDatatypeConstructorDecl("nil");
    DatatypeConstructorDecl[] ctors2 = new DatatypeConstructorDecl[] {cons, nil2};
    assertDoesNotThrow(() -> d_solver.declareDatatype(("b"), ctors2));
    DatatypeConstructorDecl cons2 = d_solver.mkDatatypeConstructorDecl("cons");
    DatatypeConstructorDecl nil3 = d_solver.mkDatatypeConstructorDecl("nil");
    DatatypeConstructorDecl[] ctors3 = new DatatypeConstructorDecl[] {cons2, nil3};
    assertDoesNotThrow(() -> d_solver.declareDatatype((""), ctors3));
    DatatypeConstructorDecl[] ctors4 = new DatatypeConstructorDecl[0];
    assertThrows(CVC5ApiException.class, () -> d_solver.declareDatatype(("c"), ctors4));

    assertThrows(CVC5ApiException.class, () -> d_solver.declareDatatype((""), ctors4));

    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.declareDatatype(("a"), ctors1));
    slv.close();
  }

  @Test void declareFun() throws CVC5ApiException
  {
    Sort bvSort = d_solver.mkBitVectorSort(32);
    Sort funSort =
        d_solver.mkFunctionSort(d_solver.mkUninterpretedSort("u"), d_solver.getIntegerSort());
    assertDoesNotThrow(() -> d_solver.declareFun("f1", new Sort[] {}, bvSort));
    assertDoesNotThrow(
        () -> d_solver.declareFun("f3", new Sort[] {bvSort, d_solver.getIntegerSort()}, bvSort));
    assertThrows(CVC5ApiException.class, () -> d_solver.declareFun("f2", new Sort[] {}, funSort));
    // functions as arguments is allowed
    assertDoesNotThrow(() -> d_solver.declareFun("f4", new Sort[] {bvSort, funSort}, bvSort));
    assertThrows(CVC5ApiException.class,
        () -> d_solver.declareFun("f5", new Sort[] {bvSort, bvSort}, funSort));

    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.declareFun("f1", new Sort[] {}, bvSort));
    slv.close();
  }

  @Test void declareSort()
  {
    assertDoesNotThrow(() -> d_solver.declareSort("s", 0));
    assertDoesNotThrow(() -> d_solver.declareSort("s", 2));
    assertDoesNotThrow(() -> d_solver.declareSort("", 2));
  }

  @Test void defineSort()
  {
    Sort sortVar0 = d_solver.mkParamSort("T0");
    Sort sortVar1 = d_solver.mkParamSort("T1");
    Sort intSort = d_solver.getIntegerSort();
    Sort realSort = d_solver.getRealSort();
    Sort arraySort0 = d_solver.mkArraySort(sortVar0, sortVar0);
    Sort arraySort1 = d_solver.mkArraySort(sortVar0, sortVar1);
    // Now create instantiations of the defined sorts
    assertDoesNotThrow(() -> arraySort0.substitute(sortVar0, intSort));
    assertDoesNotThrow(()
                           -> arraySort1.substitute(
                               new Sort[] {sortVar0, sortVar1}, new Sort[] {intSort, realSort}));
  }

  @Test void defineFun() throws CVC5ApiException
  {
    Sort bvSort = d_solver.mkBitVectorSort(32);
    Sort funSort =
        d_solver.mkFunctionSort(d_solver.mkUninterpretedSort("u"), d_solver.getIntegerSort());
    Term b1 = d_solver.mkVar(bvSort, "b1");
    Term b2 = d_solver.mkVar(d_solver.getIntegerSort(), "b2");
    Term b3 = d_solver.mkVar(funSort, "b3");
    Term v1 = d_solver.mkConst(bvSort, "v1");
    Term v2 = d_solver.mkConst(funSort, "v2");
    assertDoesNotThrow(() -> d_solver.defineFun("f", new Term[] {}, bvSort, v1));
    assertDoesNotThrow(() -> d_solver.defineFun("ff", new Term[] {b1, b2}, bvSort, v1));
    assertThrows(
        CVC5ApiException.class, () -> d_solver.defineFun("ff", new Term[] {v1, b2}, bvSort, v1));

    assertThrows(
        CVC5ApiException.class, () -> d_solver.defineFun("fff", new Term[] {b1}, bvSort, v2));
    assertThrows(
        CVC5ApiException.class, () -> d_solver.defineFun("ffff", new Term[] {b1}, funSort, v2));

    // b3 has function sort, which is allowed as an argument
    assertDoesNotThrow(() -> d_solver.defineFun("fffff", new Term[] {b1, b3}, bvSort, v1));

    Solver slv = new Solver();
    Sort bvSort2 = slv.mkBitVectorSort(32);
    Term v12 = slv.mkConst(bvSort2, "v1");
    Term b12 = slv.mkVar(bvSort2, "b1");
    Term b22 = slv.mkVar(slv.getIntegerSort(), "b2");
    assertThrows(CVC5ApiException.class, () -> slv.defineFun("f", new Term[] {}, bvSort, v12));
    assertThrows(CVC5ApiException.class, () -> slv.defineFun("f", new Term[] {}, bvSort2, v1));
    assertThrows(
        CVC5ApiException.class, () -> slv.defineFun("ff", new Term[] {b1, b22}, bvSort2, v12));
    assertThrows(
        CVC5ApiException.class, () -> slv.defineFun("ff", new Term[] {b12, b2}, bvSort2, v12));
    assertThrows(
        CVC5ApiException.class, () -> slv.defineFun("ff", new Term[] {b12, b22}, bvSort, v12));
    assertThrows(
        CVC5ApiException.class, () -> slv.defineFun("ff", new Term[] {b12, b22}, bvSort2, v1));
    slv.close();
  }

  @Test void defineFunGlobal()
  {
    Sort bSort = d_solver.getBooleanSort();

    Term bTrue = d_solver.mkBoolean(true);
    // (define-fun f () Bool true)
    Term f = d_solver.defineFun("f", new Term[] {}, bSort, bTrue, true);
    Term b = d_solver.mkVar(bSort, "b");
    // (define-fun g (b Bool) Bool b)
    Term g = d_solver.defineFun("g", new Term[] {b}, bSort, b, true);

    // (assert (or (not f) (not (g true))))
    d_solver.assertFormula(
        d_solver.mkTerm(OR, f.notTerm(), d_solver.mkTerm(APPLY_UF, g, bTrue).notTerm()));
    assertTrue(d_solver.checkSat().isUnsat());
    d_solver.resetAssertions();
    // (assert (or (not f) (not (g true))))
    d_solver.assertFormula(
        d_solver.mkTerm(OR, f.notTerm(), d_solver.mkTerm(APPLY_UF, g, bTrue).notTerm()));
    assertTrue(d_solver.checkSat().isUnsat());
  }

  @Test void defineFunRec() throws CVC5ApiException
  {
    Sort bvSort = d_solver.mkBitVectorSort(32);
    Sort funSort1 = d_solver.mkFunctionSort(new Sort[] {bvSort, bvSort}, bvSort);
    Sort funSort2 =
        d_solver.mkFunctionSort(d_solver.mkUninterpretedSort("u"), d_solver.getIntegerSort());
    Term b1 = d_solver.mkVar(bvSort, "b1");
    Term b11 = d_solver.mkVar(bvSort, "b1");
    Term b2 = d_solver.mkVar(d_solver.getIntegerSort(), "b2");
    Term b3 = d_solver.mkVar(funSort2, "b3");
    Term v1 = d_solver.mkConst(bvSort, "v1");
    Term v2 = d_solver.mkConst(d_solver.getIntegerSort(), "v2");
    Term v3 = d_solver.mkConst(funSort2, "v3");
    Term f1 = d_solver.mkConst(funSort1, "f1");
    Term f2 = d_solver.mkConst(funSort2, "f2");
    Term f3 = d_solver.mkConst(bvSort, "f3");
    assertDoesNotThrow(() -> d_solver.defineFunRec("f", new Term[] {}, bvSort, v1));
    assertDoesNotThrow(() -> d_solver.defineFunRec("ff", new Term[] {b1, b2}, bvSort, v1));
    assertDoesNotThrow(() -> d_solver.defineFunRec(f1, new Term[] {b1, b11}, v1));
    assertThrows(
        CVC5ApiException.class, () -> d_solver.defineFunRec("fff", new Term[] {b1}, bvSort, v3));

    assertThrows(
        CVC5ApiException.class, () -> d_solver.defineFunRec("ff", new Term[] {b1, v2}, bvSort, v1));

    assertThrows(
        CVC5ApiException.class, () -> d_solver.defineFunRec("ffff", new Term[] {b1}, funSort2, v3));

    // b3 has function sort, which is allowed as an argument
    assertDoesNotThrow(() -> d_solver.defineFunRec("fffff", new Term[] {b1, b3}, bvSort, v1));
    assertThrows(CVC5ApiException.class, () -> d_solver.defineFunRec(f1, new Term[] {b1}, v1));
    assertThrows(CVC5ApiException.class, () -> d_solver.defineFunRec(f1, new Term[] {b1, b11}, v2));
    assertThrows(CVC5ApiException.class, () -> d_solver.defineFunRec(f1, new Term[] {b1, b11}, v3));
    assertThrows(CVC5ApiException.class, () -> d_solver.defineFunRec(f2, new Term[] {b1}, v2));
    assertThrows(CVC5ApiException.class, () -> d_solver.defineFunRec(f3, new Term[] {b1}, v1));

    Solver slv = new Solver();
    Sort bvSort2 = slv.mkBitVectorSort(32);
    Term v12 = slv.mkConst(bvSort2, "v1");
    Term b12 = slv.mkVar(bvSort2, "b1");
    Term b22 = slv.mkVar(slv.getIntegerSort(), "b2");
    assertDoesNotThrow(() -> slv.defineFunRec("f", new Term[] {}, bvSort2, v12));
    assertDoesNotThrow(() -> slv.defineFunRec("ff", new Term[] {b12, b22}, bvSort2, v12));
    assertThrows(CVC5ApiException.class, () -> slv.defineFunRec("f", new Term[] {}, bvSort, v12));
    assertThrows(CVC5ApiException.class, () -> slv.defineFunRec("f", new Term[] {}, bvSort2, v1));
    assertThrows(
        CVC5ApiException.class, () -> slv.defineFunRec("ff", new Term[] {b1, b22}, bvSort2, v12));

    assertThrows(
        CVC5ApiException.class, () -> slv.defineFunRec("ff", new Term[] {b12, b2}, bvSort2, v12));

    assertThrows(
        CVC5ApiException.class, () -> slv.defineFunRec("ff", new Term[] {b12, b22}, bvSort, v12));

    assertThrows(
        CVC5ApiException.class, () -> slv.defineFunRec("ff", new Term[] {b12, b22}, bvSort2, v1));
    slv.close();
  }

  @Test void defineFunRecWrongLogic() throws CVC5ApiException
  {
    d_solver.setLogic("QF_BV");
    Sort bvSort = d_solver.mkBitVectorSort(32);
    Sort funSort = d_solver.mkFunctionSort(new Sort[] {bvSort, bvSort}, bvSort);
    Term b = d_solver.mkVar(bvSort, "b");
    Term v = d_solver.mkConst(bvSort, "v");
    Term f = d_solver.mkConst(funSort, "f");
    assertThrows(
        CVC5ApiException.class, () -> d_solver.defineFunRec("f", new Term[] {}, bvSort, v));
    assertThrows(CVC5ApiException.class, () -> d_solver.defineFunRec(f, new Term[] {b, b}, v));
  }

  @Test void defineFunRecGlobal() throws CVC5ApiException
  {
    Sort bSort = d_solver.getBooleanSort();
    Sort fSort = d_solver.mkFunctionSort(bSort, bSort);

    d_solver.push();
    Term bTrue = d_solver.mkBoolean(true);
    // (define-fun f () Bool true)
    Term f = d_solver.defineFunRec("f", new Term[] {}, bSort, bTrue, true);
    Term b = d_solver.mkVar(bSort, "b");
    Term gSym = d_solver.mkConst(fSort, "g");
    // (define-fun g (b Bool) Bool b)
    Term g = d_solver.defineFunRec(gSym, new Term[] {b}, b, true);

    // (assert (or (not f) (not (g true))))
    d_solver.assertFormula(
        d_solver.mkTerm(OR, f.notTerm(), d_solver.mkTerm(APPLY_UF, g, bTrue).notTerm()));
    assertTrue(d_solver.checkSat().isUnsat());
    d_solver.pop();
    // (assert (or (not f) (not (g true))))
    d_solver.assertFormula(
        d_solver.mkTerm(OR, f.notTerm(), d_solver.mkTerm(APPLY_UF, g, bTrue).notTerm()));
    assertTrue(d_solver.checkSat().isUnsat());
  }

  @Test void defineFunsRec() throws CVC5ApiException
  {
    Sort uSort = d_solver.mkUninterpretedSort("u");
    Sort bvSort = d_solver.mkBitVectorSort(32);
    Sort funSort1 = d_solver.mkFunctionSort(new Sort[] {bvSort, bvSort}, bvSort);
    Sort funSort2 = d_solver.mkFunctionSort(uSort, d_solver.getIntegerSort());
    Term b1 = d_solver.mkVar(bvSort, "b1");
    Term b11 = d_solver.mkVar(bvSort, "b1");
    Term b2 = d_solver.mkVar(d_solver.getIntegerSort(), "b2");
    Term b3 = d_solver.mkVar(funSort2, "b3");
    Term b4 = d_solver.mkVar(uSort, "b4");
    Term v1 = d_solver.mkConst(bvSort, "v1");
    Term v2 = d_solver.mkConst(d_solver.getIntegerSort(), "v2");
    Term v3 = d_solver.mkConst(funSort2, "v3");
    Term v4 = d_solver.mkConst(uSort, "v4");
    Term f1 = d_solver.mkConst(funSort1, "f1");
    Term f2 = d_solver.mkConst(funSort2, "f2");
    Term f3 = d_solver.mkConst(bvSort, "f3");
    assertDoesNotThrow(
        ()
            -> d_solver.defineFunsRec(
                new Term[] {f1, f2}, new Term[][] {{b1, b11}, {b4}}, new Term[] {v1, v2}));
    assertThrows(CVC5ApiException.class,
        ()
            -> d_solver.defineFunsRec(
                new Term[] {f1, f2}, new Term[][] {{v1, b11}, {b4}}, new Term[] {v1, v2}));
    assertThrows(CVC5ApiException.class,
        ()
            -> d_solver.defineFunsRec(
                new Term[] {f1, f3}, new Term[][] {{b1, b11}, {b4}}, new Term[] {v1, v2}));
    assertThrows(CVC5ApiException.class,
        ()
            -> d_solver.defineFunsRec(
                new Term[] {f1, f2}, new Term[][] {{b1}, {b4}}, new Term[] {v1, v2}));
    assertThrows(CVC5ApiException.class,
        ()
            -> d_solver.defineFunsRec(
                new Term[] {f1, f2}, new Term[][] {{b1, b2}, {b4}}, new Term[] {v1, v2}));
    assertThrows(CVC5ApiException.class,
        ()
            -> d_solver.defineFunsRec(
                new Term[] {f1, f2}, new Term[][] {{b1, b11}, {b4}}, new Term[] {v1, v4}));

    Solver slv = new Solver();
    Sort uSort2 = slv.mkUninterpretedSort("u");
    Sort bvSort2 = slv.mkBitVectorSort(32);
    Sort funSort12 = slv.mkFunctionSort(new Sort[] {bvSort2, bvSort2}, bvSort2);
    Sort funSort22 = slv.mkFunctionSort(uSort2, slv.getIntegerSort());
    Term b12 = slv.mkVar(bvSort2, "b1");
    Term b112 = slv.mkVar(bvSort2, "b1");
    Term b42 = slv.mkVar(uSort2, "b4");
    Term v12 = slv.mkConst(bvSort2, "v1");
    Term v22 = slv.mkConst(slv.getIntegerSort(), "v2");
    Term f12 = slv.mkConst(funSort12, "f1");
    Term f22 = slv.mkConst(funSort22, "f2");
    assertDoesNotThrow(
        ()
            -> slv.defineFunsRec(
                new Term[] {f12, f22}, new Term[][] {{b12, b112}, {b42}}, new Term[] {v12, v22}));
    assertThrows(CVC5ApiException.class,
        ()
            -> slv.defineFunsRec(
                new Term[] {f1, f22}, new Term[][] {{b12, b112}, {b42}}, new Term[] {v12, v22}));
    assertThrows(CVC5ApiException.class,
        ()
            -> slv.defineFunsRec(
                new Term[] {f12, f2}, new Term[][] {{b12, b112}, {b42}}, new Term[] {v12, v22}));
    assertThrows(CVC5ApiException.class,
        ()
            -> slv.defineFunsRec(
                new Term[] {f12, f22}, new Term[][] {{b1, b112}, {b42}}, new Term[] {v12, v22}));
    assertThrows(CVC5ApiException.class,
        ()
            -> slv.defineFunsRec(
                new Term[] {f12, f22}, new Term[][] {{b12, b11}, {b42}}, new Term[] {v12, v22}));
    assertThrows(CVC5ApiException.class,
        ()
            -> slv.defineFunsRec(
                new Term[] {f12, f22}, new Term[][] {{b12, b112}, {b4}}, new Term[] {v12, v22}));
    assertThrows(CVC5ApiException.class,
        ()
            -> slv.defineFunsRec(
                new Term[] {f12, f22}, new Term[][] {{b12, b112}, {b42}}, new Term[] {v1, v22}));
    assertThrows(CVC5ApiException.class,
        ()
            -> slv.defineFunsRec(
                new Term[] {f12, f22}, new Term[][] {{b12, b112}, {b42}}, new Term[] {v12, v2}));
    slv.close();
  }

  @Test void defineFunsRecWrongLogic() throws CVC5ApiException
  {
    d_solver.setLogic("QF_BV");
    Sort uSort = d_solver.mkUninterpretedSort("u");
    Sort bvSort = d_solver.mkBitVectorSort(32);
    Sort funSort1 = d_solver.mkFunctionSort(new Sort[] {bvSort, bvSort}, bvSort);
    Sort funSort2 = d_solver.mkFunctionSort(uSort, d_solver.getIntegerSort());
    Term b = d_solver.mkVar(bvSort, "b");
    Term u = d_solver.mkVar(uSort, "u");
    Term v1 = d_solver.mkConst(bvSort, "v1");
    Term v2 = d_solver.mkConst(d_solver.getIntegerSort(), "v2");
    Term f1 = d_solver.mkConst(funSort1, "f1");
    Term f2 = d_solver.mkConst(funSort2, "f2");
    assertThrows(CVC5ApiException.class,
        ()
            -> d_solver.defineFunsRec(
                new Term[] {f1, f2}, new Term[][] {{b, b}, {u}}, new Term[] {v1, v2}));
  }

  @Test void defineFunsRecGlobal() throws CVC5ApiException
  {
    Sort bSort = d_solver.getBooleanSort();
    Sort fSort = d_solver.mkFunctionSort(bSort, bSort);

    d_solver.push();
    Term bTrue = d_solver.mkBoolean(true);
    Term b = d_solver.mkVar(bSort, "b");
    Term gSym = d_solver.mkConst(fSort, "g");
    // (define-funs-rec ((g ((b Bool)) Bool)) (b))
    d_solver.defineFunsRec(new Term[] {gSym}, new Term[][] {{b}}, new Term[] {b}, true);

    // (assert (not (g true)))
    d_solver.assertFormula(d_solver.mkTerm(APPLY_UF, gSym, bTrue).notTerm());
    assertTrue(d_solver.checkSat().isUnsat());
    d_solver.pop();
    // (assert (not (g true)))
    d_solver.assertFormula(d_solver.mkTerm(APPLY_UF, gSym, bTrue).notTerm());
    assertTrue(d_solver.checkSat().isUnsat());
  }

  @Test void uFIteration()
  {
    Sort intSort = d_solver.getIntegerSort();
    Sort funSort = d_solver.mkFunctionSort(new Sort[] {intSort, intSort}, intSort);
    Term x = d_solver.mkConst(intSort, "x");
    Term y = d_solver.mkConst(intSort, "y");
    Term f = d_solver.mkConst(funSort, "f");
    Term fxy = d_solver.mkTerm(APPLY_UF, f, x, y);

    // Expecting the uninterpreted function to be one of the children
    Term expected_children[] = new Term[] {f, x, y};
    int idx = 0;
    for (Term c : fxy)
    {
      assertEquals(c, expected_children[idx]);
      idx++;
    }
  }

  @Test void getInfo()
  {
    assertDoesNotThrow(() -> d_solver.getInfo("name"));
    assertThrows(CVC5ApiException.class, () -> d_solver.getInfo("asdf"));
  }

  @Test void getInterpolant() throws CVC5ApiException
  {
    d_solver.setLogic("QF_LIA");
    d_solver.setOption("produce-interpols", "default");
    d_solver.setOption("incremental", "false");

    Sort intSort = d_solver.getIntegerSort();
    Term zero = d_solver.mkInteger(0);
    Term x = d_solver.mkConst(intSort, "x");
    Term y = d_solver.mkConst(intSort, "y");
    Term z = d_solver.mkConst(intSort, "z");

    // Assumptions for interpolation: x + y > 0 /\ x < 0
    d_solver.assertFormula(d_solver.mkTerm(GT, d_solver.mkTerm(PLUS, x, y), zero));
    d_solver.assertFormula(d_solver.mkTerm(LT, x, zero));
    // Conjecture for interpolation: y + z > 0 \/ z < 0
    Term conj = d_solver.mkTerm(
        OR, d_solver.mkTerm(GT, d_solver.mkTerm(PLUS, y, z), zero), d_solver.mkTerm(LT, z, zero));
    Term output = d_solver.getNullTerm();
    // Call the interpolation api, while the resulting interpolant is the output
    d_solver.getInterpolant(conj, output);

    // We expect the resulting output to be a boolean formula
    assertTrue(output.getSort().isBoolean());
  }

  @Test void getOp() throws CVC5ApiException
  {
    Sort bv32 = d_solver.mkBitVectorSort(32);
    Term a = d_solver.mkConst(bv32, "a");
    Op ext = d_solver.mkOp(BITVECTOR_EXTRACT, 2, 1);
    Term exta = d_solver.mkTerm(ext, a);

    assertFalse(a.hasOp());
    assertThrows(CVC5ApiException.class, () -> a.getOp());
    assertTrue(exta.hasOp());
    assertEquals(exta.getOp(), ext);

    // Test Datatypes -- more complicated
    DatatypeDecl consListSpec = d_solver.mkDatatypeDecl("list");
    DatatypeConstructorDecl cons = d_solver.mkDatatypeConstructorDecl("cons");
    cons.addSelector("head", d_solver.getIntegerSort());
    cons.addSelectorSelf("tail");
    consListSpec.addConstructor(cons);
    DatatypeConstructorDecl nil = d_solver.mkDatatypeConstructorDecl("nil");
    consListSpec.addConstructor(nil);
    Sort consListSort = d_solver.mkDatatypeSort(consListSpec);
    Datatype consList = consListSort.getDatatype();

    Term consTerm = consList.getConstructorTerm("cons");
    Term nilTerm = consList.getConstructorTerm("nil");
    Term headTerm = consList.getConstructor("cons").getSelectorTerm("head");

    Term listnil = d_solver.mkTerm(APPLY_CONSTRUCTOR, nilTerm);
    Term listcons1 = d_solver.mkTerm(APPLY_CONSTRUCTOR, consTerm, d_solver.mkInteger(1), listnil);
    Term listhead = d_solver.mkTerm(APPLY_SELECTOR, headTerm, listcons1);

    assertTrue(listnil.hasOp());
    assertTrue(listcons1.hasOp());
    assertTrue(listhead.hasOp());
  }

  @Test void getOption()
  {
    assertDoesNotThrow(() -> d_solver.getOption("incremental"));
    assertThrows(CVC5ApiException.class, () -> d_solver.getOption("asdf"));
  }

  @Test void getOptionNames()
  {
    String[] names = d_solver.getOptionNames();
    assertTrue(names.length > 100);
    assertTrue(Arrays.asList(names).contains("verbose"));
    assertFalse(Arrays.asList(names).contains("foobar"));
  }

  @Test void getOptionInfo()
  {
    List<Executable> assertions = new ArrayList<>();
    {
      assertions.add(
          () -> assertThrows(CVC5ApiException.class, () -> d_solver.getOptionInfo("asdf-invalid")));
    }
    {
      OptionInfo info = d_solver.getOptionInfo("verbose");
      assertions.add(() -> assertEquals("verbose", info.getName()));
      assertions.add(
          () -> assertEquals(Arrays.asList(new String[] {}), Arrays.asList(info.getAliases())));
      assertions.add(() -> assertTrue(info.getBaseInfo() instanceof OptionInfo.VoidInfo));
    }
    {
      // int64 type with default
      OptionInfo info = d_solver.getOptionInfo("verbosity");
      assertions.add(() -> assertEquals("verbosity", info.getName()));
      assertions.add(
          () -> assertEquals(Arrays.asList(new String[] {}), Arrays.asList(info.getAliases())));
      assertions.add(
          () -> assertTrue(info.getBaseInfo().getClass() == OptionInfo.NumberInfo.class));
      OptionInfo.NumberInfo<BigInteger> numInfo =
          (OptionInfo.NumberInfo<BigInteger>) info.getBaseInfo();
      assertions.add(() -> assertEquals(0, numInfo.getDefaultValue().intValue()));
      assertions.add(() -> assertEquals(0, numInfo.getCurrentValue().intValue()));
      assertions.add(
          () -> assertFalse(numInfo.getMinimum() != null || numInfo.getMaximum() != null));
      assertions.add(() -> assertEquals(info.intValue().intValue(), 0));
    }
    assertAll(assertions);
    {
      OptionInfo info = d_solver.getOptionInfo("random-freq");
      assertEquals(info.getName(), "random-freq");
      assertEquals(
          Arrays.asList(info.getAliases()), Arrays.asList(new String[] {"random-frequency"}));
      assertTrue(info.getBaseInfo().getClass() == OptionInfo.NumberInfo.class);
      OptionInfo.NumberInfo<Double> ni = (OptionInfo.NumberInfo<Double>) info.getBaseInfo();
      assertEquals(ni.getCurrentValue(), 0.0);
      assertEquals(ni.getDefaultValue(), 0.0);
      assertTrue(ni.getMinimum() != null && ni.getMaximum() != null);
      assertEquals(ni.getMinimum(), 0.0);
      assertEquals(ni.getMaximum(), 1.0);
      assertEquals(info.doubleValue(), 0.0);
    }
    {
      // mode option
      OptionInfo info = d_solver.getOptionInfo("output");
      assertions.clear();
      assertions.add(() -> assertEquals("output", info.getName()));
      assertions.add(
          () -> assertEquals(Arrays.asList(new String[] {}), Arrays.asList(info.getAliases())));
      assertions.add(() -> assertTrue(info.getBaseInfo().getClass() == OptionInfo.ModeInfo.class));
      OptionInfo.ModeInfo modeInfo = (OptionInfo.ModeInfo) info.getBaseInfo();
      assertions.add(() -> assertEquals("NONE", modeInfo.getDefaultValue()));
      assertions.add(() -> assertEquals("none", modeInfo.getCurrentValue()));
      assertions.add(() -> assertTrue(Arrays.asList(modeInfo.getModes()).contains("NONE")));
    }
    assertAll(assertions);
  }

  @Test void getUnsatAssumptions1()
  {
    d_solver.setOption("incremental", "false");
    d_solver.checkSatAssuming(d_solver.mkFalse());
    assertThrows(CVC5ApiException.class, () -> d_solver.getUnsatAssumptions());
  }

  @Test void getUnsatAssumptions2()
  {
    d_solver.setOption("incremental", "true");
    d_solver.setOption("produce-unsat-assumptions", "false");
    d_solver.checkSatAssuming(d_solver.mkFalse());
    assertThrows(CVC5ApiException.class, () -> d_solver.getUnsatAssumptions());
  }

  @Test void getUnsatAssumptions3()
  {
    d_solver.setOption("incremental", "true");
    d_solver.setOption("produce-unsat-assumptions", "true");
    d_solver.checkSatAssuming(d_solver.mkFalse());
    assertDoesNotThrow(() -> d_solver.getUnsatAssumptions());
    d_solver.checkSatAssuming(d_solver.mkTrue());
    assertThrows(CVC5ApiException.class, () -> d_solver.getUnsatAssumptions());
  }

  @Test void getUnsatCore1()
  {
    d_solver.setOption("incremental", "false");
    d_solver.assertFormula(d_solver.mkFalse());
    d_solver.checkSat();
    assertThrows(CVC5ApiException.class, () -> d_solver.getUnsatCore());
  }

  @Test void getUnsatCore2()
  {
    d_solver.setOption("incremental", "false");
    d_solver.setOption("produce-unsat-cores", "false");
    d_solver.assertFormula(d_solver.mkFalse());
    d_solver.checkSat();
    assertThrows(CVC5ApiException.class, () -> d_solver.getUnsatCore());
  }

  @Test void getUnsatCoreAndProof()
  {
    d_solver.setOption("incremental", "true");
    d_solver.setOption("produce-unsat-cores", "true");
    d_solver.setOption("produce-proofs", "true");

    Sort uSort = d_solver.mkUninterpretedSort("u");
    Sort intSort = d_solver.getIntegerSort();
    Sort boolSort = d_solver.getBooleanSort();
    Sort uToIntSort = d_solver.mkFunctionSort(uSort, intSort);
    Sort intPredSort = d_solver.mkFunctionSort(intSort, boolSort);

    Term x = d_solver.mkConst(uSort, "x");
    Term y = d_solver.mkConst(uSort, "y");
    Term f = d_solver.mkConst(uToIntSort, "f");
    Term p = d_solver.mkConst(intPredSort, "p");
    Term zero = d_solver.mkInteger(0);
    Term one = d_solver.mkInteger(1);
    Term f_x = d_solver.mkTerm(Kind.APPLY_UF, f, x);
    Term f_y = d_solver.mkTerm(Kind.APPLY_UF, f, y);
    Term sum = d_solver.mkTerm(Kind.PLUS, f_x, f_y);
    Term p_0 = d_solver.mkTerm(Kind.APPLY_UF, p, zero);
    Term p_f_y = d_solver.mkTerm(APPLY_UF, p, f_y);
    d_solver.assertFormula(d_solver.mkTerm(Kind.GT, zero, f_x));
    d_solver.assertFormula(d_solver.mkTerm(Kind.GT, zero, f_y));
    d_solver.assertFormula(d_solver.mkTerm(Kind.GT, sum, one));
    d_solver.assertFormula(p_0);
    d_solver.assertFormula(p_f_y.notTerm());
    assertTrue(d_solver.checkSat().isUnsat());

    Term[] unsat_core = d_solver.getUnsatCore();

    assertDoesNotThrow(() -> d_solver.getProof());

    d_solver.resetAssertions();
    for (Term t : unsat_core)
    {
      d_solver.assertFormula(t);
    }
    Result res = d_solver.checkSat();
    assertTrue(res.isUnsat());
    assertDoesNotThrow(() -> d_solver.getProof());
  }

  @Test void getDifficulty()
  {
    d_solver.setOption("produce-difficulty", "true");
    // cannot ask before a check sat
    assertThrows(CVC5ApiException.class, () -> d_solver.getDifficulty());
    d_solver.checkSat();
    assertDoesNotThrow(() -> d_solver.getDifficulty());
  }

  @Test void getDifficulty2()
  {
    d_solver.checkSat();
    // option is not set
    assertThrows(CVC5ApiException.class, () -> d_solver.getDifficulty());
  }

  @Test void getDifficulty3() throws CVC5ApiException
  {
    d_solver.setOption("produce-difficulty", "true");
    Sort intSort = d_solver.getIntegerSort();
    Term x = d_solver.mkConst(intSort, "x");
    Term zero = d_solver.mkInteger(0);
    Term ten = d_solver.mkInteger(10);
    Term f0 = d_solver.mkTerm(GEQ, x, ten);
    Term f1 = d_solver.mkTerm(GEQ, zero, x);
    d_solver.checkSat();
    Map<Term, Term> dmap = d_solver.getDifficulty();
    // difficulty should map assertions to integer values
    for (Map.Entry<Term, Term> t : dmap.entrySet())
    {
      assertTrue(t.getKey() == f0 || t.getKey() == f1);
      assertTrue(t.getValue().getKind() == Kind.CONST_RATIONAL);
    }
  }

  @Test void getValue1()
  {
    d_solver.setOption("produce-models", "false");
    Term t = d_solver.mkTrue();
    d_solver.assertFormula(t);
    d_solver.checkSat();
    assertThrows(CVC5ApiException.class, () -> d_solver.getValue(t));
  }

  @Test void getValue2()
  {
    d_solver.setOption("produce-models", "true");
    Term t = d_solver.mkFalse();
    d_solver.assertFormula(t);
    d_solver.checkSat();
    assertThrows(CVC5ApiException.class, () -> d_solver.getValue(t));
  }

  @Test void getValue3()
  {
    d_solver.setOption("produce-models", "true");
    Sort uSort = d_solver.mkUninterpretedSort("u");
    Sort intSort = d_solver.getIntegerSort();
    Sort boolSort = d_solver.getBooleanSort();
    Sort uToIntSort = d_solver.mkFunctionSort(uSort, intSort);
    Sort intPredSort = d_solver.mkFunctionSort(intSort, boolSort);
    Term[] unsat_core;

    Term x = d_solver.mkConst(uSort, "x");
    Term y = d_solver.mkConst(uSort, "y");
    Term z = d_solver.mkConst(uSort, "z");
    Term f = d_solver.mkConst(uToIntSort, "f");
    Term p = d_solver.mkConst(intPredSort, "p");
    Term zero = d_solver.mkInteger(0);
    Term one = d_solver.mkInteger(1);
    Term f_x = d_solver.mkTerm(APPLY_UF, f, x);
    Term f_y = d_solver.mkTerm(APPLY_UF, f, y);
    Term sum = d_solver.mkTerm(PLUS, f_x, f_y);
    Term p_0 = d_solver.mkTerm(APPLY_UF, p, zero);
    Term p_f_y = d_solver.mkTerm(APPLY_UF, p, f_y);

    d_solver.assertFormula(d_solver.mkTerm(LEQ, zero, f_x));
    d_solver.assertFormula(d_solver.mkTerm(LEQ, zero, f_y));
    d_solver.assertFormula(d_solver.mkTerm(LEQ, sum, one));
    d_solver.assertFormula(p_0.notTerm());
    d_solver.assertFormula(p_f_y);
    assertTrue(d_solver.checkSat().isSat());
    assertDoesNotThrow(() -> d_solver.getValue(x));
    assertDoesNotThrow(() -> d_solver.getValue(y));
    assertDoesNotThrow(() -> d_solver.getValue(z));
    assertDoesNotThrow(() -> d_solver.getValue(sum));
    assertDoesNotThrow(() -> d_solver.getValue(p_f_y));

    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.getValue(x));
    slv.close();
  }

  @Test void getModelDomainElements()
  {
    d_solver.setOption("produce-models", "true");
    Sort uSort = d_solver.mkUninterpretedSort("u");
    Sort intSort = d_solver.getIntegerSort();
    Term x = d_solver.mkConst(uSort, "x");
    Term y = d_solver.mkConst(uSort, "y");
    Term z = d_solver.mkConst(uSort, "z");
    Term f = d_solver.mkTerm(DISTINCT, x, y, z);
    d_solver.assertFormula(f);
    d_solver.checkSat();
    assertDoesNotThrow(() -> d_solver.getModelDomainElements(uSort));
    assertTrue(d_solver.getModelDomainElements(uSort).length >= 3);
    assertThrows(CVC5ApiException.class, () -> d_solver.getModelDomainElements(intSort));
  }

  @Test void getModelDomainElements2()
  {
    d_solver.setOption("produce-models", "true");
    d_solver.setOption("finite-model-find", "true");
    Sort uSort = d_solver.mkUninterpretedSort("u");
    Term x = d_solver.mkVar(uSort, "x");
    Term y = d_solver.mkVar(uSort, "y");
    Term eq = d_solver.mkTerm(EQUAL, x, y);
    Term bvl = d_solver.mkTerm(BOUND_VAR_LIST, x, y);
    Term f = d_solver.mkTerm(FORALL, bvl, eq);
    d_solver.assertFormula(f);
    d_solver.checkSat();
    assertDoesNotThrow(() -> d_solver.getModelDomainElements(uSort));
    // a model for the above must interpret u as size 1
    assertTrue(d_solver.getModelDomainElements(uSort).length == 1);
  }

  @Test void isModelCoreSymbol()
  {
    d_solver.setOption("produce-models", "true");
    d_solver.setOption("model-cores", "simple");
    Sort uSort = d_solver.mkUninterpretedSort("u");
    Term x = d_solver.mkConst(uSort, "x");
    Term y = d_solver.mkConst(uSort, "y");
    Term z = d_solver.mkConst(uSort, "z");
    Term zero = d_solver.mkInteger(0);
    Term f = d_solver.mkTerm(NOT, d_solver.mkTerm(EQUAL, x, y));
    d_solver.assertFormula(f);
    d_solver.checkSat();
    assertTrue(d_solver.isModelCoreSymbol(x));
    assertTrue(d_solver.isModelCoreSymbol(y));
    assertFalse(d_solver.isModelCoreSymbol(z));
    assertThrows(CVC5ApiException.class, () -> d_solver.isModelCoreSymbol(zero));
  }

  @Test void getModel()
  {
    d_solver.setOption("produce-models", "true");
    Sort uSort = d_solver.mkUninterpretedSort("u");
    Term x = d_solver.mkConst(uSort, "x");
    Term y = d_solver.mkConst(uSort, "y");
    Term z = d_solver.mkConst(uSort, "z");
    Term f = d_solver.mkTerm(NOT, d_solver.mkTerm(EQUAL, x, y));
    d_solver.assertFormula(f);
    d_solver.checkSat();
    Sort[] sorts = new Sort[] {uSort};
    Term[] terms = new Term[] {x, y};
    assertDoesNotThrow(() -> d_solver.getModel(sorts, terms));
    Term nullTerm = d_solver.getNullTerm();
    Term[] terms2 = new Term[] {x, y, nullTerm};
    assertThrows(CVC5ApiException.class, () -> d_solver.getModel(sorts, terms2));
  }

  @Test void getModel2()
  {
    d_solver.setOption("produce-models", "true");
    Sort[] sorts = new Sort[] {};
    Term[] terms = new Term[] {};
    assertThrows(CVC5ApiException.class, () -> d_solver.getModel(sorts, terms));
  }

  @Test void getModel3()
  {
    d_solver.setOption("produce-models", "true");
    Sort[] sorts = new Sort[] {};
    final Term[] terms = new Term[] {};
    d_solver.checkSat();
    assertDoesNotThrow(() -> d_solver.getModel(sorts, terms));
    Sort integer = d_solver.getIntegerSort();
    Sort[] sorts2 = new Sort[] {integer};
    assertThrows(CVC5ApiException.class, () -> d_solver.getModel(sorts2, terms));
  }

  @Test void getQuantifierElimination()
  {
    Term x = d_solver.mkVar(d_solver.getBooleanSort(), "x");
    Term forall = d_solver.mkTerm(FORALL,
        d_solver.mkTerm(BOUND_VAR_LIST, x),
        d_solver.mkTerm(OR, x, d_solver.mkTerm(NOT, x)));
    assertThrows(
        CVC5ApiException.class, () -> d_solver.getQuantifierElimination(d_solver.getNullTerm()));
    Solver slv = new Solver();
    assertThrows(
        CVC5ApiException.class, () -> d_solver.getQuantifierElimination(slv.mkBoolean(false)));

    assertDoesNotThrow(() -> d_solver.getQuantifierElimination(forall));
    slv.close();
  }

  @Test void getQuantifierEliminationDisjunct()
  {
    Term x = d_solver.mkVar(d_solver.getBooleanSort(), "x");
    Term forall = d_solver.mkTerm(FORALL,
        d_solver.mkTerm(BOUND_VAR_LIST, x),
        d_solver.mkTerm(OR, x, d_solver.mkTerm(NOT, x)));
    assertThrows(CVC5ApiException.class,
        () -> d_solver.getQuantifierEliminationDisjunct(d_solver.getNullTerm()));

    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class,
        () -> d_solver.getQuantifierEliminationDisjunct(slv.mkBoolean(false)));

    assertDoesNotThrow(() -> d_solver.getQuantifierEliminationDisjunct(forall));
    slv.close();
  }

  @Test void declareSepHeap() throws CVC5ApiException
  {
    d_solver.setLogic("ALL");
    Sort integer = d_solver.getIntegerSort();
    assertDoesNotThrow(() -> d_solver.declareSepHeap(integer, integer));
    // cannot declare separation logic heap more than once
    assertThrows(CVC5ApiException.class, () -> d_solver.declareSepHeap(integer, integer));
  }

  /**
   * Helper function for testGetSeparation{Heap,Nil}TermX. Asserts and checks
   * some simple separation logic constraints.
   */

  void checkSimpleSeparationConstraints(Solver solver)
  {
    Sort integer = solver.getIntegerSort();
    // declare the separation heap
    solver.declareSepHeap(integer, integer);
    Term x = solver.mkConst(integer, "x");
    Term p = solver.mkConst(integer, "p");
    Term heap = solver.mkTerm(SEP_PTO, p, x);
    solver.assertFormula(heap);
    Term nil = solver.mkSepNil(integer);
    solver.assertFormula(nil.eqTerm(solver.mkReal(5)));
    solver.checkSat();
  }

  @Test void getValueSepHeap1() throws CVC5ApiException
  {
    d_solver.setLogic("QF_BV");
    d_solver.setOption("incremental", "false");
    d_solver.setOption("produce-models", "true");
    Term t = d_solver.mkTrue();
    d_solver.assertFormula(t);
    assertThrows(CVC5ApiException.class, () -> d_solver.getValueSepHeap());
  }

  @Test void getValueSepHeap2() throws CVC5ApiException
  {
    d_solver.setLogic("ALL");
    d_solver.setOption("incremental", "false");
    d_solver.setOption("produce-models", "false");
    checkSimpleSeparationConstraints(d_solver);
    assertThrows(CVC5ApiException.class, () -> d_solver.getValueSepHeap());
  }

  @Test void getValueSepHeap3() throws CVC5ApiException
  {
    d_solver.setLogic("ALL");
    d_solver.setOption("incremental", "false");
    d_solver.setOption("produce-models", "true");
    Term t = d_solver.mkFalse();
    d_solver.assertFormula(t);
    d_solver.checkSat();
    assertThrows(CVC5ApiException.class, () -> d_solver.getValueSepHeap());
  }

  @Test void getValueSepHeap4() throws CVC5ApiException
  {
    d_solver.setLogic("ALL");
    d_solver.setOption("incremental", "false");
    d_solver.setOption("produce-models", "true");
    Term t = d_solver.mkTrue();
    d_solver.assertFormula(t);
    d_solver.checkSat();
    assertThrows(CVC5ApiException.class, () -> d_solver.getValueSepHeap());
  }

  @Test void getValueSepHeap5() throws CVC5ApiException
  {
    d_solver.setLogic("ALL");
    d_solver.setOption("incremental", "false");
    d_solver.setOption("produce-models", "true");
    checkSimpleSeparationConstraints(d_solver);
    assertDoesNotThrow(() -> d_solver.getValueSepHeap());
  }

  @Test void getValueSepNil1() throws CVC5ApiException
  {
    d_solver.setLogic("QF_BV");
    d_solver.setOption("incremental", "false");
    d_solver.setOption("produce-models", "true");
    Term t = d_solver.mkTrue();
    d_solver.assertFormula(t);
    assertThrows(CVC5ApiException.class, () -> d_solver.getValueSepNil());
  }

  @Test void getValueSepNil2() throws CVC5ApiException
  {
    d_solver.setLogic("ALL");
    d_solver.setOption("incremental", "false");
    d_solver.setOption("produce-models", "false");
    checkSimpleSeparationConstraints(d_solver);
    assertThrows(CVC5ApiException.class, () -> d_solver.getValueSepNil());
  }

  @Test void getValueSepNil3() throws CVC5ApiException
  {
    d_solver.setLogic("ALL");
    d_solver.setOption("incremental", "false");
    d_solver.setOption("produce-models", "true");
    Term t = d_solver.mkFalse();
    d_solver.assertFormula(t);
    d_solver.checkSat();
    assertThrows(CVC5ApiException.class, () -> d_solver.getValueSepNil());
  }

  @Test void getValueSepNil4() throws CVC5ApiException
  {
    d_solver.setLogic("ALL");
    d_solver.setOption("incremental", "false");
    d_solver.setOption("produce-models", "true");
    Term t = d_solver.mkTrue();
    d_solver.assertFormula(t);
    d_solver.checkSat();
    assertThrows(CVC5ApiException.class, () -> d_solver.getValueSepNil());
  }

  @Test void getValueSepNil5() throws CVC5ApiException
  {
    d_solver.setLogic("ALL");
    d_solver.setOption("incremental", "false");
    d_solver.setOption("produce-models", "true");
    checkSimpleSeparationConstraints(d_solver);
    assertDoesNotThrow(() -> d_solver.getValueSepNil());
  }

  @Test void push1()
  {
    d_solver.setOption("incremental", "true");
    assertDoesNotThrow(() -> d_solver.push(1));
    assertThrows(CVC5ApiException.class, () -> d_solver.setOption("incremental", "false"));
    assertThrows(CVC5ApiException.class, () -> d_solver.setOption("incremental", "true"));
  }

  @Test void push2()
  {
    d_solver.setOption("incremental", "false");
    assertThrows(CVC5ApiException.class, () -> d_solver.push(1));
  }

  @Test void pop1()
  {
    d_solver.setOption("incremental", "false");
    assertThrows(CVC5ApiException.class, () -> d_solver.pop(1));
  }

  @Test void pop2()
  {
    d_solver.setOption("incremental", "true");
    assertThrows(CVC5ApiException.class, () -> d_solver.pop(1));
  }

  @Test void pop3()
  {
    d_solver.setOption("incremental", "true");
    assertDoesNotThrow(() -> d_solver.push(1));
    assertDoesNotThrow(() -> d_solver.pop(1));
    assertThrows(CVC5ApiException.class, () -> d_solver.pop(1));
  }

  @Test void blockModel1()
  {
    d_solver.setOption("produce-models", "true");
    Term x = d_solver.mkConst(d_solver.getBooleanSort(), "x");
    d_solver.assertFormula(x.eqTerm(x));
    d_solver.checkSat();
    assertThrows(CVC5ApiException.class, () -> d_solver.blockModel());
  }

  @Test void blockModel2() throws CVC5ApiException
  {
    d_solver.setOption("block-models", "literals");
    Term x = d_solver.mkConst(d_solver.getBooleanSort(), "x");
    d_solver.assertFormula(x.eqTerm(x));
    d_solver.checkSat();
    assertThrows(CVC5ApiException.class, () -> d_solver.blockModel());
  }

  @Test void blockModel3() throws CVC5ApiException
  {
    d_solver.setOption("produce-models", "true");
    d_solver.setOption("block-models", "literals");
    Term x = d_solver.mkConst(d_solver.getBooleanSort(), "x");
    d_solver.assertFormula(x.eqTerm(x));
    assertThrows(CVC5ApiException.class, () -> d_solver.blockModel());
  }

  @Test void blockModel4() throws CVC5ApiException
  {
    d_solver.setOption("produce-models", "true");
    d_solver.setOption("block-models", "literals");
    Term x = d_solver.mkConst(d_solver.getBooleanSort(), "x");
    d_solver.assertFormula(x.eqTerm(x));
    d_solver.checkSat();
    assertDoesNotThrow(() -> d_solver.blockModel());
  }

  @Test void blockModelValues1() throws CVC5ApiException
  {
    d_solver.setOption("produce-models", "true");
    d_solver.setOption("block-models", "literals");
    Term x = d_solver.mkConst(d_solver.getBooleanSort(), "x");
    d_solver.assertFormula(x.eqTerm(x));
    d_solver.checkSat();
    assertThrows(CVC5ApiException.class, () -> d_solver.blockModelValues(new Term[] {}));
    assertThrows(CVC5ApiException.class,
        () -> d_solver.blockModelValues(new Term[] {d_solver.getNullTerm()}));
    Solver slv = new Solver();
    assertThrows(
        CVC5ApiException.class, () -> d_solver.blockModelValues(new Term[] {slv.mkBoolean(false)}));
    slv.close();
  }

  @Test void blockModelValues2() throws CVC5ApiException
  {
    d_solver.setOption("produce-models", "true");
    Term x = d_solver.mkConst(d_solver.getBooleanSort(), "x");
    d_solver.assertFormula(x.eqTerm(x));
    d_solver.checkSat();
    assertThrows(CVC5ApiException.class, () -> d_solver.blockModelValues(new Term[] {x}));
  }

  @Test void blockModelValues3() throws CVC5ApiException
  {
    d_solver.setOption("block-models", "literals");
    Term x = d_solver.mkConst(d_solver.getBooleanSort(), "x");
    d_solver.assertFormula(x.eqTerm(x));
    d_solver.checkSat();
    assertThrows(CVC5ApiException.class, () -> d_solver.blockModelValues(new Term[] {x}));
  }

  @Test void blockModelValues4() throws CVC5ApiException
  {
    d_solver.setOption("produce-models", "true");
    d_solver.setOption("block-models", "literals");
    Term x = d_solver.mkConst(d_solver.getBooleanSort(), "x");
    d_solver.assertFormula(x.eqTerm(x));
    assertThrows(CVC5ApiException.class, () -> d_solver.blockModelValues(new Term[] {x}));
  }

  @Test void blockModelValues5() throws CVC5ApiException
  {
    d_solver.setOption("produce-models", "true");
    d_solver.setOption("block-models", "literals");
    Term x = d_solver.mkConst(d_solver.getBooleanSort(), "x");
    d_solver.assertFormula(x.eqTerm(x));
    d_solver.checkSat();
    assertDoesNotThrow(() -> d_solver.blockModelValues(new Term[] {x}));
  }

  @Test void setInfo() throws CVC5ApiException
  {
    assertThrows(CVC5ApiException.class, () -> d_solver.setInfo("cvc4-lagic", "QF_BV"));
    assertThrows(CVC5ApiException.class, () -> d_solver.setInfo("cvc2-logic", "QF_BV"));
    assertThrows(CVC5ApiException.class, () -> d_solver.setInfo("cvc4-logic", "asdf"));

    assertDoesNotThrow(() -> d_solver.setInfo("source", "asdf"));
    assertDoesNotThrow(() -> d_solver.setInfo("category", "asdf"));
    assertDoesNotThrow(() -> d_solver.setInfo("difficulty", "asdf"));
    assertDoesNotThrow(() -> d_solver.setInfo("filename", "asdf"));
    assertDoesNotThrow(() -> d_solver.setInfo("license", "asdf"));
    assertDoesNotThrow(() -> d_solver.setInfo("name", "asdf"));
    assertDoesNotThrow(() -> d_solver.setInfo("notes", "asdf"));

    assertDoesNotThrow(() -> d_solver.setInfo("smt-lib-version", "2"));
    assertDoesNotThrow(() -> d_solver.setInfo("smt-lib-version", "2.0"));
    assertDoesNotThrow(() -> d_solver.setInfo("smt-lib-version", "2.5"));
    assertDoesNotThrow(() -> d_solver.setInfo("smt-lib-version", "2.6"));
    assertThrows(CVC5ApiException.class, () -> d_solver.setInfo("smt-lib-version", ".0"));

    assertDoesNotThrow(() -> d_solver.setInfo("status", "sat"));
    assertDoesNotThrow(() -> d_solver.setInfo("status", "unsat"));
    assertDoesNotThrow(() -> d_solver.setInfo("status", "unknown"));
    assertThrows(CVC5ApiException.class, () -> d_solver.setInfo("status", "asdf"));
  }

  @Test void simplify() throws CVC5ApiException
  {
    assertThrows(CVC5ApiException.class, () -> d_solver.simplify(d_solver.getNullTerm()));

    Sort bvSort = d_solver.mkBitVectorSort(32);
    Sort uSort = d_solver.mkUninterpretedSort("u");
    Sort funSort1 = d_solver.mkFunctionSort(new Sort[] {bvSort, bvSort}, bvSort);
    Sort funSort2 = d_solver.mkFunctionSort(uSort, d_solver.getIntegerSort());
    DatatypeDecl consListSpec = d_solver.mkDatatypeDecl("list");
    DatatypeConstructorDecl cons = d_solver.mkDatatypeConstructorDecl("cons");
    cons.addSelector("head", d_solver.getIntegerSort());
    cons.addSelectorSelf("tail");
    consListSpec.addConstructor(cons);
    DatatypeConstructorDecl nil = d_solver.mkDatatypeConstructorDecl("nil");
    consListSpec.addConstructor(nil);
    Sort consListSort = d_solver.mkDatatypeSort(consListSpec);

    Term x = d_solver.mkConst(bvSort, "x");
    assertDoesNotThrow(() -> d_solver.simplify(x));
    Term a = d_solver.mkConst(bvSort, "a");
    assertDoesNotThrow(() -> d_solver.simplify(a));
    Term b = d_solver.mkConst(bvSort, "b");
    assertDoesNotThrow(() -> d_solver.simplify(b));
    Term x_eq_x = d_solver.mkTerm(EQUAL, x, x);
    assertDoesNotThrow(() -> d_solver.simplify(x_eq_x));
    assertNotEquals(d_solver.mkTrue(), x_eq_x);
    assertEquals(d_solver.mkTrue(), d_solver.simplify(x_eq_x));
    Term x_eq_b = d_solver.mkTerm(EQUAL, x, b);
    assertDoesNotThrow(() -> d_solver.simplify(x_eq_b));
    assertNotEquals(d_solver.mkTrue(), x_eq_b);
    assertNotEquals(d_solver.mkTrue(), d_solver.simplify(x_eq_b));
    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.simplify(x));

    Term i1 = d_solver.mkConst(d_solver.getIntegerSort(), "i1");
    assertDoesNotThrow(() -> d_solver.simplify(i1));
    Term i2 = d_solver.mkTerm(MULT, i1, d_solver.mkInteger("23"));
    assertDoesNotThrow(() -> d_solver.simplify(i2));
    assertNotEquals(i1, i2);
    assertNotEquals(i1, d_solver.simplify(i2));
    Term i3 = d_solver.mkTerm(PLUS, i1, d_solver.mkInteger(0));
    assertDoesNotThrow(() -> d_solver.simplify(i3));
    assertNotEquals(i1, i3);
    assertEquals(i1, d_solver.simplify(i3));

    Datatype consList = consListSort.getDatatype();
    Term dt1 = d_solver.mkTerm(APPLY_CONSTRUCTOR,
        consList.getConstructorTerm("cons"),
        d_solver.mkInteger(0),
        d_solver.mkTerm(APPLY_CONSTRUCTOR, consList.getConstructorTerm("nil")));
    assertDoesNotThrow(() -> d_solver.simplify(dt1));
    Term dt2 = d_solver.mkTerm(
        APPLY_SELECTOR, consList.getConstructor("cons").getSelectorTerm("head"), dt1);
    assertDoesNotThrow(() -> d_solver.simplify(dt2));

    Term b1 = d_solver.mkVar(bvSort, "b1");
    assertDoesNotThrow(() -> d_solver.simplify(b1));
    Term b2 = d_solver.mkVar(bvSort, "b1");
    assertDoesNotThrow(() -> d_solver.simplify(b2));
    Term b3 = d_solver.mkVar(uSort, "b3");
    assertDoesNotThrow(() -> d_solver.simplify(b3));
    Term v1 = d_solver.mkConst(bvSort, "v1");
    assertDoesNotThrow(() -> d_solver.simplify(v1));
    Term v2 = d_solver.mkConst(d_solver.getIntegerSort(), "v2");
    assertDoesNotThrow(() -> d_solver.simplify(v2));
    Term f1 = d_solver.mkConst(funSort1, "f1");
    assertDoesNotThrow(() -> d_solver.simplify(f1));
    Term f2 = d_solver.mkConst(funSort2, "f2");
    assertDoesNotThrow(() -> d_solver.simplify(f2));
    d_solver.defineFunsRec(new Term[] {f1, f2}, new Term[][] {{b1, b2}, {b3}}, new Term[] {v1, v2});
    assertDoesNotThrow(() -> d_solver.simplify(f1));
    assertDoesNotThrow(() -> d_solver.simplify(f2));
    slv.close();
  }

  @Test void assertFormula()
  {
    assertDoesNotThrow(() -> d_solver.assertFormula(d_solver.mkTrue()));
    assertThrows(CVC5ApiException.class, () -> d_solver.assertFormula(d_solver.getNullTerm()));
    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.assertFormula(d_solver.mkTrue()));
    slv.close();
  }

  @Test void checkEntailed()
  {
    d_solver.setOption("incremental", "false");
    assertDoesNotThrow(() -> d_solver.checkEntailed(d_solver.mkTrue()));
    assertThrows(CVC5ApiException.class, () -> d_solver.checkEntailed(d_solver.mkTrue()));
    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.checkEntailed(d_solver.mkTrue()));
    slv.close();
  }

  @Test void checkEntailed1()
  {
    Sort boolSort = d_solver.getBooleanSort();
    Term x = d_solver.mkConst(boolSort, "x");
    Term y = d_solver.mkConst(boolSort, "y");
    Term z = d_solver.mkTerm(AND, x, y);
    d_solver.setOption("incremental", "true");
    assertDoesNotThrow(() -> d_solver.checkEntailed(d_solver.mkTrue()));
    assertThrows(CVC5ApiException.class, () -> d_solver.checkEntailed(d_solver.getNullTerm()));
    assertDoesNotThrow(() -> d_solver.checkEntailed(d_solver.mkTrue()));
    assertDoesNotThrow(() -> d_solver.checkEntailed(z));
    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.checkEntailed(d_solver.mkTrue()));
    slv.close();
  }

  @Test void checkEntailed2()
  {
    d_solver.setOption("incremental", "true");

    Sort uSort = d_solver.mkUninterpretedSort("u");
    Sort intSort = d_solver.getIntegerSort();
    Sort boolSort = d_solver.getBooleanSort();
    Sort uToIntSort = d_solver.mkFunctionSort(uSort, intSort);
    Sort intPredSort = d_solver.mkFunctionSort(intSort, boolSort);

    Term n = d_solver.getNullTerm();
    // Constants
    Term x = d_solver.mkConst(uSort, "x");
    Term y = d_solver.mkConst(uSort, "y");
    // Functions
    Term f = d_solver.mkConst(uToIntSort, "f");
    Term p = d_solver.mkConst(intPredSort, "p");
    // Values
    Term zero = d_solver.mkInteger(0);
    Term one = d_solver.mkInteger(1);
    // Terms
    Term f_x = d_solver.mkTerm(APPLY_UF, f, x);
    Term f_y = d_solver.mkTerm(APPLY_UF, f, y);
    Term sum = d_solver.mkTerm(PLUS, f_x, f_y);
    Term p_0 = d_solver.mkTerm(APPLY_UF, p, zero);
    Term p_f_y = d_solver.mkTerm(APPLY_UF, p, f_y);
    // Assertions
    Term assertions = d_solver.mkTerm(AND,
        new Term[] {
            d_solver.mkTerm(LEQ, zero, f_x), // 0 <= f(x)
            d_solver.mkTerm(LEQ, zero, f_y), // 0 <= f(y)
            d_solver.mkTerm(LEQ, sum, one), // f(x) + f(y) <= 1
            p_0.notTerm(), // not p(0)
            p_f_y // p(f(y))
        });

    assertDoesNotThrow(() -> d_solver.checkEntailed(d_solver.mkTrue()));
    d_solver.assertFormula(assertions);
    assertDoesNotThrow(() -> d_solver.checkEntailed(d_solver.mkTerm(DISTINCT, x, y)));
    assertDoesNotThrow(()
                           -> d_solver.checkEntailed(
                               new Term[] {d_solver.mkFalse(), d_solver.mkTerm(DISTINCT, x, y)}));
    assertThrows(CVC5ApiException.class, () -> d_solver.checkEntailed(n));
    assertThrows(CVC5ApiException.class,
        () -> d_solver.checkEntailed(new Term[] {n, d_solver.mkTerm(DISTINCT, x, y)}));

    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.checkEntailed(d_solver.mkTrue()));
    slv.close();
  }

  @Test void checkSat() throws CVC5ApiException
  {
    d_solver.setOption("incremental", "false");
    assertDoesNotThrow(() -> d_solver.checkSat());
    assertThrows(CVC5ApiException.class, () -> d_solver.checkSat());
  }

  @Test void checkSatAssuming() throws CVC5ApiException
  {
    d_solver.setOption("incremental", "false");
    assertDoesNotThrow(() -> d_solver.checkSatAssuming(d_solver.mkTrue()));
    assertThrows(CVC5ApiException.class, () -> d_solver.checkSatAssuming(d_solver.mkTrue()));
    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.checkSatAssuming(d_solver.mkTrue()));
    slv.close();
  }

  @Test void checkSatAssuming1() throws CVC5ApiException
  {
    Sort boolSort = d_solver.getBooleanSort();
    Term x = d_solver.mkConst(boolSort, "x");
    Term y = d_solver.mkConst(boolSort, "y");
    Term z = d_solver.mkTerm(AND, x, y);
    d_solver.setOption("incremental", "true");
    assertDoesNotThrow(() -> d_solver.checkSatAssuming(d_solver.mkTrue()));
    assertThrows(CVC5ApiException.class, () -> d_solver.checkSatAssuming(d_solver.getNullTerm()));
    assertDoesNotThrow(() -> d_solver.checkSatAssuming(d_solver.mkTrue()));
    assertDoesNotThrow(() -> d_solver.checkSatAssuming(z));
    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.checkSatAssuming(d_solver.mkTrue()));
    slv.close();
  }

  @Test void checkSatAssuming2() throws CVC5ApiException
  {
    d_solver.setOption("incremental", "true");

    Sort uSort = d_solver.mkUninterpretedSort("u");
    Sort intSort = d_solver.getIntegerSort();
    Sort boolSort = d_solver.getBooleanSort();
    Sort uToIntSort = d_solver.mkFunctionSort(uSort, intSort);
    Sort intPredSort = d_solver.mkFunctionSort(intSort, boolSort);

    Term n = d_solver.getNullTerm();
    // Constants
    Term x = d_solver.mkConst(uSort, "x");
    Term y = d_solver.mkConst(uSort, "y");
    // Functions
    Term f = d_solver.mkConst(uToIntSort, "f");
    Term p = d_solver.mkConst(intPredSort, "p");
    // Values
    Term zero = d_solver.mkInteger(0);
    Term one = d_solver.mkInteger(1);
    // Terms
    Term f_x = d_solver.mkTerm(APPLY_UF, f, x);
    Term f_y = d_solver.mkTerm(APPLY_UF, f, y);
    Term sum = d_solver.mkTerm(PLUS, f_x, f_y);
    Term p_0 = d_solver.mkTerm(APPLY_UF, p, zero);
    Term p_f_y = d_solver.mkTerm(APPLY_UF, p, f_y);
    // Assertions
    Term assertions = d_solver.mkTerm(AND,
        new Term[] {
            d_solver.mkTerm(LEQ, zero, f_x), // 0 <= f(x)
            d_solver.mkTerm(LEQ, zero, f_y), // 0 <= f(y)
            d_solver.mkTerm(LEQ, sum, one), // f(x) + f(y) <= 1
            p_0.notTerm(), // not p(0)
            p_f_y // p(f(y))
        });

    assertDoesNotThrow(() -> d_solver.checkSatAssuming(d_solver.mkTrue()));
    d_solver.assertFormula(assertions);
    assertDoesNotThrow(() -> d_solver.checkSatAssuming(d_solver.mkTerm(DISTINCT, x, y)));
    assertDoesNotThrow(()
                           -> d_solver.checkSatAssuming(
                               new Term[] {d_solver.mkFalse(), d_solver.mkTerm(DISTINCT, x, y)}));
    assertThrows(CVC5ApiException.class, () -> d_solver.checkSatAssuming(n));
    assertThrows(CVC5ApiException.class,
        () -> d_solver.checkSatAssuming(new Term[] {n, d_solver.mkTerm(DISTINCT, x, y)}));

    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.checkSatAssuming(d_solver.mkTrue()));
    slv.close();
  }

  @Test void setLogic() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.setLogic("AUFLIRA"));
    assertThrows(CVC5ApiException.class, () -> d_solver.setLogic("AF_BV"));
    d_solver.assertFormula(d_solver.mkTrue());
    assertThrows(CVC5ApiException.class, () -> d_solver.setLogic("AUFLIRA"));
  }

  @Test void setOption() throws CVC5ApiException
  {
    assertDoesNotThrow(() -> d_solver.setOption("bv-sat-solver", "minisat"));
    assertThrows(CVC5ApiException.class, () -> d_solver.setOption("bv-sat-solver", "1"));
    d_solver.assertFormula(d_solver.mkTrue());
    assertThrows(CVC5ApiException.class, () -> d_solver.setOption("bv-sat-solver", "minisat"));
  }

  @Test void resetAssertions() throws CVC5ApiException
  {
    d_solver.setOption("incremental", "true");

    Sort bvSort = d_solver.mkBitVectorSort(4);
    Term one = d_solver.mkBitVector(4, 1);
    Term x = d_solver.mkConst(bvSort, "x");
    Term ule = d_solver.mkTerm(BITVECTOR_ULE, x, one);
    Term srem = d_solver.mkTerm(BITVECTOR_SREM, one, x);
    d_solver.push(4);
    Term slt = d_solver.mkTerm(BITVECTOR_SLT, srem, one);
    d_solver.resetAssertions();
    d_solver.checkSatAssuming(new Term[] {slt, ule});
  }

  @Test void mkSygusVar() throws CVC5ApiException
  {
    Sort boolSort = d_solver.getBooleanSort();
    Sort intSort = d_solver.getIntegerSort();
    Sort funSort = d_solver.mkFunctionSort(intSort, boolSort);

    assertDoesNotThrow(() -> d_solver.mkSygusVar(boolSort));
    assertDoesNotThrow(() -> d_solver.mkSygusVar(funSort));
    assertDoesNotThrow(() -> d_solver.mkSygusVar(boolSort, ("b")));
    assertDoesNotThrow(() -> d_solver.mkSygusVar(funSort, ""));

    assertThrows(CVC5ApiException.class, () -> d_solver.mkSygusVar(d_solver.getNullSort()));
    assertThrows(CVC5ApiException.class, () -> d_solver.mkSygusVar(d_solver.getNullSort(), "a"));

    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.mkSygusVar(boolSort));
    slv.close();
  }

  @Test void mkSygusGrammar() throws CVC5ApiException
  {
    Term nullTerm = d_solver.getNullTerm();
    Term boolTerm = d_solver.mkBoolean(true);
    Term boolVar = d_solver.mkVar(d_solver.getBooleanSort());
    Term intVar = d_solver.mkVar(d_solver.getIntegerSort());

    assertDoesNotThrow(() -> d_solver.mkSygusGrammar(new Term[] {}, new Term[] {intVar}));
    assertDoesNotThrow(() -> d_solver.mkSygusGrammar(new Term[] {boolVar}, new Term[] {intVar}));

    assertThrows(
        CVC5ApiException.class, () -> d_solver.mkSygusGrammar(new Term[] {}, new Term[] {}));
    assertThrows(CVC5ApiException.class,
        () -> d_solver.mkSygusGrammar(new Term[] {}, new Term[] {nullTerm}));
    assertThrows(CVC5ApiException.class,
        () -> d_solver.mkSygusGrammar(new Term[] {}, new Term[] {boolTerm}));
    assertThrows(CVC5ApiException.class,
        () -> d_solver.mkSygusGrammar(new Term[] {boolTerm}, new Term[] {intVar}));

    Solver slv = new Solver();
    Term boolVar2 = slv.mkVar(slv.getBooleanSort());
    Term intVar2 = slv.mkVar(slv.getIntegerSort());
    assertDoesNotThrow(() -> slv.mkSygusGrammar(new Term[] {boolVar2}, new Term[] {intVar2}));

    assertThrows(CVC5ApiException.class,
        () -> slv.mkSygusGrammar(new Term[] {boolVar}, new Term[] {intVar2}));
    assertThrows(CVC5ApiException.class,
        () -> slv.mkSygusGrammar(new Term[] {boolVar2}, new Term[] {intVar}));
    slv.close();
  }

  @Test void synthFun() throws CVC5ApiException
  {
    Sort nullSort = d_solver.getNullSort();
    Sort bool = d_solver.getBooleanSort();
    Sort integer = d_solver.getIntegerSort();

    Term nullTerm = d_solver.getNullTerm();
    Term x = d_solver.mkVar(bool);

    Term start1 = d_solver.mkVar(bool);
    Term start2 = d_solver.mkVar(integer);

    Grammar g1 = d_solver.mkSygusGrammar(new Term[] {x}, new Term[] {start1});
    g1.addRule(start1, d_solver.mkBoolean(false));

    Grammar g2 = d_solver.mkSygusGrammar(new Term[] {x}, new Term[] {start2});
    g2.addRule(start2, d_solver.mkInteger(0));

    assertDoesNotThrow(() -> d_solver.synthFun("", new Term[] {}, bool));
    assertDoesNotThrow(() -> d_solver.synthFun("f1", new Term[] {x}, bool));
    assertDoesNotThrow(() -> d_solver.synthFun("f2", new Term[] {x}, bool, g1));

    assertThrows(
        CVC5ApiException.class, () -> d_solver.synthFun("f3", new Term[] {nullTerm}, bool));
    assertThrows(CVC5ApiException.class, () -> d_solver.synthFun("f4", new Term[] {}, nullSort));
    assertThrows(CVC5ApiException.class, () -> d_solver.synthFun("f6", new Term[] {x}, bool, g2));

    Solver slv = new Solver();
    Term x2 = slv.mkVar(slv.getBooleanSort());
    assertDoesNotThrow(() -> slv.synthFun("f1", new Term[] {x2}, slv.getBooleanSort()));

    assertThrows(
        CVC5ApiException.class, () -> slv.synthFun("", new Term[] {}, d_solver.getBooleanSort()));
    assertThrows(CVC5ApiException.class,
        () -> slv.synthFun("f1", new Term[] {x}, d_solver.getBooleanSort()));
    slv.close();
  }

  @Test void synthInv() throws CVC5ApiException
  {
    Sort bool = d_solver.getBooleanSort();
    Sort integer = d_solver.getIntegerSort();

    Term nullTerm = d_solver.getNullTerm();
    Term x = d_solver.mkVar(bool);

    Term start1 = d_solver.mkVar(bool);
    Term start2 = d_solver.mkVar(integer);

    Grammar g1 = d_solver.mkSygusGrammar(new Term[] {x}, new Term[] {start1});
    g1.addRule(start1, d_solver.mkBoolean(false));

    Grammar g2 = d_solver.mkSygusGrammar(new Term[] {x}, new Term[] {start2});
    g2.addRule(start2, d_solver.mkInteger(0));

    assertDoesNotThrow(() -> d_solver.synthInv("", new Term[] {}));
    assertDoesNotThrow(() -> d_solver.synthInv("i1", new Term[] {x}));
    assertDoesNotThrow(() -> d_solver.synthInv("i2", new Term[] {x}, g1));

    assertThrows(CVC5ApiException.class, () -> d_solver.synthInv("i3", new Term[] {nullTerm}));
    assertThrows(CVC5ApiException.class, () -> d_solver.synthInv("i4", new Term[] {x}, g2));
  }

  @Test void addSygusConstraint() throws CVC5ApiException
  {
    Term nullTerm = d_solver.getNullTerm();
    Term boolTerm = d_solver.mkBoolean(true);
    Term intTerm = d_solver.mkInteger(1);

    assertDoesNotThrow(() -> d_solver.addSygusConstraint(boolTerm));
    assertThrows(CVC5ApiException.class, () -> d_solver.addSygusConstraint(nullTerm));
    assertThrows(CVC5ApiException.class, () -> d_solver.addSygusConstraint(intTerm));

    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.addSygusConstraint(boolTerm));
    slv.close();
  }

  @Test void addSygusAssume()
  {
    Term nullTerm = d_solver.getNullTerm();
    Term boolTerm = d_solver.mkBoolean(false);
    Term intTerm = d_solver.mkInteger(1);

    assertDoesNotThrow(() -> d_solver.addSygusAssume(boolTerm));
    assertThrows(CVC5ApiException.class, () -> d_solver.addSygusAssume(nullTerm));
    assertThrows(CVC5ApiException.class, () -> d_solver.addSygusAssume(intTerm));

    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.addSygusAssume(boolTerm));
    slv.close();
  }

  @Test void addSygusInvConstraint() throws CVC5ApiException
  {
    Sort bool = d_solver.getBooleanSort();
    Sort real = d_solver.getRealSort();

    Term nullTerm = d_solver.getNullTerm();
    Term intTerm = d_solver.mkInteger(1);

    Term inv = d_solver.declareFun("inv", new Sort[] {real}, bool);
    Term pre = d_solver.declareFun("pre", new Sort[] {real}, bool);
    Term trans = d_solver.declareFun("trans", new Sort[] {real, real}, bool);
    Term post = d_solver.declareFun("post", new Sort[] {real}, bool);

    Term inv1 = d_solver.declareFun("inv1", new Sort[] {real}, real);

    Term trans1 = d_solver.declareFun("trans1", new Sort[] {bool, real}, bool);
    Term trans2 = d_solver.declareFun("trans2", new Sort[] {real, bool}, bool);
    Term trans3 = d_solver.declareFun("trans3", new Sort[] {real, real}, real);

    assertDoesNotThrow(() -> d_solver.addSygusInvConstraint(inv, pre, trans, post));

    assertThrows(
        CVC5ApiException.class, () -> d_solver.addSygusInvConstraint(nullTerm, pre, trans, post));
    assertThrows(
        CVC5ApiException.class, () -> d_solver.addSygusInvConstraint(inv, nullTerm, trans, post));
    assertThrows(
        CVC5ApiException.class, () -> d_solver.addSygusInvConstraint(inv, pre, nullTerm, post));
    assertThrows(
        CVC5ApiException.class, () -> d_solver.addSygusInvConstraint(inv, pre, trans, nullTerm));
    assertThrows(
        CVC5ApiException.class, () -> d_solver.addSygusInvConstraint(intTerm, pre, trans, post));
    assertThrows(
        CVC5ApiException.class, () -> d_solver.addSygusInvConstraint(inv1, pre, trans, post));
    assertThrows(
        CVC5ApiException.class, () -> d_solver.addSygusInvConstraint(inv, trans, trans, post));
    assertThrows(CVC5ApiException.class, () -> d_solver.addSygusInvConstraint(inv, pre, pre, post));
    assertThrows(
        CVC5ApiException.class, () -> d_solver.addSygusInvConstraint(inv, pre, trans1, post));
    assertThrows(
        CVC5ApiException.class, () -> d_solver.addSygusInvConstraint(inv, pre, trans2, post));
    assertThrows(
        CVC5ApiException.class, () -> d_solver.addSygusInvConstraint(inv, pre, trans3, post));
    assertThrows(
        CVC5ApiException.class, () -> d_solver.addSygusInvConstraint(inv, pre, trans, trans));

    Solver slv = new Solver();
    Sort boolean2 = slv.getBooleanSort();
    Sort real2 = slv.getRealSort();
    Term inv22 = slv.declareFun("inv", new Sort[] {real2}, boolean2);
    Term pre22 = slv.declareFun("pre", new Sort[] {real2}, boolean2);
    Term trans22 = slv.declareFun("trans", new Sort[] {real2, real2}, boolean2);
    Term post22 = slv.declareFun("post", new Sort[] {real2}, boolean2);
    assertDoesNotThrow(() -> slv.addSygusInvConstraint(inv22, pre22, trans22, post22));

    assertThrows(
        CVC5ApiException.class, () -> slv.addSygusInvConstraint(inv, pre22, trans22, post22));
    assertThrows(
        CVC5ApiException.class, () -> slv.addSygusInvConstraint(inv22, pre, trans22, post22));
    assertThrows(
        CVC5ApiException.class, () -> slv.addSygusInvConstraint(inv22, pre22, trans, post22));
    assertThrows(
        CVC5ApiException.class, () -> slv.addSygusInvConstraint(inv22, pre22, trans22, post));
    slv.close();
  }

  @Test void getSynthSolution() throws CVC5ApiException
  {
    d_solver.setOption("lang", "sygus2");
    d_solver.setOption("incremental", "false");

    Term nullTerm = d_solver.getNullTerm();
    Term x = d_solver.mkBoolean(false);
    Term f = d_solver.synthFun("f", new Term[] {}, d_solver.getBooleanSort());

    assertThrows(CVC5ApiException.class, () -> d_solver.getSynthSolution(f));

    d_solver.checkSynth();

    assertDoesNotThrow(() -> d_solver.getSynthSolution(f));
    assertDoesNotThrow(() -> d_solver.getSynthSolution(f));

    assertThrows(CVC5ApiException.class, () -> d_solver.getSynthSolution(nullTerm));
    assertThrows(CVC5ApiException.class, () -> d_solver.getSynthSolution(x));

    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.getSynthSolution(f));
    slv.close();
  }

  @Test void getSynthSolutions() throws CVC5ApiException
  {
    d_solver.setOption("lang", "sygus2");
    d_solver.setOption("incremental", "false");

    Term nullTerm = d_solver.getNullTerm();
    Term x = d_solver.mkBoolean(false);
    Term f = d_solver.synthFun("f", new Term[] {}, d_solver.getBooleanSort());

    assertThrows(CVC5ApiException.class, () -> d_solver.getSynthSolutions(new Term[] {}));
    assertThrows(CVC5ApiException.class, () -> d_solver.getSynthSolutions(new Term[] {f}));

    d_solver.checkSynth();

    assertDoesNotThrow(() -> d_solver.getSynthSolutions(new Term[] {f}));
    assertDoesNotThrow(() -> d_solver.getSynthSolutions(new Term[] {f, f}));

    assertThrows(CVC5ApiException.class, () -> d_solver.getSynthSolutions(new Term[] {}));
    assertThrows(CVC5ApiException.class, () -> d_solver.getSynthSolutions(new Term[] {nullTerm}));
    assertThrows(CVC5ApiException.class, () -> d_solver.getSynthSolutions(new Term[] {x}));

    Solver slv = new Solver();
    assertThrows(CVC5ApiException.class, () -> slv.getSynthSolutions(new Term[] {x}));
    slv.close();
  }

  @Test void tupleProject() throws CVC5ApiException
  {
    Sort[] sorts = new Sort[] {d_solver.getBooleanSort(),
        d_solver.getIntegerSort(),
        d_solver.getStringSort(),
        d_solver.mkSetSort(d_solver.getStringSort())};
    Term[] elements = new Term[] {d_solver.mkBoolean(true),
        d_solver.mkInteger(3),
        d_solver.mkString("C"),
        d_solver.mkTerm(SET_SINGLETON, d_solver.mkString("Z"))};

    Term tuple = d_solver.mkTuple(sorts, elements);

    int[] indices1 = new int[] {};
    int[] indices2 = new int[] {0};
    int[] indices3 = new int[] {0, 1};
    int[] indices4 = new int[] {0, 0, 2, 2, 3, 3, 0};
    int[] indices5 = new int[] {4};
    int[] indices6 = new int[] {0, 4};

    assertDoesNotThrow(() -> d_solver.mkTerm(d_solver.mkOp(TUPLE_PROJECT, indices1), tuple));
    assertDoesNotThrow(() -> d_solver.mkTerm(d_solver.mkOp(TUPLE_PROJECT, indices2), tuple));
    assertDoesNotThrow(() -> d_solver.mkTerm(d_solver.mkOp(TUPLE_PROJECT, indices3), tuple));
    assertDoesNotThrow(() -> d_solver.mkTerm(d_solver.mkOp(TUPLE_PROJECT, indices4), tuple));

    assertThrows(CVC5ApiException.class,
        () -> d_solver.mkTerm(d_solver.mkOp(TUPLE_PROJECT, indices5), tuple));
    assertThrows(CVC5ApiException.class,
        () -> d_solver.mkTerm(d_solver.mkOp(TUPLE_PROJECT, indices6), tuple));

    int[] indices = new int[] {0, 3, 2, 0, 1, 2};

    Op op = d_solver.mkOp(TUPLE_PROJECT, indices);
    Term projection = d_solver.mkTerm(op, tuple);

    Datatype datatype = tuple.getSort().getDatatype();
    DatatypeConstructor constructor = datatype.getConstructor(0);

    for (int i = 0; i < indices.length; i++)
    {
      Term selectorTerm = constructor.getSelector(indices[i]).getSelectorTerm();
      Term selectedTerm = d_solver.mkTerm(APPLY_SELECTOR, selectorTerm, tuple);
      Term simplifiedTerm = d_solver.simplify(selectedTerm);
      assertEquals(elements[indices[i]], simplifiedTerm);
    }

    assertEquals("((_ tuple_project 0 3 2 0 1 2) (tuple true 3 \"C\" "
            + "(set.singleton \"Z\")))",
        projection.toString());
  }
}
