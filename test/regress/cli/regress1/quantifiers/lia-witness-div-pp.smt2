; COMMAND-LINE: --no-sygus-inst
; times out after change to skolems
; DISABLE-TESTER: unsat-core
(set-info :smt-lib-version 2.6)
(set-logic NIA)
(set-info :status unsat)
(declare-fun c_main_~x~0 () Int)
(declare-fun c_main_~y~0 () Int)
(declare-fun c_main_~z~0 () Int)
(assert (forall ((|main_#t~nondet0| Int) (|main_#t~nondet1| Int) (|main_#t~nondet2| Int) (v_subst_6 Int) (v_subst_5 Int) (v_subst_4 Int) (v_subst_3 Int) (v_subst_2 Int) (v_subst_1 Int)) (not (= (mod (+ (* 4194304 |main_#t~nondet0|) (* 4 c_main_~x~0) (* 4294967294 c_main_~y~0) c_main_~z~0 (* 4290772992 |main_#t~nondet1|) (* 4194304 |main_#t~nondet2|) (* 4194304 v_subst_6) (* 4290772992 v_subst_5) (* 4194304 v_subst_4) (* 4194304 v_subst_3) (* 4290772992 v_subst_2) (* 4194304 v_subst_1)) 4294967296) 1048576))))
(assert (not (forall ((|main_#t~nondet0| Int) (|main_#t~nondet1| Int) (|main_#t~nondet2| Int)) (not (= (mod (+ (* 4194304 |main_#t~nondet0|) (* 4 c_main_~x~0) (* 4294967294 c_main_~y~0) c_main_~z~0 (* 4290772992 |main_#t~nondet1|) (* 4194304 |main_#t~nondet2|)) 4294967296) 1048576)))))
(check-sat)
(exit)
