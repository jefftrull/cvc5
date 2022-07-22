; COMMAND-LINE: -i
; SCRUBBER: grep -v -E '(\(define-fun)'
; EXIT: 0

(set-option :produce-models true)
(set-option :global-declarations true)
(set-option :produce-unsat-cores true)
(set-option :produce-abducts true)
(set-logic ALL)
(push 1)
(declare-fun y () Int)
(define-fun x!0 () Bool (<= 0 y))
(assert x!0)
(declare-fun x () Int)
(declare-fun z () Int)
(define-fun x!1 () Int (+ z y x))
(define-fun x!2 () Bool (<= 0 x!1))
(get-abduct abd x!2)
