; COMMAND-LINE: -q
; EXPECT: sat
(set-logic ALL)
(set-info :status sat)
(declare-fun b () Int)
(declare-fun c () Int)
(declare-fun g () Int)
(declare-fun e () Int)
(declare-fun f () Int)
(assert (and (>= f 0) (>= b 0) (> g 0) (= 1 (* e b)) (= 0 (* c b)) (> 0 (div (* f b) 0)) (> (* f (div (* g b) c)) 0)))
(check-sat)
