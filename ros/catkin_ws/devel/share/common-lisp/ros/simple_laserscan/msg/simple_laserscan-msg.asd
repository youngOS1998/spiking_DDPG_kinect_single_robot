
(cl:in-package :asdf)

(defsystem "simple_laserscan-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "Spying" :depends-on ("_package_Spying"))
    (:file "_package_Spying" :depends-on ("_package"))
  ))