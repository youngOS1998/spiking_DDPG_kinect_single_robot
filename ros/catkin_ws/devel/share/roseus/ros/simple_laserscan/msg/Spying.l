;; Auto-generated. Do not edit!


(when (boundp 'simple_laserscan::Spying)
  (if (not (find-package "SIMPLE_LASERSCAN"))
    (make-package "SIMPLE_LASERSCAN"))
  (shadow 'Spying (find-package "SIMPLE_LASERSCAN")))
(unless (find-package "SIMPLE_LASERSCAN::SPYING")
  (make-package "SIMPLE_LASERSCAN::SPYING"))

(in-package "ROS")
;;//! \htmlinclude Spying.msg.html


(defclass simple_laserscan::Spying
  :super ros::object
  :slots (_distance _direction ))

(defmethod simple_laserscan::Spying
  (:init
   (&key
    ((:distance __distance) 0.0)
    ((:direction __direction) 0.0)
    )
   (send-super :init)
   (setq _distance (float __distance))
   (setq _direction (float __direction))
   self)
  (:distance
   (&optional __distance)
   (if __distance (setq _distance __distance)) _distance)
  (:direction
   (&optional __direction)
   (if __direction (setq _direction __direction)) _direction)
  (:serialization-length
   ()
   (+
    ;; float32 _distance
    4
    ;; float32 _direction
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _distance
       (sys::poke _distance (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _direction
       (sys::poke _direction (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _distance
     (setq _distance (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _direction
     (setq _direction (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get simple_laserscan::Spying :md5sum-) "e2ac8ba2b8d9e9c7149b80102e904708")
(setf (get simple_laserscan::Spying :datatype-) "simple_laserscan/Spying")
(setf (get simple_laserscan::Spying :definition-)
      "float32 distance
float32 direction

")



(provide :simple_laserscan/Spying "e2ac8ba2b8d9e9c7149b80102e904708")


