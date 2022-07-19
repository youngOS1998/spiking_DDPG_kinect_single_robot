; Auto-generated. Do not edit!


(cl:in-package simple_laserscan-msg)


;//! \htmlinclude Spying.msg.html

(cl:defclass <Spying> (roslisp-msg-protocol:ros-message)
  ((distance
    :reader distance
    :initarg :distance
    :type cl:float
    :initform 0.0)
   (direction
    :reader direction
    :initarg :direction
    :type cl:float
    :initform 0.0))
)

(cl:defclass Spying (<Spying>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Spying>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Spying)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name simple_laserscan-msg:<Spying> is deprecated: use simple_laserscan-msg:Spying instead.")))

(cl:ensure-generic-function 'distance-val :lambda-list '(m))
(cl:defmethod distance-val ((m <Spying>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader simple_laserscan-msg:distance-val is deprecated.  Use simple_laserscan-msg:distance instead.")
  (distance m))

(cl:ensure-generic-function 'direction-val :lambda-list '(m))
(cl:defmethod direction-val ((m <Spying>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader simple_laserscan-msg:direction-val is deprecated.  Use simple_laserscan-msg:direction instead.")
  (direction m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Spying>) ostream)
  "Serializes a message object of type '<Spying>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'distance))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'direction))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Spying>) istream)
  "Deserializes a message object of type '<Spying>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'distance) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'direction) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Spying>)))
  "Returns string type for a message object of type '<Spying>"
  "simple_laserscan/Spying")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Spying)))
  "Returns string type for a message object of type 'Spying"
  "simple_laserscan/Spying")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Spying>)))
  "Returns md5sum for a message object of type '<Spying>"
  "e2ac8ba2b8d9e9c7149b80102e904708")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Spying)))
  "Returns md5sum for a message object of type 'Spying"
  "e2ac8ba2b8d9e9c7149b80102e904708")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Spying>)))
  "Returns full string definition for message of type '<Spying>"
  (cl:format cl:nil "float32 distance~%float32 direction~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Spying)))
  "Returns full string definition for message of type 'Spying"
  (cl:format cl:nil "float32 distance~%float32 direction~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Spying>))
  (cl:+ 0
     4
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Spying>))
  "Converts a ROS message object to a list"
  (cl:list 'Spying
    (cl:cons ':distance (distance msg))
    (cl:cons ':direction (direction msg))
))
