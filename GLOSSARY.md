# Predictive Maintenance

Language for assessing the failure risk of industrial machines from sensor data, so maintenance engineers can prioritize work.

## Language

**Reading**:
One snapshot of a machine's sensor values (air temperature, process temperature, rotational speed, torque, tool wear) together with its product Type. The unit a prediction is made about.
_Avoid_: Sample, record, row, observation

**Machine failure**:
The recorded outcome that a machine failed at the time of a reading. The prediction target.
_Avoid_: Breakdown, fault, defect

**Failure mode**:
One of the five recorded causes of a Machine failure (TWF, HDF, PWF, OSF, RNF). An outcome field, never an input to a prediction.
_Avoid_: Failure type, failure class, label

**Failure risk**:
The estimated likelihood that a Machine failure is associated with a given Reading. An aid for prioritization, not a guarantee.
_Avoid_: Failure prediction, failure probability (as a promise)

**Incomplete reading**:
A Reading with one or more sensor values absent or unusable at assessment time. Absent from the training data; arises only in use.
_Avoid_: Missing row, null record
