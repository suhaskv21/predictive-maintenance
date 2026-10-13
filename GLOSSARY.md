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
A Reading with one or more of its six inputs (the five sensor values and the product Type) absent or unusable at assessment time. An unusable value (not a number, not finite, or physically impossible) counts as absent. Absent from the training data; arises only in use.
_Avoid_: Missing row, null record

**Refused reading**:
An Incomplete reading for which the app declines to give a Failure risk or a Flag, because too much of the evidence is absent for the result to be trusted. The app says which inputs are needed.
_Avoid_: Rejected row, dropped record, invalid reading

**Recoverable refusal**:
A Refused reading whose absent inputs are ones a measured imputation strategy would have scored acceptably. Counting these tells the team how much imputation would win back, and so when it is worth building.
_Avoid_: Fixable row, salvageable record

## Acting on risk

**Flag**:
The call that a Reading's Failure risk is high enough to warrant an inspection. The Failure risk at or above the Operating threshold.
_Avoid_: Alarm, alert, positive, prediction of failure

**Missed failure**:
A Machine failure whose Reading was not Flagged. Costs unplanned downtime.
_Avoid_: False negative (in engineer-facing text)

**False alarm**:
A Flag on a Reading that did not end in a Machine failure. Costs one unnecessary inspection.
_Avoid_: False positive (in engineer-facing text), false flag

**Cost ratio**:
How many False alarms one Missed failure is worth. An assumption set by maintenance management, not measured from the data.
_Avoid_: Cost weight, penalty

**Operating threshold**:
The Failure risk at or above which a Reading is Flagged, chosen to minimise expected cost at the Cost ratio. Specific to each model.
_Avoid_: Cutoff, decision boundary, probability threshold

**Risk band**:
One of three named ranges of Failure risk: Low (not Flagged), Elevated (Flagged, but not at the lower Cost ratio considered), High (Flagged under every Cost ratio considered).
_Avoid_: Risk level, tier, severity, category

**Risk floor**:
The lowest Failure risk shown as a number; anything below is shown as "under" the floor, because the models cannot resolve very small risks.
_Avoid_: Minimum probability, zero risk
