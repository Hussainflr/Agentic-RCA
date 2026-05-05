# Root Cause Analysis Report

**Most likely root cause:** Power Failure
**Confidence:** 0.84
**Risk level:** high

## Dataset Evidence
- Rotational speed [rpm]: value=2861.00, z=7.38, percentile=1.00
- Torque [Nm]: value=4.60, z=-3.55, percentile=0.00
- Process temperature [K]: value=309.10, z=-0.61, percentile=0.31
- Torque and rotational speed combination implies abnormal power demand.

## Manual / SOP Evidence
- maintenance_sop.md: # Maintenance SOP Excerpts  ## High Torque With Elevated Tool Wear  When torque increases while tool wear is also high, maintenance should inspect cutting tools, bearings, coupling
- fault_code_reference.md: # AI4I Fault Code Reference  ## TWF - Tool Wear Failure  Tool wear failure indicates the cutting tool may have exceeded useful life. Recommended checks include visual tool inspecti

## Alternative Causes
- Sensor or process drift

## Recommended Checks
- Inspect drivetrain load
- Check motor current trend
- Verify torque sensor calibration

## Safety Notes
- Use this output as decision support, not final certification.
- Follow lockout/tagout, site safety rules, and OEM procedures.
- Confirm with physical inspection and maintenance history.

## Final Action Plan
- Place equipment in a controlled troubleshooting workflow if currently unsafe.
- Verify condition with qualified maintenance personnel before returning to service.
- Trend the top contributing sensor deviations over time.
- Document findings in CMMS or maintenance log.

_Decision-support only. Not final engineering certification or maintenance authorization. Follow site procedures, lockout/tagout, qualified inspection requirements, and OEM documentation._