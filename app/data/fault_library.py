AI4I_FAULT_LIBRARY = {
    "TWF": {
        "name": "Tool Wear Failure",
        "description": "Tool wear has exceeded a reliability threshold and may degrade machining quality.",
        "checks": ["Inspect cutting tool wear", "Confirm replacement interval", "Review tool-life counter calibration"],
    },
    "HDF": {
        "name": "Heat Dissipation Failure",
        "description": "Temperature differential and operating conditions suggest insufficient heat removal.",
        "checks": ["Inspect cooling flow", "Check blocked vents/heat exchangers", "Verify ambient temperature conditions"],
    },
    "PWF": {
        "name": "Power Failure",
        "description": "Torque and rotational speed combination implies abnormal power demand.",
        "checks": ["Inspect drivetrain load", "Check motor current trend", "Verify torque sensor calibration"],
    },
    "OSF": {
        "name": "Overstrain Failure",
        "description": "Tool wear and torque imply mechanical stress above expected operating envelope.",
        "checks": ["Inspect bearings and couplings", "Check alignment", "Review feed/load settings"],
    },
    "RNF": {
        "name": "Random Failure",
        "description": "Failure label without deterministic feature signature in the dataset.",
        "checks": ["Review maintenance history", "Inspect for intermittent electrical/mechanical issues"],
    },
}

