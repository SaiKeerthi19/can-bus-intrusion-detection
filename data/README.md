# CAN Bus Data Files

This directory contains real CAN bus data collected from the UW EcoCar vehicle for the EEP 595 Clock Skew-Based Intrusion Detection System project.

## Data Files

### 184.txt - ECU A Messages (0x184)
- **Description**: Arrival timestamps for CAN messages with ID 0x184
- **Source**: ECU A in the UW EcoCar vehicle
- **Frequency**: 10 Hz (nominal period T = 0.1 seconds)
- **Format**: One timestamp per line (floating point seconds)
- **Usage**: Primary target for masquerade and cloaking attack simulations

### 3d1.txt - ECU B Messages (0x3d1)  
- **Description**: Arrival timestamps for CAN messages with ID 0x3d1
- **Source**: ECU B in the UW EcoCar vehicle
- **Frequency**: 10 Hz (nominal period T = 0.1 seconds)
- **Format**: One timestamp per line (floating point seconds)
- **Usage**: Attacker ECU in masquerade attack simulations

### 180.txt - ECU C Messages (0x180)
- **Description**: Arrival timestamps for CAN messages with ID 0x180
- **Source**: ECU C in the UW EcoCar vehicle
- **Frequency**: 10 Hz (nominal period T = 0.1 seconds)
- **Format**: One timestamp per line (floating point seconds)
- **Usage**: Attacker ECU in cloaking attack simulations

## Expected Clock Characteristics

Based on the EEP 595 Project 1 analysis:

| ECU | Clock Skew (ppm) | Behavior |
|-----|------------------|----------|
| ECU A (0x184) | ~-19.2 | Fast-running clock |
| ECU B (0x3d1) | ~-1.0 | Near-nominal clock |
| ECU C (0x180) | ~-17.3 | Fast-running clock |

## Data Usage

The data files are used by the simulation framework to:

1. **Load timestamps**: `CANSimulation.load_data()`
2. **Create batches**: Group N consecutive messages for analysis
3. **Train IDS**: Establish baseline clock behavior
4. **Simulate attacks**: Replace legitimate messages with attacker messages
5. **Validate detection**: Test IDS performance under different scenarios

## File Format

Each data file contains one timestamp per line:
```
1234567890.123456
1234567890.223445
1234567890.323434
...
```

Timestamps represent the arrival time of CAN messages as measured by the receiving system, with microsecond precision.

## Data Quality

- **Completeness**: Files contain sufficient data for batch analysis (>1000 messages each)
- **Consistency**: 10 Hz periodic transmission with clock skew variations
- **Authenticity**: Real vehicle data from UW EcoCar project
- **Precision**: Microsecond-level timestamp accuracy for clock skew analysis
