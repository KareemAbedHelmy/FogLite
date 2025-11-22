"""
Node configuration for the fog + cloud environment.

You can later tune these values to better match:
- the table in your paper, and/or
- Azure VM sizes (B1s, B2s, etc.).

Units:
- mips: million instructions per second
- power_idle, power_max: Watts
- base_latency: seconds
"""

NODES_CONFIG = [
    # -------- Fog nodes (close to IoT) --------
    {
        "name": "Fog1",
        "is_cloud": False,
        "mips": 500.0,        # 0.5 Giga-instructions/s
        "cores": 2,
        "power_idle": 40.0,
        "power_max": 80.0,
        "base_latency": 0.010,  # 10 ms
    },
    {
        "name": "Fog2",
        "is_cloud": False,
        "mips": 600.0,
        "cores": 2,
        "power_idle": 45.0,
        "power_max": 90.0,
        "base_latency": 0.015,  # 15 ms
    },
    {
        "name": "Fog3",
        "is_cloud": False,
        "mips": 700.0,
        "cores": 4,
        "power_idle": 50.0,
        "power_max": 110.0,
        "base_latency": 0.020,  # 20 ms
    },

    # -------- Cloud datacenter node --------
    {
        "name": "CloudDC1",
        "is_cloud": True,
        "mips": 2000.0,        # 2 Giga-instructions/s
        "cores": 8,
        "power_idle": 150.0,
        "power_max": 300.0,
        "base_latency": 0.050,  # 50 ms round-trip
    },
]
