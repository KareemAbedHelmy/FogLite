NODES_CONFIG = [
    # --- Latency-optimised but weaker CPU ---
    {
        "name": "Fog1",
        "node_id": 0,
        "is_cloud": False,
        "mips": 3500,
        "cores": 4,
        "power_idle": 8.0,
        "power_max": 25.0,
        "base_latency": 0.006,   # very close to IoT
        "uplink_mbps": 40.0,
        "downlink_mbps": 50.0,
    },
    {
        "name": "Fog2",
        "node_id": 1,
        "is_cloud": False,
        "mips": 4000,
        "cores": 4,
        "power_idle": 9.0,
        "power_max": 27.0,
        "base_latency": 0.014,
        "uplink_mbps": 30.0,
        "downlink_mbps": 40.0,
    },

    # --- Balanced nodes (mid CPU, mid latency/BW) ---
    {
        "name": "Fog3",
        "node_id": 2,
        "is_cloud": False,
        "mips": 4500,
        "cores": 4,
        "power_idle": 10.0,
        "power_max": 30.0,
        "base_latency": 0.016,
        "uplink_mbps": 25.0,
        "downlink_mbps": 35.0,
    },
    {
        "name": "Fog4",
        "node_id": 3,
        "is_cloud": False,
        "mips": 5000,
        "cores": 6,
        "power_idle": 11.0,
        "power_max": 32.0,
        "base_latency": 0.015,
        "uplink_mbps": 35.0,
        "downlink_mbps": 45.0,
    },

    # --- Compute-strong but higher latency ---
    {
        "name": "Fog5",
        "node_id": 4,
        "is_cloud": False,
        "mips": 5500,
        "cores": 8,
        "power_idle": 13.0,
        "power_max": 38.0,
        "base_latency": 0.020,
        "uplink_mbps": 20.0,
        "downlink_mbps": 30.0,
    },
    {
        "name": "Fog6",
        "node_id": 5,
        "is_cloud": False,
        "mips": 6000,
        "cores": 8,
        "power_idle": 14.0,
        "power_max": 40.0,
        "base_latency": 0.022,
        "uplink_mbps": 25.0,
        "downlink_mbps": 35.0,
    },

    # --- Very strong CPU, but “farther” + moderate BW ---
    {
        "name": "Fog7",
        "node_id": 6,
        "is_cloud": False,
        "mips": 5800,
        "cores": 10,
        "power_idle": 25.0,
        "power_max": 65.0,
        "base_latency": 0.045,
        "uplink_mbps": 20.0,
        "downlink_mbps": 25.0,
    },
    {
        "name": "Fog8",
        "node_id": 7,
        "is_cloud": False,
        "mips": 6800,
        "cores": 10,
        "power_idle": 30.0,
        "power_max": 75.0,
        "base_latency": 0.055,
        "uplink_mbps": 22.0,
        "downlink_mbps": 30.0,
    },

    # --- Cloud datacenter (still expensive, but not useless) ---
    {
        "name": "CloudDC1",
        "node_id": 8,
        "is_cloud": True,
        "mips": 15000,
        "cores": 32,
        "power_idle": 60.0,
        "power_max": 180.0,
        "base_latency": 0.20,   # keep higher than fogs
        "uplink_mbps": 4.0,     # slightly less crippled than 3/6
        "downlink_mbps": 8.0,
    },
]
