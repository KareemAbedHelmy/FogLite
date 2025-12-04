NODES_CONFIG = [
    # ---- Fog layer: relatively similar bandwidth, mild latency gradient ----
    {
        "name": "Fog1",
        "node_id": 0,
        "is_cloud": False,
        "mips": 3500,
        "cores": 4,
        "power_idle": 8.0,
        "power_max": 25.0,
        "base_latency": 0.010,   # 10 ms
        "uplink_mbps": 35.0,
        "downlink_mbps": 40.0,
    },
    {
        "name": "Fog2",
        "node_id": 1,
        "is_cloud": False,
        "mips": 4000,
        "cores": 4,
        "power_idle": 9.0,
        "power_max": 27.0,
        "base_latency": 0.011,   # 11 ms
        "uplink_mbps": 36.0,
        "downlink_mbps": 42.0,
    },
    {
        "name": "Fog3",
        "node_id": 2,
        "is_cloud": False,
        "mips": 4500,
        "cores": 4,
        "power_idle": 10.0,
        "power_max": 30.0,
        "base_latency": 0.012,   # 12 ms
        "uplink_mbps": 37.0,
        "downlink_mbps": 44.0,
    },
    {
        "name": "Fog4",
        "node_id": 3,
        "is_cloud": False,
        "mips": 5000,
        "cores": 6,
        "power_idle": 11.0,
        "power_max": 32.0,
        "base_latency": 0.013,   # 13 ms
        "uplink_mbps": 38.0,
        "downlink_mbps": 46.0,
    },
    {
        "name": "Fog5",
        "node_id": 4,
        "is_cloud": False,
        "mips": 5500,
        "cores": 6,
        "power_idle": 13.0,
        "power_max": 38.0,
        "base_latency": 0.015,   # 15 ms
        "uplink_mbps": 36.0,
        "downlink_mbps": 42.0,
    },
    {
        "name": "Fog6",
        "node_id": 5,
        "is_cloud": False,
        "mips": 6000,
        "cores": 8,
        "power_idle": 14.0,
        "power_max": 40.0,
        "base_latency": 0.017,   # 17 ms
        "uplink_mbps": 34.0,
        "downlink_mbps": 40.0,
    },
    {
        "name": "Fog7",
        "node_id": 6,
        "is_cloud": False,
        "mips": 6500,
        "cores": 8,
        "power_idle": 16.0,
        "power_max": 45.0,
        "base_latency": 0.019,   # 19 ms
        "uplink_mbps": 32.0,
        "downlink_mbps": 38.0,
    },
    {
        "name": "Fog8",
        "node_id": 7,
        "is_cloud": False,
        "mips": 7000,
        "cores": 8,
        "power_idle": 17.0,
        "power_max": 48.0,
        "base_latency": 0.021,   # 21 ms
        "uplink_mbps": 30.0,
        "downlink_mbps": 36.0,
    },

    # ---- Cloud datacenter: strong CPU, clearly "farther" & narrower link ----
    {
        "name": "CloudDC1",
        "node_id": 8,
        "is_cloud": True,
        "mips": 12000,        # still stronger than fog, but less extreme
        "cores": 32,
        "power_idle": 60.0,
        "power_max": 180.0,
        "base_latency": 0.30,    # 300 ms, long-haul WAN
        "uplink_mbps": 4.0,      # narrow upstream
        "downlink_mbps": 8.0,    # still decent, but not crazy
    },
]
