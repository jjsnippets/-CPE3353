"""
agent.py: Live SOAR (Security Orchestration, Automation and Response) Loop

Overview:
    This script implements the operational SOAR loop for the SDN ping-flood
    detection and mitigation system.
        Orchestration   Scapy acts as the traffic sensor; the Floodlight ACL
                        REST API acts as the enforcement point on OVS.
        Automation      The soar_loop() function classifies traffic and triggers
                        blocking with no human input required.
        Response        block_ip() issues a DENY ACL rule via REST, which
                        Floodlight converts to an OpenFlow DROP flow entry on
                        the OVS bridge in the RPi.

Usage:
    Run from an from an ELEVATED (Administrator) command prompt:
        python agent.py --mirror-ip <control_ip> --hosts 192.168.100.a 192.168.100.b 192.168.100.c

    --mirror-ip     : IP address assigned to the mirror NIC. Used to locate
                      the mirror interface and exclude PC3's own traffic.
    --floodlight-ip : IP where Floodlight REST API is reachable (default: localhost).
    --hosts         : Space-separated IPs of all hosts in the testbed. Used only
                      for logging; no ALLOW rules need to be added for ACL.
    --model         : Path to the trained pipeline (default: model.joblib).
    --log           : Path to the log file (default: agent.log).

Requirements:
    pip install scapy scikit-learn joblib numpy requests
    Npcap must be installed (https://npcap.com).
    Run as Administrator.

Floodlight ACL REST endpoints used:
    POST   /wm/acl/rules/json      Add a DENY rule; returns {"ruleid": "<id>"}
    GET    /wm/acl/clear/json      Remove ALL ACL rules
    GET    /wm/acl/rules/json      List all current rules
"""

import argparse
import collections
import logging
import sys
import threading
import time
from datetime import datetime

import joblib
import numpy as np
import requests

try:
    from scapy.all import sniff, ICMP, IP
    from scapy.arch.windows import get_windows_if_list
except ImportError:
    print("[ERROR] Scapy is not installed. Run: pip install scapy")
    sys.exit(1)

# Constants
# Must match with those in other files exactly
WINDOW_DURATION = 2.0   # seconds: how far back each window looks
STEP_SIZE       = 1.0   # seconds: how often the SOAR loop runs
FEATURES        = ["icmp_count", "icmp_rate", "avg_pkt_size", "iat_mean", "iat_std"]

# Shared packet buffer
packet_buffer: list[tuple[float, str, int]] = []
buffer_lock   = threading.Lock()

# Global
blocked_ips: dict[str, dict] = {}   # maps src_ip to {"ruleid": str, "blocked_at": float}
MIRROR_IP = ""
REST_BASE = ""          # Floodlight REST base URL

# Logging
logger = logging.getLogger("SOAR")
def setup_logging(log_path: str) -> None:
    """
    Configures logging to both stdout and a persistent log file.
    """
    fmt = "[%(asctime)s] %(levelname)-7s %(message)s"
    logging.basicConfig(
        level    = logging.INFO,
        format   = fmt,
        datefmt  = "%H:%M:%S",
        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    logger.info(f"Logs also saved to '{log_path}'.")

# Floodlight ACL
def fl_get(path: str) -> dict | list | None:
    """
    Sends an HTTP GET to the Floodlight REST API at the given path.
    """
    url = REST_BASE + path
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error(f"GET {url} failed: {e}")
        return None

def fl_post(path: str, payload: dict) -> dict | None:
    """
    Sends an HTTP POST with a JSON body to the Floodlight REST API.
    """
    url = REST_BASE + path
    try:
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error(f"POST {url} payload={payload} failed: {e}")
        return None

def block_ip(src_ip: str) -> bool:
    """
    Adds a DENY ACL rule in Floodlight to drop all ICMP traffic from src_ip.
    """

    # Do not block an already blocked IP.
    if src_ip in blocked_ips:
        logger.debug(f"[BLOCK] {src_ip} is already blocked, skipping.")
        return True

    # Rule to post
    acl_rule = {
        "src-ip"   : f"{src_ip}/32",   # /32 = exact host match
        "nw-proto" : "ICMP",           # block only ICMP, not all protocols
        "action"   : "deny",
    }
    result = fl_post("/wm/acl/rules/json", acl_rule)

    # If failed to post:
    if result is None:
        logger.error(f"[BLOCK] Failed to post ACL rule for {src_ip}.\n"
                     f"Check that Floodlight is running and reachable at {REST_BASE}.")
        return False

    rule_id = result.get("ruleid", "unknown")
    blocked_ips[src_ip] = {
        "ruleid"    : rule_id,
        "blocked_at": time.time(),
    }

    # Otherwise, report success:
    logger.warning(
        f"[BLOCK] *** ATTACK BLOCKED ***  src={src_ip}  "
        f"time={datetime.now().strftime('%H:%M:%S')}\n"
    )
    return True

def clear_all_acl_rules() -> None:
    """
    Removes ALL ACL rules from Floodlight.
    """

    result = fl_get("/wm/acl/clear/json")
    logger.info("All ACL rules cleared.")

def verify_floodlight_acl() -> None:
    """
    Confirms that the Floodlight REST API is reachable.
    """
    logger.info(f"Checking Floodlight ACL API at {REST_BASE}...")
    result = fl_get("/wm/acl/rules/json")

    # If no response
    if result is None:
        logger.error(
            "Cannot reach Floodlight ACL REST API. Check:\n"
            "  1. Floodlight is running: java -jar target/floodlight.jar\n"
            f"  2. REST API is on port 8080 at {REST_BASE}\n"
            "  3. Windows Firewall allows inbound TCP on port 8080\n"
            "  4. ACL module is in floodlightdefault.properties:\n"
            "     net.floodlightcontroller.accesscontrollist.ACLSwitchManager"
        )
        sys.exit(1)

    # If ACL rules already exist
    n_existing = len(result) if isinstance(result, list) else 0
    if n_existing > 0:
        logger.warning(
            f"Floodlight already has {n_existing} ACL rule(s) loaded. "
            f"These may be leftover from a previous run. "
            f"To clear them: curl {REST_BASE}/wm/acl/clear/json"
        )
    else:
        logger.info("Floodlight ACL module is reachable. No pre-existing rules.")

def packet_callback(pkt) -> None:
    """
    Packet callback for every captured ICMP frame.
    """
    # Discard non-IP frames (raw Ethernet, ARP, etc.)
    if pkt.haslayer(IP) and pkt.haslayer(ICMP):
        # Echo Request only; ignore Echo Replies (type 0)
        # also ignore the mirror host's own traffic
        if pkt[ICMP].type == 8 and pkt[IP].src != MIRROR_IP:
            ts  = time.time()       # current timestamp in seconds (float)
            src = pkt[IP].src       # source IP address (string)
            sz  = len(pkt)          # total frame size in bytes
            with buffer_lock:
                packet_buffer.append((ts, src, sz))

def compute_features(pkts: list[tuple[float, int]]) -> list[float] | None:
    '''
    Feature computation.
    '''
    if not pkts:
        return None

    # Number of packets in this window
    count    = len(pkts)

    # Packets per second
    rate     = count / WINDOW_DURATION

    # Mean packet size in bytes
    sizes    = [sz for (_, sz) in pkts]
    avg_size = float(np.mean(sizes))

    # Mean and stddev of inter-arrival times
    timestamps = sorted(ts for (ts, _) in pkts)
    if len(timestamps) > 1:     # For 2 or more packets
        iats     = np.diff(timestamps)       # time between consecutive packets
        iat_mean = float(np.mean(iats))
        iat_std  = float(np.std(iats))
    else:                       # Otherwise, return 0
        iat_mean = 0.0
        iat_std  = 0.0

    return [count, rate, avg_size, iat_mean, iat_std]


def soar_loop(pipeline) -> None:
    """
    Main SOAR automation loop. 
     - Takes a snapshot of packets within the last WINDOW_DURATION seconds
     - Groups them by source IP
     - Computes features for each source IP
     - Feed each feature vector to the loaded sklearn Pipeline.
     - If an IP is classified as attack (pred == 1) and is not already blocked,
       call block_ip() to insert an ACL DENY rule.
    """

    logger.info("SOAR loop started. Monitoring for ICMP ping flood attacks.")

    try:
        while True:
            # Sleep until next window step
            time.sleep(STEP_SIZE)
            now = time.time()

            # Take a thread-safe snapshot of the packet buffer
            with buffer_lock:
                # Extract packets inside the current sliding window
                window_pkts = [
                    (ts, src, sz) for (ts, src, sz) in packet_buffer
                    if now - ts <= WINDOW_DURATION
                ]
                # Prune entries older than 2× the window
                packet_buffer[:] = [
                    (ts, src, sz) for (ts, src, sz) in packet_buffer
                    if now - ts <= WINDOW_DURATION * 2
                ]

            # If no ICMP traffic, then skip classification for the current window
            if not window_pkts:
                continue

            # Group packets in the window by source IP
            per_src: dict[str, list[tuple[float, int]]] = collections.defaultdict(list)
            for (ts, src, sz) in window_pkts:
                per_src[src].append((ts, sz))

            # Classification and Response per source IP
            for src_ip, pkts in per_src.items():
                # Skip IPs that are already blocked
                if src_ip in blocked_ips:
                    continue

                feature_vector = compute_features(pkts)
                if feature_vector is None:
                    continue

                # Use classifier pipeline to predict the label
                # 0=normal, 1=attack
                pred = pipeline.predict([feature_vector])[0]
                label_str  = "ATTACK" if pred == 1 else "normal"

                # Also return the probability of being classified as attack (pred=1)
                try:
                    proba      = pipeline.predict_proba([feature_vector])[0]
                    conf_str   = f"{proba[1]:.3f}"
                except AttributeError:
                    conf_str   = "N/A"

                # Live status display
                logger.info(
                    f"  {src_ip:15s} | "
                    f"count={len(pkts):4d} | "
                    f"rate={feature_vector[1]:7.2f} pps | "
                    f"avg_size={feature_vector[2]:7.2f}B | "
                    f"iat_mean={feature_vector[3]:.4f}s | "
                    f"iat_std={feature_vector[4]:.4f}s | "
                    f"→ {label_str} (conf={conf_str})"
                )

                # Block if classified as attack
                if pred == 1:
                    block_ip(src_ip)
    
    except KeyboardInterrupt:
        print()
        logger.info("\nTerminated by user!\n")

    finally:
        if blocked_ips:
            logger.info(
                f"Currently blocked IPs: {list(blocked_ips.keys())}"
            )
        else:
            logger.info("No IPs were blocked in this session.")

        clear_all_acl_rules()

        logger.info("Shutdown complete.")

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SDN SOAR Agent: Live ACL-based Detection and Response Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mirror-ip", type=str, required=True,
           help="IP address assigned to the mirror NIC"
    )
    parser.add_argument(
        "--floodlight-ip", type=str, default="localhost",
        help="IP where Floodlight REST API is reachable (default: localhost)."
    )
    parser.add_argument(
        "--hosts", nargs="+", default=[], metavar="IP",
        help="Space-separated IPs of all hosts in the testbed."
    )
    parser.add_argument(
        "--model", type=str, default="model.joblib",
        help="Path to the trained sklearn Pipeline produced by train.py "
             "(default: model.joblib)."
    )
    parser.add_argument(
        "--log", type=str, default="agent.log",
        help="Path to the log file (default: agent.log)."
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log)

    # --mirror-ip should exist
    MIRROR_IP = args.mirror_ip
    logger.info(f"Mirror NIC IP (excluded from sniffing): {MIRROR_IP}")

    resolved_iface = None
    for iface in get_windows_if_list():
        name = iface.get("name", "")
        ips = iface.get("ips", [])

        normalized_ips = {ip.split("/")[0] for ip in ips}
        if MIRROR_IP in normalized_ips:
            resolved_iface = name
            break

    if not resolved_iface:
        logger.error(f"Could not find a Scapy interface that has IP {MIRROR_IP}")
        sys.exit(1)

    # Floodlight REST base URL
    REST_BASE = f"http://{args.floodlight_ip}:8080"
    logger.info(f"Floodlight REST API base URL: {REST_BASE}")

    # Hosts in the testbed
    if args.hosts:
        logger.info(f"Known hosts in testbed: {args.hosts}")

    # Verify Floodlight ACL REST API is reachable
    verify_floodlight_acl()

    # Load the trained model pipeline
    logger.info(f"Loading model from: {args.model}")
    try:
        pipeline = joblib.load(args.model)
    except FileNotFoundError:
        logger.error(
            f"Model file '{args.model}' not found."
        )
        sys.exit(1)
    logger.info("Model pipeline loaded successfully (MinMaxScaler + KNN).")

    # Sniffer thread in the background
    logger.info(f"Starting ICMP sniffer on interface: '{resolved_iface}'")
    sniff_thread = threading.Thread(
        target = lambda: sniff(
            iface   = resolved_iface,   # Scapy interface name
            filter  = "icmp",           # Non-ICMP frames are dropped
            prn     = packet_callback,  # Function called for each captured packet
            store   = False,            # do not accumulate packets in memory
        ),
        daemon = True,                  # thread dies when main thread exits
        name   = "SnifferThread",
    )
    sniff_thread.start()
    logger.info("Sniffer thread started.")

    # Main SOAR loop
    soar_loop(pipeline)
