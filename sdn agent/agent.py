"""
agent.py — Live SOAR (Security Orchestration, Automation and Response) Loop
=============================================================================
Phase: Live Detection and Response (run AFTER train.py has produced model.joblib)

Overview:
    This script implements the operational SOAR loop for the SDN ping-flood
    detection and mitigation system. It runs continuously on PC3, which receives
    a mirrored copy of all traffic passing through the OVS bridge on the RPi 4B.

    The three SOAR responsibilities and how they are fulfilled here:
        Orchestration  → Scapy acts as the traffic sensor; the Floodlight ACL
                         REST API acts as the enforcement point on OVS.
        Automation     → The soar_loop() function classifies traffic and triggers
                         blocking with no human input required.
        Response       → block_ip() issues a DENY ACL rule via REST, which
                         Floodlight converts to an OpenFlow DROP flow entry on
                         the OVS bridge in the RPi.

    Why ACL instead of the Floodlight Firewall module:
        The Floodlight ACL module installs rules PROACTIVELY as static OpenFlow
        flow entries directly into the switch's flow table. This means a DENY rule
        takes effect immediately upon the REST POST, regardless of whether the
        controller is actively seeing new packet-in events.

        The Floodlight Firewall module operates REACTIVELY — it intercepts
        packet-in events and decides whether to allow or deny each new flow.
        Rules installed through the Firewall module also require careful priority
        management (DENY must outrank ALLOW) and require the Firewall module to
        be explicitly enabled before any rules apply.

        ACL avoids all of these complications: there is no module enable step,
        no default-deny-all behavior, and no priority conflict with pre-existing
        rules. You only add what you want to block, and everything else passes.

Physical topology (for reference):
    Data plane  : PC1 and PC2 connect to the RPi 4B via USB-to-Ethernet adapters.
                  PC3 also connects via a USB-to-Ethernet adapter; OVS on the RPi
                  mirrors ALL port traffic to PC3's adapter so PC3 receives a copy
                  of every frame forwarded by the switch.
    Control plane: OVS on the RPi connects to Floodlight on PC3 over Wi-Fi
                   (tcp:<PC3_WiFi_IP>:6653). This is an out-of-band control path
                   separate from the Ethernet data plane, which is the correct SDN
                   design — the controller never rides the network it is managing.

Usage:
    Run this script from an ELEVATED (Administrator) command prompt on PC3,
    because Scapy requires raw socket access (Npcap) on Windows.

    python agent.py --iface "Ethernet 2" --hosts 192.168.100.1 192.168.100.2 192.168.100.3

    python agent.py --list-ifaces
        Prints all Scapy-visible network interfaces so you can find the mirror NIC.

    Arguments:
        --iface          Mirror NIC name as reported by Scapy on Windows.
        --floodlight-ip  IP where Floodlight REST API is reachable (default: localhost).
        --hosts          Space-separated IPs of all hosts in the testbed. Used only
                         for logging; no ALLOW rules need to be added for ACL.
        --pc3-ip         IP of PC3 itself, excluded from sniffing. Auto-detected if
                         not provided.
        --model          Path to the trained pipeline (default: model.joblib).
        --unblock-after  Seconds before a blocked IP is automatically unblocked.
                         0 = never auto-unblock (default).
        --log            Path to the log file (default: agent.log).
        --list-ifaces    Print available Scapy interfaces and exit.

Requirements:
    pip install scapy scikit-learn joblib numpy requests
    Npcap must be installed (https://npcap.com).
    Run as Administrator.

Floodlight ACL REST endpoints used:
    POST   /wm/acl/rules/json      Add a DENY rule; returns {"ruleid": "<id>"}
    DELETE /wm/acl/rules/json      Remove one rule by ruleid
    GET    /wm/acl/clear/json      Remove ALL ACL rules (used during clean shutdown)
    GET    /wm/acl/rules/json      List all current rules (used for startup check)
"""

import argparse
import collections
import logging
import signal
import sys
import threading
import time
from datetime import datetime

import joblib
import numpy as np
import requests

# ---------------------------------------------------------------------------
# Scapy import — requires Npcap to be installed and script run as Administrator
# ---------------------------------------------------------------------------
try:
    from scapy.all import sniff, ICMP, IP, get_if_addr
    from scapy.arch.windows import get_windows_if_list
except ImportError:
    print("[ERROR] Scapy is not installed. Run: pip install scapy")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Feature definition — must match collect.py and train.py exactly.
# The sklearn Pipeline expects a vector with these 5 features in this order.
# ---------------------------------------------------------------------------
FEATURES        = ["icmp_count", "icmp_rate", "avg_pkt_size", "iat_mean", "iat_std"]

# Sliding window parameters — must match collect.py exactly.
# If these values differ from what was used during data collection, the live
# feature vectors will not match the distribution the model was trained on,
# and classification accuracy will degrade silently.
WINDOW_DURATION = 2.0   # seconds: how far back each window looks
STEP_SIZE       = 1.0   # seconds: how often the SOAR loop runs

# ---------------------------------------------------------------------------
# Global state — shared between the sniffer thread and the SOAR loop
# ---------------------------------------------------------------------------

# Packet buffer: list of (timestamp: float, src_ip: str, pkt_size: int)
# Written by the sniffer thread, read and pruned by the SOAR loop.
packet_buffer: list[tuple[float, str, int]] = []
buffer_lock   = threading.Lock()

# Blocked IPs: maps src_ip → {"ruleid": str, "blocked_at": float}
# Used to avoid re-blocking an already-blocked IP and to clean up on exit.
blocked_ips: dict[str, dict] = {}

# Set at startup from --pc3-ip or auto-detected. Packets whose source IP
# matches this are ignored — the OVS mirror reflects PC3's own frames too.
PC3_IP = ""

# Floodlight REST base URL — set at startup from --floodlight-ip argument.
REST_BASE = ""

# Logger — configured in setup_logging() and used throughout.
logger = logging.getLogger("SOAR")


# ===========================================================================
# Logging
# ===========================================================================

def setup_logging(log_path: str) -> None:
    """
    Configures logging to both stdout and a persistent log file.

    Having both outputs serves different purposes during a demo:
        - Console: real-time feedback for the operator watching the terminal.
        - Log file: timestamped audit trail showing detection latency (the time
          between when the attack starts and when the BLOCK entry appears),
          which can be presented to the instructor as evidence of system response.

    Both handlers use the same format: [HH:MM:SS] LEVEL message.
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
    logger.info(f"SOAR agent started. Logging to console and '{log_path}'.")


# ===========================================================================
# Floodlight ACL REST API helpers
# ===========================================================================

def fl_get(path: str) -> dict | list | None:
    """
    Sends an HTTP GET to the Floodlight REST API at the given path.
    Returns the parsed JSON body, or None if the request failed.

    Example: fl_get("/wm/acl/rules/json") → list of current ACL rules.
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
    Returns the parsed JSON response, or None if the request failed.

    Used to add ACL rules. Floodlight returns {"ruleid": "<id>", "status": "Rule added"}
    on success. The ruleid is stored in blocked_ips for later deletion.
    """
    url = REST_BASE + path
    try:
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error(f"POST {url} payload={payload} failed: {e}")
        return None


def fl_delete(path: str, payload: dict) -> dict | None:
    """
    Sends an HTTP DELETE with a JSON body to the Floodlight REST API.
    Returns the parsed JSON response, or None if the request failed.

    The Floodlight ACL DELETE endpoint requires a JSON body:
        {"ruleid": "<id>"}
    This is non-standard REST (DELETE with a body), but it is what
    the Floodlight ACL API specifies — requests handles it correctly.
    """
    url = REST_BASE + path
    try:
        resp = requests.delete(url, json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error(f"DELETE {url} payload={payload} failed: {e}")
        return None


# ===========================================================================
# ACL blocking and unblocking
# ===========================================================================

def block_ip(src_ip: str) -> bool:
    """
    Adds a DENY ACL rule in Floodlight to drop all ICMP traffic from src_ip.

    How it works end-to-end:
        1. This function POSTs a rule to Floodlight's REST API on PC3.
        2. Floodlight converts the ACL rule into an OpenFlow FLOW_MOD message.
        3. The FLOW_MOD is sent to OVS on the RPi over the OpenFlow control
           channel (TCP 6653 over Wi-Fi).
        4. OVS installs the flow entry. From that point on, any ICMP packet
           from src_ip is dropped IN HARDWARE by the switch — the packet never
           reaches the victim host.

    The ACL rule format for Floodlight:
        src-ip   : attacker's IP with /32 (exact host match)
        nw-proto : "ICMP" — only ICMP is blocked, not all traffic from that IP.
                   This ensures the attacker can still send non-ICMP traffic
                   (e.g., ARP, TCP), which matters for keeping the testbed
                   functional during the demo.
        action   : "deny" — Floodlight accepts "deny" or "DENY" (case-insensitive).

    The returned ruleid is stored in blocked_ips so it can be deleted later
    by unblock_ip() or clear_all_acl_rules() during shutdown.

    Returns True if the rule was successfully posted, False otherwise.
    """
    if src_ip in blocked_ips:
        # Already blocked — do not post a duplicate rule. Duplicate rules
        # would accumulate in Floodlight and require multiple DELETEs to clean up.
        logger.debug(f"[BLOCK] {src_ip} is already blocked, skipping.")
        return True

    acl_rule = {
        "src-ip"   : f"{src_ip}/32",   # /32 = exact host match (not a subnet)
        "nw-proto" : "ICMP",           # block only ICMP, not all protocols
        "action"   : "deny",           # "deny" or "DENY" — case-insensitive
    }

    logger.warning(f"[BLOCK] Attack detected from {src_ip}. Posting ACL DENY rule...")
    result = fl_post("/wm/acl/rules/json", acl_rule)

    if result is None:
        logger.error(f"[BLOCK] Failed to post ACL rule for {src_ip}. "
                     f"Check that Floodlight is running and reachable at {REST_BASE}.")
        return False

    # Floodlight ACL API returns: {"ruleid": "<integer_string>", "status": "Rule added"}
    # The ruleid is used to delete this specific rule later.
    rule_id = result.get("ruleid", "unknown")
    blocked_ips[src_ip] = {
        "ruleid"    : rule_id,
        "blocked_at": time.time(),
    }

    logger.warning(
        f"[BLOCK] *** ATTACK BLOCKED ***  src={src_ip}  "
        f"rule-id={rule_id}  "
        f"time={datetime.now().strftime('%H:%M:%S')}"
    )
    return True


def unblock_ip(src_ip: str) -> bool:
    """
    Removes the ACL DENY rule for src_ip from Floodlight by its ruleid.

    Floodlight deletes the corresponding OpenFlow flow entry from OVS when the
    ACL rule is removed, restoring normal forwarding for traffic from that IP.

    Note: There may be a brief period (up to the flow's idle/hard timeout, if any)
    during which the flow entry persists in OVS even after the REST DELETE. ACL
    flow entries typically have no timeout and are removed immediately, but this
    behavior can vary depending on the Floodlight version and OVS configuration.

    Returns True if successfully unblocked, False otherwise.
    """
    if src_ip not in blocked_ips:
        logger.warning(f"[UNBLOCK] {src_ip} is not in the blocked list.")
        return False

    rule_id = blocked_ips[src_ip].get("ruleid", "unknown")

    if rule_id == "unknown":
        logger.warning(
            f"[UNBLOCK] Cannot unblock {src_ip}: ruleid was not recorded at block time. "
            f"Use GET /wm/acl/rules/json to find and manually delete the rule."
        )
        return False

    result = fl_delete("/wm/acl/rules/json", {"ruleid": rule_id})

    if result is not None:
        del blocked_ips[src_ip]
        logger.info(f"[UNBLOCK] {src_ip} unblocked. ACL rule {rule_id} deleted.")
        return True
    else:
        logger.error(f"[UNBLOCK] Failed to delete ACL rule {rule_id} for {src_ip}.")
        return False


def clear_all_acl_rules() -> None:
    """
    Removes ALL ACL rules from Floodlight using the bulk-clear endpoint.

    This is called during shutdown (Ctrl+C) to leave the testbed in a clean
    state. Without this cleanup step, ACL DENY rules would persist in Floodlight
    across restarts, causing the victim to remain unreachable after the demo ends.

    The Floodlight ACL clear endpoint is:
        GET /wm/acl/clear/json
    Despite being a destructive operation, Floodlight exposes it as a GET.
    This is an API design quirk — we call it correctly as a GET request.

    After calling this, the blocked_ips dict is also cleared in memory so the
    script's internal state matches Floodlight's actual state.
    """
    logger.info("[CLEANUP] Clearing all ACL rules from Floodlight...")
    result = fl_get("/wm/acl/clear/json")

    if result is not None:
        blocked_ips.clear()
        logger.info("[CLEANUP] All ACL rules cleared. Testbed restored to clean state.")
    else:
        logger.error(
            "[CLEANUP] Failed to clear ACL rules via REST. "
            "Rules may persist in Floodlight. To clear manually, run: "
            f"curl {REST_BASE}/wm/acl/clear/json"
        )


# ===========================================================================
# Auto-unblock watchdog (optional)
# ===========================================================================

def auto_unblock_watchdog(unblock_after: float) -> None:
    """
    Background thread that automatically unblocks IPs after a configurable duration.

    Only active when --unblock-after > 0. Useful during demo testing when you
    want to run the attack multiple times from the same attacker IP: after the
    block expires, the ACL rule is removed and the next attack from that IP will
    trigger detection and blocking again.

    Checks every 5 seconds. Unblocks any IP whose block age exceeds unblock_after.

    Args:
        unblock_after: How many seconds a block should remain before auto-removal.
    """
    logger.info(
        f"[WATCHDOG] Auto-unblock active: blocked IPs will be released "
        f"after {unblock_after:.0f}s."
    )
    while True:
        time.sleep(5)
        now = time.time()
        # Iterate over a snapshot because unblock_ip() modifies blocked_ips
        for src_ip, info in list(blocked_ips.items()):
            age = now - info.get("blocked_at", now)
            if age >= unblock_after:
                logger.info(
                    f"[WATCHDOG] Auto-unblocking {src_ip} "
                    f"(blocked {age:.1f}s ago, limit={unblock_after:.0f}s)."
                )
                unblock_ip(src_ip)


# ===========================================================================
# Packet sniffer callback
# ===========================================================================

def packet_callback(pkt) -> None:
    """
    Called by Scapy's sniff() for every packet that passes the BPF filter.

    This function runs in the dedicated sniffer thread. It must be fast — any
    heavy processing here would cause packets to be dropped. All it does is
    append qualifying packets to the shared buffer; feature computation and
    classification happen in the main SOAR loop thread.

    Filtering applied here (in addition to the kernel-level BPF filter="icmp"):
        - Must have both an IP layer and an ICMP layer (defensive check against
          non-IP ICMP-like frames that occasionally slip through on some drivers).
        - ICMP type must be 8 (Echo Request). Type 0 (Echo Reply) is excluded
          because those are response packets traveling in the opposite direction.
          Counting replies would double-count round-trip traffic and distort the
          icmp_rate and iat_mean features for the source being classified.
        - Source IP must not equal PC3_IP. OVS with select-all=true mirrors
          PC3's own outgoing frames back to it. REST API calls from this agent
          generate TCP packets that become ICMP-unrelated, but PC3 may also
          initiate pings during testing. Filtering PC3's own IP prevents any
          self-generated ICMP from being misclassified.
    """
    if pkt.haslayer(IP) and pkt.haslayer(ICMP):
        if pkt[ICMP].type == 8 and pkt[IP].src != PC3_IP:
            ts  = time.time()
            src = pkt[IP].src
            sz  = len(pkt)          # total frame size in bytes
            with buffer_lock:
                packet_buffer.append((ts, src, sz))


# ===========================================================================
# Feature extraction
# ===========================================================================

def compute_features(pkts: list[tuple[float, int]]) -> list[float] | None:
    """
    Computes a 5-element feature vector from a list of packets for one source IP.

    This function is an exact replica of the same function in collect.py. The
    feature definitions, variable names, and edge-case handling must match
    collect.py exactly, because the sklearn pipeline was trained on data produced
    by that function. Any divergence — even a subtle one like using len(pkts)
    instead of a rounded count — would cause the live feature distribution to
    drift from the training distribution and degrade classification accuracy.

    Args:
        pkts: list of (timestamp_float, packet_size_int) tuples for one source IP,
              covering only packets within the current sliding window.

    Returns:
        List of 5 floats in FEATURES order:
            [icmp_count, icmp_rate, avg_pkt_size, iat_mean, iat_std]
        Returns None if the input list is empty.

    Notes on edge cases:
        - Single-packet window: IAT cannot be computed (need at least 2 timestamps).
          Both iat_mean and iat_std are set to 0.0 as a sentinel. This does not
          cause misclassification in practice because a window with count=1 and
          rate=0.5 pps is well within the normal traffic range.
        - Zero packets: returns None; the caller skips this source IP silently.
    """
    if not pkts:
        return None

    count    = len(pkts)
    rate     = count / WINDOW_DURATION       # packets per second

    sizes    = [sz for (_, sz) in pkts]
    avg_size = float(np.mean(sizes))

    timestamps = sorted(ts for (ts, _) in pkts)
    if len(timestamps) > 1:
        iats     = np.diff(timestamps)       # time between consecutive packets
        iat_mean = float(np.mean(iats))
        iat_std  = float(np.std(iats))
    else:
        iat_mean = 0.0
        iat_std  = 0.0

    # Return features in the exact order declared in FEATURES
    return [count, rate, avg_size, iat_mean, iat_std]


# ===========================================================================
# Startup verification
# ===========================================================================

def verify_floodlight_acl() -> None:
    """
    Confirms that the Floodlight REST API is reachable and that the ACL module
    is responding before the SOAR loop starts.

    Unlike the Floodlight Firewall module, the ACL module requires no explicit
    enable step. It is always active as long as the module is listed in
    floodlightdefault.properties:
        net.floodlightcontroller.accesscontrollist.ACLSwitchManager

    This function performs a GET on /wm/acl/rules/json as a connectivity check.
    If it fails, the most likely causes are:
        1. Floodlight is not running on PC3.
        2. Windows Firewall on PC3 is blocking inbound connections on port 8080.
        3. The --floodlight-ip argument points to the wrong address.
        4. The ACL module is not loaded in floodlightdefault.properties.

    Exits the program if Floodlight is not reachable, because the SOAR loop
    would be useless without an enforcement backend.
    """
    logger.info(f"Checking Floodlight ACL API at {REST_BASE}...")
    result = fl_get("/wm/acl/rules/json")

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

    n_existing = len(result) if isinstance(result, list) else 0
    if n_existing > 0:
        logger.warning(
            f"[STARTUP] Floodlight already has {n_existing} ACL rule(s) loaded. "
            f"These may be leftover from a previous run. "
            f"To clear them: curl {REST_BASE}/wm/acl/clear/json"
        )
    else:
        logger.info("[STARTUP] Floodlight ACL module is reachable. No pre-existing rules.")


# ===========================================================================
# Graceful shutdown
# ===========================================================================

def shutdown_handler(signum, frame) -> None:
    """
    Signal handler invoked when the user presses Ctrl+C (SIGINT) or the process
    receives SIGTERM.

    Performs an orderly shutdown:
        1. Logs the current list of blocked IPs.
        2. Calls clear_all_acl_rules() to remove every ACL rule this session added.
           This is critical — without cleanup, DENY rules persist in Floodlight
           even after this script exits, leaving the victim permanently unreachable.
        3. Exits cleanly with code 0.

    This function is registered via signal.signal() in main() rather than using
    a try/except KeyboardInterrupt block, so that it also catches SIGTERM (e.g.,
    when the process is killed by a task manager or a shell script).
    """
    print()     # newline after the ^C character printed by the terminal
    logger.info("Shutdown signal received. Performing cleanup...")

    if blocked_ips:
        logger.info(
            f"Currently blocked IPs: {list(blocked_ips.keys())}"
        )
    else:
        logger.info("No IPs were blocked in this session.")

    clear_all_acl_rules()

    logger.info("Shutdown complete. Exiting.")
    sys.exit(0)


# ===========================================================================
# Main SOAR loop
# ===========================================================================

def soar_loop(pipeline) -> None:
    """
    The core automation loop. Runs indefinitely until the process is terminated.

    Each iteration (every STEP_SIZE seconds) performs the full SOAR cycle:

        1. Collect — Snapshot packets from the shared buffer that fall within
           the last WINDOW_DURATION seconds.

        2. Extract — Group the snapshot by source IP, then compute a 5-element
           feature vector for each source using compute_features().

        3. Classify — Feed each feature vector to the loaded sklearn Pipeline.
           pipeline.predict() internally applies MinMaxScaler.transform() before
           calling KNeighborsClassifier.predict(). The result is 0 (normal) or
           1 (attack). pipeline.predict_proba() gives the confidence as the
           fraction of k=3 neighbors that voted for each class.

        4. Respond — If an IP is classified as attack (pred == 1) and is not
           already blocked, call block_ip() to insert an ACL DENY rule.

        5. Prune — Remove old entries from the packet buffer that are outside
           the 2× window horizon, preventing unbounded memory growth during
           long monitoring sessions.

    Args:
        pipeline: Loaded sklearn Pipeline (MinMaxScaler + KNN) from joblib.load().
    """
    logger.info("SOAR loop started. Monitoring for ICMP ping flood attacks.")
    logger.info(
        f"Parameters: window={WINDOW_DURATION}s | "
        f"step={STEP_SIZE}s | "
        f"features={FEATURES}"
    )

    while True:
        time.sleep(STEP_SIZE)
        now = time.time()

        # ------------------------------------------------------------------
        # Step 1 — Collect: snapshot the packet buffer under the lock.
        # The sniffer thread writes to packet_buffer concurrently, so all
        # reads and writes to it must be done inside buffer_lock.
        # ------------------------------------------------------------------
        with buffer_lock:
            # Packets that fall within the current 2-second sliding window
            window_pkts = [
                (ts, src, sz) for (ts, src, sz) in packet_buffer
                if now - ts <= WINDOW_DURATION
            ]
            # Prune entries older than 2× the window.
            # Keeping 2× (not 1×) ensures the sniffer thread can safely
            # append to the tail while the loop reads the head, without the
            # loop cutting off packets that the sniffer just added.
            packet_buffer[:] = [
                (ts, src, sz) for (ts, src, sz) in packet_buffer
                if now - ts <= WINDOW_DURATION * 2
            ]

        if not window_pkts:
            # No ICMP traffic in this window — nothing to classify
            continue

        # ------------------------------------------------------------------
        # Step 2 — Extract: group packets by source IP
        # ------------------------------------------------------------------
        per_src: dict[str, list[tuple[float, int]]] = collections.defaultdict(list)
        for (ts, src, sz) in window_pkts:
            per_src[src].append((ts, sz))

        # ------------------------------------------------------------------
        # Steps 3 & 4 — Classify and Respond, one source IP at a time
        # ------------------------------------------------------------------
        for src_ip, pkts in per_src.items():

            # Skip IPs that are already blocked. There is no point classifying
            # traffic from an IP whose ICMP packets are being dropped at the
            # switch — those packets would not appear in the mirror anyway once
            # the ACL flow entry is installed in OVS. However, there may be a
            # brief window between when we post the ACL rule and when OVS
            # installs it, during which a few more packets appear in the mirror.
            # This guard prevents redundant block_ip() calls in that window.
            if src_ip in blocked_ips:
                continue

            feature_vector = compute_features(pkts)
            if feature_vector is None:
                continue

            # pipeline.predict() applies scaler internally — pass raw values.
            # Returns array of shape (1,); index [0] gives the scalar label.
            pred = pipeline.predict([feature_vector])[0]    # 0=normal, 1=attack

            # predict_proba() returns [[prob_class0, prob_class1]].
            # prob_attack is the fraction of the k nearest neighbors that are
            # labeled as attack in the training data. With k=3, this is 0%, 33%,
            # 67%, or 100%. We show it as a confidence score in the log.
            try:
                proba      = pipeline.predict_proba([feature_vector])[0]
                conf_str   = f"{proba[1]*100:.0f}%"
            except AttributeError:
                conf_str   = "N/A"      # should not happen with KNN, but defensive

            label_str  = "ATTACK" if pred == 1 else "normal"

            # Log every classification for the demo audit trail.
            # The iat_mean is re-extracted from the feature vector by index
            # (index 3 corresponds to iat_mean in the FEATURES list).
            logger.info(
                f"  {src_ip:15s} | "
                f"count={len(pkts):4d} | "
                f"rate={feature_vector[1]:6.1f} pps | "
                f"iat_mean={feature_vector[3]:.4f}s | "
                f"→ {label_str} (conf={conf_str})"
            )

            # Step 4 — block if classified as attack
            if pred == 1:
                block_ip(src_ip)


# ===========================================================================
# Interface listing helper
# ===========================================================================

def list_interfaces() -> None:
    """
    Prints all Scapy-visible network interfaces on Windows using Npcap.

    Use the 'Name' field as the --iface argument. On Windows, interface names
    are typically in the form 'Ethernet', 'Ethernet 2', 'Wi-Fi', etc.
    The 'Description' field shows the hardware adapter name (e.g. 'USB Ethernet
    Adapter') which is useful for identifying the correct mirror NIC.
    """
    print("\nAvailable network interfaces (as seen by Scapy / Npcap):")
    print("-" * 65)
    for iface in get_windows_if_list():
        name = iface.get("name", "")
        desc = iface.get("description", "")
        ips  = iface.get("ips", [])
        print(f"  Name        : {name}")
        print(f"  Description : {desc}")
        print(f"  IPs         : {ips}")
        print()
    print("Pass the 'Name' value to --iface. Example: --iface \"Ethernet 2\"")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SDN SOAR — Live ACL-based Detection and Response Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python agent.py --list-ifaces\n"
            "  python agent.py --iface \"Ethernet 2\" "
            "--hosts 192.168.100.1 192.168.100.2 192.168.100.3\n"
            "  python agent.py --iface \"Ethernet 2\" --unblock-after 60\n"
        ),
    )
    parser.add_argument(
        "--iface", type=str, default=None,
        help="Scapy interface name for the OVS mirror port (e.g., 'Ethernet 2'). "
             "Use --list-ifaces to discover available names."
    )
    parser.add_argument(
        "--list-ifaces", action="store_true",
        help="Print all available Scapy network interface names and exit."
    )
    parser.add_argument(
        "--floodlight-ip", type=str, default="localhost",
        help="IP where Floodlight REST API is reachable (default: localhost). "
             "Use 'localhost' when Floodlight runs natively on this PC."
    )
    parser.add_argument(
        "--hosts", nargs="+", default=[], metavar="IP",
        help="(Optional) Space-separated IPs of all hosts in the testbed. "
             "Used for informational logging only; no setup steps are needed "
             "for the ACL module (unlike the Firewall module)."
    )
    parser.add_argument(
        "--pc3-ip", type=str, default=None,
        help="IP of this machine (PC3), excluded from packet capture. "
             "Auto-detected from the chosen interface if not specified."
    )
    parser.add_argument(
        "--model", type=str, default="model.joblib",
        help="Path to the trained sklearn Pipeline produced by train.py "
             "(default: model.joblib)."
    )
    parser.add_argument(
        "--unblock-after", type=float, default=0,
        help="Automatically remove ACL block rules after this many seconds. "
             "0 = never auto-unblock (default). Set > 0 to allow repeated "
             "attack demos without manual cleanup between runs."
    )
    parser.add_argument(
        "--log", type=str, default="agent.log",
        help="Path to the log file (default: agent.log)."
    )

    args = parser.parse_args()

    # Interface listing mode — print interfaces and exit before doing anything else
    if args.list_ifaces:
        list_interfaces()
        sys.exit(0)

    # Interface is required for normal operation
    if not args.iface:
        print(
            "[ERROR] --iface is required. "
            "Run with --list-ifaces to see available interface names."
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Setup logging
    # -----------------------------------------------------------------------
    setup_logging(args.log)

    # -----------------------------------------------------------------------
    # Resolve PC3's own IP for self-traffic filtering in packet_callback
    # -----------------------------------------------------------------------
    if args.pc3_ip:
        PC3_IP = args.pc3_ip
    else:
        try:
            PC3_IP = get_if_addr(args.iface)
            if not PC3_IP or PC3_IP == "0.0.0.0":
                # Fallback: use the hostname resolution path
                import socket
                PC3_IP = socket.gethostbyname(socket.gethostname())
        except Exception:
            PC3_IP = ""
        logger.info(
            f"PC3 self-IP (auto-detected, excluded from sniffing): {PC3_IP}. "
            f"Use --pc3-ip to override."
        )

    # -----------------------------------------------------------------------
    # Set the Floodlight REST base URL
    # -----------------------------------------------------------------------
    REST_BASE = f"http://{args.floodlight_ip}:8080"
    logger.info(f"Floodlight REST API base URL: {REST_BASE}")

    if args.hosts:
        logger.info(f"Known hosts in testbed: {args.hosts}")

    # -----------------------------------------------------------------------
    # Register shutdown handler — must be done before any blocking calls
    # so that Ctrl+C during startup also triggers cleanup
    # -----------------------------------------------------------------------
    signal.signal(signal.SIGINT,  shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    logger.info("Shutdown handlers registered (Ctrl+C will clear all ACL rules before exit).")

    # -----------------------------------------------------------------------
    # Verify Floodlight ACL module is reachable
    # -----------------------------------------------------------------------
    verify_floodlight_acl()

    # -----------------------------------------------------------------------
    # Load the trained model pipeline
    # -----------------------------------------------------------------------
    logger.info(f"Loading model from: {args.model}")
    try:
        pipeline = joblib.load(args.model)
    except FileNotFoundError:
        logger.error(
            f"Model file '{args.model}' not found. "
            f"Run train.py first to produce it: python train.py"
        )
        sys.exit(1)
    logger.info("Model pipeline loaded successfully (MinMaxScaler + KNN).")

    # -----------------------------------------------------------------------
    # Start the Scapy packet sniffer in a background daemon thread.
    #
    # Why a separate thread?
    #   sniff() is a blocking call — it never returns until timeout or stop_filter.
    #   Running it in a daemon thread lets the main thread execute the SOAR loop
    #   at a precise 1-second cadence, independent of packet arrival timing.
    #
    # Why daemon=True?
    #   A daemon thread is automatically terminated when the main thread exits.
    #   This ensures the sniffer thread does not prevent the process from
    #   shutting down after shutdown_handler() calls sys.exit().
    #
    # Why store=False?
    #   By default, Scapy's sniff() accumulates every captured packet in a list
    #   in memory. For a long-running monitoring session this would grow without
    #   bound. store=False discards each packet after packet_callback() returns.
    #
    # Why filter="icmp" (BPF)?
    #   A BPF (Berkeley Packet Filter) expression is evaluated at the kernel level
    #   by Npcap before the packet reaches Python. Only ICMP frames are handed to
    #   packet_callback. ARP, TCP, UDP, SSDP, and all other protocols are dropped
    #   at the driver level, keeping Python-side CPU usage very low even when
    #   the mirror port is carrying heavy non-ICMP traffic.
    # -----------------------------------------------------------------------
    logger.info(f"Starting ICMP sniffer on interface: '{args.iface}'")
    sniff_thread = threading.Thread(
        target = lambda: sniff(
            iface   = args.iface,
            filter  = "icmp",           # kernel-level BPF — only ICMP reaches Python
            prn     = packet_callback,  # called once per packet in the sniffer thread
            store   = False,            # do not accumulate packets in memory
        ),
        daemon = True,                  # dies automatically when main thread exits
        name   = "SnifferThread",
    )
    sniff_thread.start()
    logger.info("Sniffer thread started.")

    # -----------------------------------------------------------------------
    # Start the optional auto-unblock watchdog thread
    # -----------------------------------------------------------------------
    if args.unblock_after > 0:
        watchdog = threading.Thread(
            target = auto_unblock_watchdog,
            args   = (args.unblock_after,),
            daemon = True,
            name   = "WatchdogThread",
        )
        watchdog.start()
        logger.info(
            f"Auto-unblock watchdog started "
            f"(blocked IPs released after {args.unblock_after:.0f}s)."
        )

    # -----------------------------------------------------------------------
    # Run the SOAR loop — blocks the main thread indefinitely.
    # Termination is handled by the registered signal handlers (shutdown_handler),
    # which clear all ACL rules before calling sys.exit().
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SOAR agent is ACTIVE. Press Ctrl+C to stop and clear all rules.")
    logger.info("=" * 60)
    soar_loop(pipeline)
