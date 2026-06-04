from config import REWARD_TABLE

_STATE_SEVERITY = {"GOOD": 0, "EARLY": 2, "LATE": 2, "FAST": 1, "SLOW": 1}


def compute_reward(prev_state: str, curr_state: str, action: str) -> float:
    if curr_state == "GOOD":
        if prev_state == "GOOD":
            return REWARD_TABLE["no_change"]
        return REWARD_TABLE["good_transition"]

    if prev_state == "GOOD" and curr_state != "GOOD":
        return REWARD_TABLE["deterioration"]

    if action == "CALL_SUPERVISOR":
        prev_sev = _STATE_SEVERITY.get(prev_state, 2)
        curr_sev = _STATE_SEVERITY.get(curr_state, 2)
        if curr_sev < prev_sev:
            return REWARD_TABLE["supervisor_effective"]
        return REWARD_TABLE["supervisor_ineffective"]

    severe = {"EARLY", "LATE"}
    mild   = {"FAST", "SLOW"}
    if prev_state in severe and curr_state in mild:
        return REWARD_TABLE["partial_improvement"]

    if prev_state == curr_state:
        return REWARD_TABLE["no_change"]

    return REWARD_TABLE["no_change"]
