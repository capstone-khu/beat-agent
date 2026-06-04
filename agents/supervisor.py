import json

from config import ACTION_ID, ACTION_FEEDBACK


def report_to_supervisor(
    supervisor,
    action:     str,
    curr_state: str,
    reward:     float | None,
    q_value:    float,
    measure:    int,
    meta:       dict | None = None,
) -> dict:
    payload = {
        "agent":     "rhythm",
        "measure":   measure,
        "state":     curr_state,
        "action_id": ACTION_ID.get(action, "SA-10"),
        "action":    action,
        "feedback":  ACTION_FEEDBACK.get(action, ""),
        "reward":    reward,
        "q":         round(q_value, 4),
        "meta":      meta if meta is not None else {},
    }

    if supervisor is not None:
        # supervisor.receive(payload)  # 실제 연결 시 활성화
        pass

    print(f"[Supervisor] {json.dumps(payload, ensure_ascii=False)}")
    return payload
