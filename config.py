TOLERANCE_MS  = 70
REALTIME_MODE = True

STATES = ["GOOD", "EARLY", "LATE", "FAST", "SLOW"]

ACTIONS = [
    "POSITIVE_RHYTHM",
    "RHYTHM_WAIT",
    "RHYTHM_CATCH_UP",
    "TEMPO_SLOW_DOWN",
    "TEMPO_SPEED_UP",
    "CALL_SUPERVISOR",
]

ACTION_ID = {
    "POSITIVE_RHYTHM":  "SA-10",
    "RHYTHM_WAIT":      "SA-06",
    "RHYTHM_CATCH_UP":  "SA-07",
    "TEMPO_SLOW_DOWN":  "SA-08",
    "TEMPO_SPEED_UP":   "SA-09",
    "CALL_SUPERVISOR":  "SA-11",
}

ACTION_FEEDBACK = {
    "POSITIVE_RHYTHM":  "잘 하고 있습니다. 계속 유지하세요",
    "RHYTHM_WAIT":      "박자보다 일찍 연주하고 있습니다. 박자를 맞추세요",
    "RHYTHM_CATCH_UP":  "박자보다 늦게 연주하고 있습니다. 박자를 맞추세요",
    "TEMPO_SLOW_DOWN":  "템포가 빠릅니다. 속도를 늦추세요",
    "TEMPO_SPEED_UP":   "템포가 느립니다. 속도를 높이세요",
    "CALL_SUPERVISOR":  "여기서 계속 같은 문제가 발생해요. 나중에 반복 연습하면서 개선해봐요",
}

DEFAULT_ACTION = {
    "GOOD":  "POSITIVE_RHYTHM",
    "EARLY": "RHYTHM_WAIT",
    "LATE":  "RHYTHM_CATCH_UP",
    "FAST":  "TEMPO_SLOW_DOWN",
    "SLOW":  "TEMPO_SPEED_UP",
}

REWARD_TABLE = {
    "good_transition":        +1.0,
    "partial_improvement":    +0.5,
    "no_change":              -0.3,
    "supervisor_effective":   +0.8,
    "supervisor_ineffective": -0.5,
    "deterioration":          -0.8,
}

ALPHA = 0.1
GAMMA = 0.9
