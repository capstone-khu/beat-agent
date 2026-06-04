from config import ALPHA, GAMMA, DEFAULT_ACTION


class RhythmQTable:
    def __init__(self, states: list, actions: list,
                 alpha: float = ALPHA, gamma: float = GAMMA):
        self.states  = states
        self.actions = actions
        self.alpha   = alpha
        self.gamma   = gamma
        self.table: dict[str, dict[str, float]] = {
            s: {a: 0.0 for a in actions} for s in states
        }

    def best_action(self, state: str) -> str:
        q_row = self.table[state]
        if all(v == 0.0 for v in q_row.values()):
            return DEFAULT_ACTION.get(state, "POSITIVE_RHYTHM")
        return max(q_row, key=lambda a: q_row[a])

    def update(self, state: str, action: str,
               reward: float, next_state: str) -> float:
        q_current  = self.table[state][action]
        max_q_next = max(self.table[next_state].values())
        td_target  = reward + self.gamma * max_q_next
        q_new      = q_current + self.alpha * (td_target - q_current)
        self.table[state][action] = q_new
        return q_new

    def get(self, state: str, action: str) -> float:
        return self.table[state][action]

    def summary(self) -> str:
        lines = ["[Q-Table 현황]"]
        for s in self.states:
            row = "  ".join(
                f"{a}:{v:+.3f}" for a, v in self.table[s].items()
            )
            lines.append(f"  {s:6s} | {row}")
        return "\n".join(lines)
