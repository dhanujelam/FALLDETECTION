from collections import deque

class TrackingEngine:
    def __init__(self):
        self.states = {} # Stores history per Person ID

    def update(self, p_id, current_score, meta):
        if p_id not in self.states:
            self.states[p_id] = {
                "score_history": deque(maxlen=10), # Window of 10 frames
                "last_meta": {},
                "alert_sent": False
            }
        
        state = self.states[p_id]
        state["score_history"].append(current_score)
        state["last_meta"] = meta

        # Temporal Smoothing: Calculate rolling average
        avg_score = sum(state["score_history"]) / len(state["score_history"])

        # Industrial Logic: Must be high risk for at least 5/10 frames
        if avg_score > 75:
            return "CRITICAL", avg_score
        elif avg_score > 40:
            return "WARNING", avg_score
        return "NORMAL", avg_score