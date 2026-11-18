"""Require consistent predictions and a hands-away pause between utterances."""
import math


class StabilityFilter:
    def __init__(self, threshold=0.8, confirmations=3, release_seconds=0.8):
        if not 0 <= threshold <= 1 or confirmations < 1 or release_seconds <= 0:
            raise ValueError("Invalid stability settings")
        self.threshold = threshold
        self.confirmations = confirmations
        self.release_seconds = release_seconds
        self.reset()

    def reset(self):
        self.candidate = None
        self.count = 0
        self.latched = False
        self.absent_since = None

    def observe_hands(self, present, now):
        if present:
            self.absent_since = None
            return False
        self.candidate, self.count = None, 0
        if self.absent_since is None:
            self.absent_since = now
        if now - self.absent_since >= self.release_seconds:
            self.latched = False
            return True
        return False

    def update(self, label, confidence):
        if self.latched:
            return None
        if label is None or not math.isfinite(confidence) or confidence < self.threshold:
            self.candidate, self.count = None, 0
            return None
        self.count = self.count + 1 if label == self.candidate else 1
        self.candidate = label
        if self.count >= self.confirmations:
            self.latched = True
            return label
        return None
