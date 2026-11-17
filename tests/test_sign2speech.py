import threading
from pathlib import Path

from signbangla.sign2speech.stability import StabilityFilter
from signbangla.sign2speech import speech


def test_stable_prediction_and_release():
    gate = StabilityFilter()
    assert gate.update(1, .9) is None
    assert gate.update(1, .9) is None
    assert gate.update(1, .9) == 1
    for _ in range(10):
        assert gate.update(2, .99) is None
    gate.observe_hands(False, 1)
    gate.observe_hands(True, 1.5)
    gate.observe_hands(False, 2)
    assert not gate.observe_hands(False, 2.5)
    assert gate.observe_hands(False, 3)
    assert gate.update(1, .9) is None
    assert gate.update(1, .9) is None
    assert gate.update(1, .9) == 1


def test_uncertain_and_changed_predictions_break_streak():
    gate = StabilityFilter(confirmations=2)
    gate.update(1, .9)
    assert gate.update(1, .5) is None
    assert gate.update(1, .9) is None
    assert gate.update(2, .9) is None
    assert gate.update(2, float('nan')) is None
    assert gate.update(2, .9) is None
    assert gate.update(2, .9) == 2
    gate.reset()
    assert not gate.latched


def test_audio_cache_and_failed_generation(tmp_path, monkeypatch):
    calls = []
    class Communicate:
        def __init__(self, text, voice):
            calls.append((text, voice))
        async def save(self, path):
            Path(path).write_bytes(b'audio')
    class FakeEdge:
        class exceptions:
            class NoAudioReceived(Exception):
                pass
    FakeEdge.Communicate = Communicate
    monkeypatch.setattr(speech, 'edge_module', lambda: FakeEdge)
    first = speech.synthesize('বাবা', tmp_path)
    assert speech.synthesize('বাবা', tmp_path) == first
    assert len(calls) == 1
    assert calls[0][1] == 'bn-BD-NabanitaNeural'
    async def fail(self, path):
        Path(path).write_bytes(b'partial')
        raise RuntimeError('offline')
    monkeypatch.setattr(Communicate, 'save', fail)
    import pytest
    with pytest.raises(RuntimeError, match='offline'):
        speech.synthesize('মা', tmp_path)
    assert list(tmp_path.iterdir()) == [first]


def test_worker_nonblocking_busy_and_error(tmp_path, monkeypatch):
    monkeypatch.setattr(speech, 'edge_module', lambda: None)
    monkeypatch.setattr(speech.shutil, 'which', lambda _: '/fake/player')
    entered, release = threading.Event(), threading.Event()
    def fail(*args):
        entered.set()
        release.wait(3)
        raise RuntimeError('offline')
    monkeypatch.setattr(speech, 'synthesize', fail)
    worker = speech.SpeechWorker(tmp_path)
    try:
        assert worker.submit('বাবা')
        assert entered.wait(1)
        assert not worker.submit('মা')
        release.set()
        worker.future.result(timeout=2)
        assert worker.status.startswith('Speech error')
    finally:
        release.set()
        worker.close()
    assert not worker.submit('বাবা')
