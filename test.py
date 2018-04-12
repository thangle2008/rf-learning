from __future__ import print_function
from utils.memory import ReplayMemory


def test_memory():
    replay = ReplayMemory(10, history_length=4)
    replay.remember(1, 0, 1)
    replay.remember(2, 1, 2)
    replay.remember(3, 2, 3)
    replay.remember(4, 3, 4)
    assert replay.size() == 4
    assert replay.get_recent_states(5) == [2, 3, 4, 5]

    replay.remember(5, 4, 5)
    replay.remember(7, 5, 6)
    replay.remember(8, 5, 6)
    replay.remember(9, 5, 6)
    assert replay.size() == 8
    assert replay.get_recent_states(10) == [7, 8, 9, 10]

    replay.remember(None, 5, 6)
    assert replay.get_recent_states(11) == [0, 0, 0, 11]

    replay.remember(5, 2, 3)
    replay.remember(9, 3, 5)
    assert replay.size() == 10

    batch = replay.sample(3)
    assert len(batch) == 3
    print(batch)


if __name__ == '__main__':
    test_memory()
    print("All tests run successfully")
