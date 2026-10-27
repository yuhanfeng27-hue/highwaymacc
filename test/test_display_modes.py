import pytest

from benchmarl.environments.myenv.common import MyCustomEnv2


@pytest.mark.parametrize('mode', [None, 'human', 'rgb_array'])
def test_display_and_debug_configuration(mode, monkeypatch, capsys):
    monkeypatch.setenv('SDL_VIDEODRIVER', 'dummy')
    monkeypatch.setenv('SDL_AUDIODRIVER', 'dummy')
    env = MyCustomEnv2(config={'render_mode': mode, 'duration': 1})
    try:
        assert env.render_mode == mode
        assert env.env.unwrapped.render_mode == mode
        assert env.env.unwrapped.config['debug_print'] is False
        assert env.env.unwrapped.config['real_time_rendering'] == (mode == 'human')
        env.reset(seed=1)
        env.step({agent: 1 for agent in env.agents})
        assert 'ActualDist' not in capsys.readouterr().out
        if mode is None:
            assert env.env.unwrapped.viewer is None
        elif mode == 'human':
            assert env.env.unwrapped.viewer is not None
        else:
            assert env.render().shape == (400, 1600, 3)
    finally:
        env.close()
    assert env.env.unwrapped.viewer is None


def test_debug_print_can_be_enabled_without_rendering(capsys):
    env = MyCustomEnv2(config={'debug_print': True})
    try:
        env.reset(seed=1)
        env.step({agent: 1 for agent in env.agents})
        assert 'ActualDist' in capsys.readouterr().out
        assert env.env.unwrapped.viewer is None
    finally:
        env.close()
