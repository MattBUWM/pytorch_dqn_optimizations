import envs.MyPyBoyEnv

if __name__ == '__main__':
    env = envs.MyPyBoyEnv.MyPyBoyEnv('roms/Super_JetPak_DX_DMG-SJPD-UKV.gbc', 'roms/Super_JetPak_DX_DMG-SJPD-UKV.gbc.state', force_gbc=False, ticks_per_action=4)
    for i in range(1000):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
