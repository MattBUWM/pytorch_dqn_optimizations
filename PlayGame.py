import pyboy

if __name__ == '__main__':
    game = 'roms/Super_JetPak_DX_DMG-SJPD-UKV.gbc'
    pyboy = pyboy.PyBoy(game, cgb=False)
    with open('roms/states/state1.state', "rb") as x:
        pyboy.load_state(x)
    while True:
        pyboy.tick()
        print(pyboy.memory[0xc186], pyboy.memory[0xc187], pyboy.memory[0xc211], pyboy.memory[0xc1f4], pyboy.memory[0xc248])
    pyboy.stop()
