import sys
import pyboy

if __name__ == '__main__':
    game = sys.argv[1]
    pyboy = pyboy.PyBoy(game, cgb=False)
    while True:
        pyboy.tick()
    pyboy.stop()
