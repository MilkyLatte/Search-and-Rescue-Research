import finalGame as game


env = game.Game()
env.mazeLevelOne(20, 8)
print(game.brute_force(env.map))

