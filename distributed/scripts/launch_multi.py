import os
import subprocess
import time
import numpy as np

ATARI_ENV_IDS = [
    "AlienNoFrameskip-v4",
    "AmidarNoFrameskip-v4",
    "AssaultNoFrameskip-v4",
    "AsterixNoFrameskip-v4",
    "AsteroidsNoFrameskip-v4",
    "AtlantisNoFrameskip-v4",
    "BankHeistNoFrameskip-v4",
    "BattleZoneNoFrameskip-v4",
    "BeamRiderNoFrameskip-v4",
    "BerzerkNoFrameskip-v4",
    "BowlingNoFrameskip-v4",
    "BoxingNoFrameskip-v4",
    "BreakoutNoFrameskip-v4",
    "CentipedeNoFrameskip-v4",
    "ChopperCommandNoFrameskip-v4",
    "CrazyClimberNoFrameskip-v4",
    "-",
    "DemonAttackNoFrameskip-v4",
    "DoubleDunkNoFrameskip-v4",
    "EnduroNoFrameskip-v4",
    "FishingDerbyNoFrameskip-v4",
    "FreewayNoFrameskip-v4",
    "FrostbiteNoFrameskip-v4",
    "GopherNoFrameskip-v4",
    "GravitarNoFrameskip-v4",
    "HeroNoFrameskip-v4",
    "IceHockeyNoFrameskip-v4",
    "JamesbondNoFrameskip-v4",
    "KangarooNoFrameskip-v4",
    "KrullNoFrameskip-v4",
    "KungFuMasterNoFrameskip-v4",
    "MontezumaRevengeNoFrameskip-v4",
    "MsPacmanNoFrameskip-v4",
    "NameThisGameNoFrameskip-v4",
    "PhoenixNoFrameskip-v4",
    "PitfallNoFrameskip-v4",
    "PongNoFrameskip-v4",
    "PrivateEyeNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "RiverraidNoFrameskip-v4",
    "RoadRunnerNoFrameskip-v4",
    "RobotankNoFrameskip-v4",
    "SeaquestNoFrameskip-v4",
    "SkiingNoFrameskip-v4",
    "SolarisNoFrameskip-v4",
    "SpaceInvadersNoFrameskip-v4",
    "StarGunnerNoFrameskip-v4",
    "-",
    "TennisNoFrameskip-v4",
    "TimePilotNoFrameskip-v4",
    "TutankhamNoFrameskip-v4",
    "UpNDownNoFrameskip-v4",
    "VentureNoFrameskip-v4",
    "VideoPinballNoFrameskip-v4",
    "WizardOfWorNoFrameskip-v4",
    "YarsRevengeNoFrameskip-v4",
    "ZaxxonNoFrameskip-v4",]

super_exp_id = str(int(time.time()))
games = ["SkiingNoFrameskip-v4", "FrostbiteNoFrameskip-v4"]
n_games = len(games)
rnd = np.random.randint(2**31)

print("This will launch {} jobs for 5 hours on 8 nodes each.".format(n_games))
print("This will cost you a maximum of {} core-hours".format(40*32*n_games))
if not input("Is your atari config file correct? y/n:")=="y":
    exit()
if not input("Is the number of hours and nodes and tasks set correctly in your slurm script? y/n:")=="y":
    exit()
print("If you wish to proceed, copy the following number: {}".format(rnd))
x=input(">>")
if x == str(rnd):

    os.environ["super_exp_id"]=super_exp_id
    os.mkdir(os.path.join("logs", super_exp_id))
    for env_id in games:
        if not env_id =="-":
            new_shell_env = os.environ.copy()
            new_shell_env["env_id"]=env_id
            os.mkdir(os.path.join("logs", super_exp_id, env_id))
            subprocess.run("sbatch slurm/slurm_python_single.peta4".split(),env=new_shell_env)

else:
    print("Closing...")
