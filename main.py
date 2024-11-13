import subprocess
import os
 
cwd = os.getcwd()

program_list = [cwd + '\\analysis.py', cwd + '\\api.py']
for program in program_list:
    subprocess.call(['python', program])
    print("Finished:" + program)

# in the temrinal
# pyinstaller main.py --onefile
# pip freeze > requirements.txt (to create an env)
# conda create --name foo_env --file requirements.txt (if new env needs to be created based on the requirement file of someone else)
    