
#!/bin/bash
#update every four days
#Author: Solomon Amos

cd ~/projects/My_python_deliberate_practice
#git init
git add .
git commit -m "update python delibrate practice)"
# git remote add origin https://github.com/udohsolomon/My_python_deliberate_practice
git push -u origin master
git config credential.helper store