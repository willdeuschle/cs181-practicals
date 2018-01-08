# CS 181 Practicals
Seed repository for practicals in cs181, s18

Welcome to CS181! This repository will house your practical assignments.

If you are reading this, the name of the repository should be 
cs181-s18-practicals-YourTeamName. 

If you don't know what any of this means, contact a TF. These instructions only apply to your private repo.

## How this repository works
This repository is a copy of a "seed" repository maintained by the TFs. 
Throughout this semester, your TFs will add practical assignment starter
material to the seed repository, and you will be responsible for copying them into your repository (using "fetch" and "merge", in git-speak). Assuming you've already cloned your repository locally, you can get the new files by:
```
git remote add seed_repo https://github.com/harvard-ml-courses/cs181-s18-practicals.git # only needs to be done once
git fetch seed_repo
git merge seed_repo/master -m "Fetched new assignment"
```
These commands (1) tell your local git repository where the seed repo is (and calls it "seed_repo"), (2) gets that repo from github.com, and (3) merges it with your local files. A final, fourth, step would be to push to your remote repository so it shows up on the web and your TFs can see it.

## Practical submission
For each assignment you will be given a number of files in this repo and some
data from Kaggle. When you complete the assignments you should always push to this repository. You should also push to this repository to collaborate with your team so that you don't lose your code. For practicals, you will submit predictions to Kaggle and writeups to Canvas.

## Finally
If you are having any trouble, please reach out to a TF. We're here to help!
