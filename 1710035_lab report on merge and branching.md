# Software Engineering & Information System Design  
# Course Code :  3117
<br>

```
Submitted by :
            Name: Abdus Salam 
            Roll: 1710035
            Dept. of ECE, RUET 
```
```
Submitted to :
            Rakibul Hassan
            Lecturer
            Dept. of ECE, RUET
```



# Documentation on How to use Git branch and merge operations

>## Used software for graphically 
   **Source Tree
        :**       https://www.sourcetreeapp.com/
    
```
can be downloaded form here
https://www.sourcetreeapp.com/
```
<br>
 
```
To work with a repository we need to clone that repository first in the local disk. To clone:
```
# Git Clone Command
The **git clone** is a command-line utility which is used to make a local copy of a remote repository. It accesses the repository through a remote URL.
```
$ git clone <repository URL>

example:
E:\Study Materials\3-1\SE>git clone https://github.com/Salam-35/Lab3118.git
```



># Git Merge and Merge Conflict

<p>Generally, git merge is used to combine two branches.In Git, the merging is a procedure to connect the forked history. It joins two or more development history together.</p>


* **The git merge command  :**
The git merge command is used to merge the branches.
```
$ git merge <query>  
```
<br>


 The above command will merge the specified commit to the currently active branch.

<br>


To merge a specified commit into master, we need the particular commit id, to find the particular commit id
```
$ git log
```
<br>

To merge the commits into the master branch, switch over to the master branch.
```
$ git checkout master  
```
<br>

Now, Switch to branch *master* to perform merging operation on a commit.Use the git merge command along with master branch name. 
```
$ git merge master  
```
---

<br><br>

# **Git Merge Conflict  :**
When two branches are trying to merge, and both are edited at the same time and in the loaction, Git won't be able to identify which version is to take for changes. Such a situation is called merge conflict. If such a situation occurs, it stops just before the merge commit so that it can resolve the conflicts manually.
<br><br>
<br>


# Git Branch

* **Git Master Branch  :**
The master branch is a default branch in Git. It is instantiated when first commit made on the project.

* **Operations on Branches :** We can perform various operations on Git branches. The git branch command allows you to **create, list, rename and delete** branches. 

   * **Create Branch :** to Create a new branch command will be used as:
   
   ```
   $ git branch  <branch name>
   ```
   * **List Branch:** command for listing all the available branch is.
   ```
   $ git branch --list  
   ```
   * **Delete Branch:** to delete a specific branch, the command is
   ```
   $ git branch -d<branch name>  
   ```
    * **Delete a Remote Branch:** to delete a remote branch from git desktop, the command is
   ```
   $ git push origin -delete <branch name>   
   ```
   * **Switch to master branch:** switch between the branches to the main branch without making a commit
   ```
   $ git branch -m master 
   ```
   * **Rename Branch:** Git allows to switch between the branches without making a commit
   ```
   $ git branch -m <old branch name><new branch name>  
   ```
   * **Merge Branch:** Git allows to merge the other branch with the currently active branch.
   ```
   $ git merge <branch name which to merge>  
   ```
 <br>
 <br>


