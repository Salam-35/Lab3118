>#  <Center> <b> Lab Report on GIT Branch & Merging </b> </Center>
---
## <Center> Rajshahi University Of Engineering & Technology <br>
### <Center>Department of Electrical & Computer Engineering 
####  <center> **Software Engineering & Information System Design** <br>  **ECE 3118**

---
---

>##  <center><b>Name:</b> Rifah Tasnia <br><b>ID:</b> 1710033 <br>  </center>
</center>

---
---
 <br><br><br><br>


# How to use github
* **First create a github account**
* **Create a git repository**
* **Create a local folder** 

  * Open git bash on local folder
  * Initialize git 
  * Create or edit a file
  * Stage that edited file
  * Check the status **[not Mandatory]**
  * Commit that file with readable message
  * Push that file on the git repository


* **Edit a previously created file**
  * First **pull** that folder from git to avoid conflict
  * Edit file
  * then add, commit, push like prevoiusly

* **Edit a file from git repository**
  * Clone that folder from git repository
  * Edit it  
     * Create ***brunch*** to work on without hampering original file
     * If any conflict arise then discard it manually by chechking
     * merge it with main brunch

  * then add, commit, push like prevoiusly
---
<br>

>## Some Software need for edit graphically
   **Source Tree        :**       https://www.sourcetreeapp.com/
<br><br><br><br><br><br><br>

# Git account Create 
># https://github.com/join

# Git Initialization

To implement version control in a folder we need to write follwing command 

```
$ git init 
```

# Staging a file

To add a file before commit <br>
**add** changes from all tracked and untracked files
```
$ git add -A
```
 

# Git Commit
It is used to record the changes in the repository. It is the next command after the **git add**.
```
$ git commit 
```

# Git status 
The status command is used to display the state of the working directory and the staging area.

```
$ git status
```
<br>

# Git push
It is used to upload local repository content to a remote repository.
```
$ git push [variable name] master  
```


# Git push -all
This command pushes all the branches to the server repository.

```
$Git push -all
```

# Git pull
Pull command is used to receive data from GitHub. It fetches and merges changes on the remote server to working directory.

```
$$ git pull URL 
```


# To update tracked files

```
$ git -u
```



# Git Clone Command
The **git clone** is a command-line utility which is used to make a local copy of a remote repository. It accesses the repository through a remote URL.
```
$ git clone <repository URL>  
```


# Cloning a Repository into a Specific Local Folder
 Git allows cloning the repository into a specific directory without switching to that particular directory. You can specify that directory as the next command-line argument in git clone command. See the below command:

 ```
$ git clone https://github.com/ImDwivedi1/Git-Example.git "Folder_name"  
```
<br>
<br>
<br>
---
---
---
># Git Branch
A branch is a version of the repository that diverges from the main working project. It is a feature available in most modern version control systems. A Git project can have more than one branch. 
* **Git Master Branch  :**
The master branch is a default branch in Git. It is instantiated when first commit made on the project.

* **Operations on Branches :** We can perform various operations on Git branches. The git branch command allows you to **create, list, rename and delete** branches. Many operations on branches are applied by git checkout and git merge command. So, the git branch is tightly integrated with the git checkout and git merge commands.

   * **Create Branch :** It can create a new branch with the help of the git branch command. This command will be used as:
   
   ```
   $ git branch  <branch name>
   ```
   * **List Branch:** It can List all of the available branches in your repository by using the following command.
   ```
   $ git branch --list  
   ```
   * **Delete Branch:** It can delete the specified branch. It is a safe operation.
   ```
   $ git branch -d<branch name>  
   ```
    * **Delete a Remote Branch:** It can delete a remote branch from Git desktop application.
   ```
   $ git push origin -delete <branch name>   
   ```
   * **Switch to master branch:** Git allows to switch between the branches without making a commit
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

># Git Merge and Merge Conflict

<p>In Git, the merging is a procedure to connect the forked history. It joins two or more development history together. The git merge command facilitates to take the data created by git branch and integrate them into a single branch. Git merge will associate a series of commits into one unified history. Generally, git merge is used to combine two branches.</p>

* **The git merge command  :**
The git merge command is used to merge the branches.
```
$ git merge <query>  
```
<br>


 The above command will merge the specified commit to the currently active branch.
```
  $ git merge <commit> 
``` 
<br>


To merge a specified commit into master, first discover its commit id. Use the log command to find the particular commit id
```
$git log
```
<br>

To merge the commits into the master branch, switch over to the master branch.
```
$ git checkout master  
```
<br>

Now, Switch to branch 'master' to perform merging operation on a commit.Use the git merge command along with master branch name. 
```
$ git merge master  
```
---
---
<br><br><br><br>

># **Git Merge Conflict  :**
When two branches are trying to merge, and both are edited at the same time and in the same file, Git won't be able to identify which version is to take for changes. Such a situation is called merge conflict. If such a situation occurs, it stops just before the merge commit so that it can resolve the conflicts manually.

  * The server knows that the file is already updated and not merged with other branches. So, the push request was rejected by the remote server. It will throw an error message like **[rejected] failed to push some refs to < remote URL>**. It will suggest to pull the repository first before the push.
  <br>


  * Git rebase command is used to pull the repository from the remote URL. Here, it will show the error message like **merge conflict in  < filename >**.

## **Resolve Conflict  :**

  * To resolve the conflict, it is necessary to know whether the conflict occurs and why it occurs. Git merge tool command is used to resolve the conflict. The merge command is used as follows:
  ```
  $ git mergetool 
  ```
  * To resolve the conflict, enter in the insert mode by merely pressing **I** key and make changes as you want. Press the **Esc** key, to come out from insert mode. Type the: w! at the bottom of the editor to save and exit the changes. To accept the changes, use the rebase command. It will be used as follows:
  ```
  $ git rebase --continue  
  ```
  * In the above output, the conflict has resolved, and the local repository is synchronized with a remote repository.

<!--    a unit stopped -->
<br>

---
---


 >### **Referrence  :**  


[1] "Git Tutorial - javatpoint", www.javatpoint.com, 2021. [Online]. Available: https://www.javatpoint.com/git. [Accessed: 02- Jul- 2021].

---
---