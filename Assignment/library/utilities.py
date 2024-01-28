from lib2to3.pgen2 import token
import requests
import pandas as pd
import pymongo

client = pymongo.MongoClient('localhost',27017)
mydb = client["githubAPI"]
Collection = mydb["popRepos"]

user = 'johnkouk469'
token = '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'

# Get info about a specific user/organization by his/her/its username
def getUser(username):
    url = "https://api.github.com/users/" + username
    r = requests.get(url = url, params = {}, auth=(user,token)) 
    response = r.json() 
    print(response["repos_url"])

# Get the public repositories of a user/organization
def getUserRepos(username):
    url = "https://api.github.com/users/" + username + "/repos"
    r = requests.get(url = url, params = {}, auth=(user,token)) 
    response = r.json() 
    print(response[0].keys())
    # for repo in response:
    #     print(repo["name"])
    #     print(repo["language"])

# Get the information of a public repository
def getRepoInfo(repoOwner, repoName):
    url = "https://api.github.com/repos/" + repoOwner + "/" + repoName
    r = requests.get(url = url, params = {}, auth=(user,token)) 
    response = r.json() 
    print(response)

# Get basic information about the most popular repositories
def getMostPopularRepositoriesInfos():
    repos = pd.read_csv("library\mostPopularRepositories.csv", sep=";")
    for _, row in repos.iterrows():
        url = "https://api.github.com/repos/" + row.RepositoryOwner + "/" + row.RepositoryName
        r = requests.get(url = url, params = {}, auth=(user,token)) 
        response = r.json()
        # Collection.insert(response) 
        if isinstance(response, list):
            Collection.insert_many(response)  
        else:
            Collection.insert_one(response)
        if "topics" in response:
            print(response["topics"])
            # Collection.insert_many(response) 

# Get the comments in issues of a public repository
def getRepoIssuesComments(repoOwner, repoName):
    url = "https://api.github.com/repos/" + repoOwner + "/" + repoName + "/issues/comments"
    r = requests.get(url = url, params = {}, auth=(user,token)) 
    response = r.json() 
    print(response)

# Get the releases of a public repository
def getRepoReleases(repoOwner, repoName):
    url = "https://api.github.com/repos/" + repoOwner + "/" + repoName + "/releases"
    r = requests.get(url = url, params = {}, auth=(user,token)) 
    response = r.json() 
    print(response)

# Get a specific commit of a public repository
def getRepoCommit(repoOwner, repoName, commitSHA):
    url = "https://api.github.com/repos/" + repoOwner + "/" + repoName + "/commits/" + commitSHA
    r = requests.get(url = url, params = {}, auth=(user,token)) 
    response = r.json() 
    print(response)

getUser("AuthEceSofteng")
getUserRepos("AuthEceSofteng")
# getRepoInfo("AuthEceSofteng", "emb-ntua-workshop")
getMostPopularRepositoriesInfos()
# getRepoIssuesComments("AuthEceSofteng", "emb-ntua-workshop")
# getRepoReleases("AuthEceSofteng", "emb-ntua-workshop")
# getRepoCommit("AuthEceSofteng", "emb-ntua-workshop", "41e03e26db38caf3d2b9c500d56be1a1327d8c84")
